#!/usr/bin/env python3
"""Independent, target-blind verification of recovered PathVQA answer intent."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pathvqa_answer_intent_judge import request_json_object


PROMPT_VERSION = "pathvqa_yesno_answer_intent_verifier_v1"
DEFAULT_MODEL = "gpt-4.1-2025-04-14"
SYSTEM_PROMPT = """You verify an answer-intent extraction; you do not solve the medical question.
You receive a binary QUESTION, another model's COMPLETION, and a PROPOSED_INTENT (yes or no) with
its quoted evidence. Accept only when the completion makes one unambiguous commitment consistent
with that intent. Reject when the completion contains a material yes/no contradiction, does not
actually answer the binary question, or the quoted evidence does not support the intent. Do not use
outside medical knowledge. A final answer tag does not override a directly contradictory reasoning
conclusion.

Return exactly one JSON object:
{"verdict":"accept|reject","contradiction_span":"exact substring or empty","reason_code":"consistent|material_contradiction|not_an_answer|unsupported_evidence"}

For accept, contradiction_span must be empty and reason_code must be consistent. For reject due to
a contradiction or unsupported evidence, quote a non-empty exact substring from COMPLETION. For
not_an_answer, contradiction_span may be empty.
"""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def cache_key(question: str, completion: str, proposed: dict[str, str], model: str) -> str:
    return hashlib.sha256(canonical_json({
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "question": question,
        "completion": completion,
        "proposed": proposed,
        "model": model,
    }).encode()).hexdigest()


def parse_verdict(content: str, completion: str) -> dict[str, str]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"verdict", "contradiction_span", "reason_code"}:
        raise ValueError("invalid verifier field set")
    verdict = str(value["verdict"]).lower()
    span = value["contradiction_span"]
    reason = str(value["reason_code"])
    if verdict not in {"accept", "reject"} or reason not in {
        "consistent", "material_contradiction", "not_an_answer", "unsupported_evidence"
    }:
        raise ValueError("invalid verifier enum")
    if not isinstance(span, str) or len(span) > len(completion) or (span and span not in completion):
        raise ValueError("invalid verifier evidence span")
    if verdict == "accept" and (span or reason != "consistent"):
        raise ValueError("invalid accept verdict")
    if verdict == "reject" and reason == "consistent":
        raise ValueError("invalid reject verdict")
    if verdict == "reject" and reason != "not_an_answer" and not span:
        raise ValueError("reject verdict lacks exact evidence")
    return {"verdict": verdict, "contradiction_span": span, "reason_code": reason}


def build_payload(model: str, question: str, completion: str, proposed: dict[str, str]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"QUESTION:\n{question}\n\nCOMPLETION:\n{completion}\n\n"
                f"PROPOSED_INTENT: {proposed['answer']}\n"
                f"QUOTED_EVIDENCE: {proposed['evidence_span']}"
            )},
        ],
        "temperature": 0,
        "max_tokens": 128,
        "response_format": {"type": "json_object"},
        "stream": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--intent-judgments", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-paid-requests", required=True, type=int)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=60)
    args = parser.parse_args()
    api_key = os.getenv("AIGCBEST_API_KEY")
    if not api_key:
        raise RuntimeError("AIGCBEST_API_KEY is not set")
    predictions = {row["index"]: row for row in map(json.loads, args.predictions.read_text(encoding="utf-8").splitlines())}
    judgments = list(map(json.loads, args.intent_judgments.read_text(encoding="utf-8").splitlines()))
    existing = {}
    if args.output.exists():
        existing = {row["cache_key"]: row for row in map(json.loads, args.output.read_text(encoding="utf-8").splitlines())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    physical = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for source in judgments:
            proposed = source["judgment"]
            if proposed["answer"] not in {"yes", "no"}:
                continue
            row = predictions[source["index"]]
            key = cache_key(row["question"], row["completion"], proposed, args.model)
            if key in existing:
                continue
            remaining = args.max_paid_requests - physical
            if remaining <= 0:
                break
            verdict, metadata = request_json_object(
                api_key=api_key,
                model=args.model,
                payload=build_payload(args.model, row["question"], row["completion"], proposed),
                parse=lambda content: parse_verdict(content, row["completion"]),
                timeout_seconds=args.timeout_seconds,
                max_attempts=min(args.max_attempts, remaining),
            )
            physical += len(metadata["attempts"])
            result = {
                "schema_version": 1,
                "index": source["index"],
                "source_record_sha256": row.get("source_record_sha256"),
                "completion_sha256": hashlib.sha256(row["completion"].encode()).hexdigest(),
                "cache_key": key,
                "requested_model": args.model,
                "proposed_judgment": proposed,
                "verdict": verdict,
                "metadata": metadata,
                "reference_answer_sent": False,
                "image_sent": False,
                "model_identity_sent": False,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            existing[key] = result
    print(json.dumps({"status": "completed", "physical_requests_this_run": physical, "cached_or_completed_rows": len(existing)}))


if __name__ == "__main__":
    main()
