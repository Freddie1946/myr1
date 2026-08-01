#!/usr/bin/env python3
"""Resumable LLM semantic judge for PathVQA free-form answers only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from external_vqa_contract import extract_pathvqa_answer, normalize_short_answer


API_URL = "https://api2.aigcbest.top/v1/chat/completions"
DEFAULT_MODEL = "claude-sonnet-4-6"
PROMPT_VERSION = "pathvqa_free_form_semantic_judge_v1"
SYSTEM_PROMPT = """You are a strict medical visual-question-answer evaluator.
Decide whether the CANDIDATE adequately and correctly answers the QUESTION, using the REFERENCE
ANSWER as the expected semantic content. The reference can have synonymous valid phrasings, so do
not require exact wording. Reject a candidate that is merely topically related, omits the requested
fact, contradicts the reference, or adds a material false claim that undermines the answer.

Return exactly one JSON object with these fields:
{"correct": boolean, "reason": string, "error_type": string}
The reason must be at most 120 characters. error_type must be one of: correct, omission,
contradiction, related_but_not_answer, materially_false_extra_claim, unclear.

Examples:
QUESTION: how are the histone subunits charged?
REFERENCE: positively charged
CANDIDATE: Each histone subunit is positively charged.
OUTPUT: {"correct":true,"reason":"The candidate states the same charge as the reference.","error_type":"correct"}

QUESTION: what is aids?
REFERENCE: acquired immunodeficiency syndrome
CANDIDATE: AIDS is a condition related to the immune system and CD4-positive lymphocytes.
OUTPUT: {"correct":false,"reason":"It does not supply the requested expansion of AIDS.","error_type":"omission"}
"""
ERROR_TYPES = {
    "correct", "omission", "contradiction", "related_but_not_answer",
    "materially_false_extra_claim", "unclear",
}
ALLOWED_SERVED_MODELS = {
    "gpt-4.1-mini": {"gpt-4.1-mini", "gpt-4.1-mini-2025-04-14"},
    "gpt-5-mini": {"gpt-5-mini", "gpt-5-mini-2025-08-07"},
    "claude-sonnet-4-6": {"claude-sonnet-4-6"},
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def cache_key(question: str, reference: str, candidate: str, model: str) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "question": question,
        "reference": reference,
        "candidate": candidate,
        "model": model,
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def parse_judgment(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"correct", "reason", "error_type"}:
        raise ValueError("PathVQA judge returned an invalid field set")
    if type(value["correct"]) is not bool:
        raise ValueError("PathVQA judge correct field is not boolean")
    # The prompt requests 120 characters. Accept a bounded overrun so a valid
    # semantic verdict is not discarded solely for gateway/model verbosity.
    if not isinstance(value["reason"], str) or len(value["reason"]) > 240:
        raise ValueError("PathVQA judge reason is invalid")
    if value["error_type"] not in ERROR_TYPES:
        raise ValueError("PathVQA judge error_type is invalid")
    if value["correct"] != (value["error_type"] == "correct"):
        raise ValueError("PathVQA judge boolean and error_type disagree")
    return value


def served_model_matches(requested: str, served: Any) -> bool:
    if served is None:
        return True
    return str(served) in ALLOWED_SERVED_MODELS.get(requested, {requested})


def semantic_inputs(row: dict[str, Any]) -> tuple[str, str, str, str]:
    """Read either an original external-VQA row or the hosted smoke schema."""

    if all(key in row for key in ("question", "answer", "answer_type")):
        candidate, source = extract_pathvqa_answer(row["completion"], str(row["answer_type"]))
        return str(row["question"]), str(row["answer"]), candidate, source
    score = row.get("score")
    semantic = score.get("semantic_judge_input") if isinstance(score, dict) else None
    if not isinstance(semantic, dict) or set(semantic) != {"question", "reference", "candidate"}:
        raise ValueError("prediction row has no recognized PathVQA semantic-judge input")
    source = str(score.get("contract_aligned_answer_source", "hosted_score_record"))
    return (
        str(semantic["question"]), str(semantic["reference"]),
        str(semantic["candidate"]), source,
    )


def request_judgment(
    *, api_key: str, model: str, question: str, reference: str, candidate: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = (
        f"QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\n"
        f"CANDIDATE ANSWER:\n{candidate}"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 160,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    if model == "gpt-5-mini":
        # GPT-5 mini is a reasoning model. Keep the comparison inexpensive and
        # deterministic while leaving enough room for billed reasoning tokens.
        payload.pop("temperature")
        payload.pop("max_tokens")
        payload["reasoning_effort"] = "minimal"
        payload["max_completion_tokens"] = 256
    request = urllib.request.Request(
        API_URL,
        data=canonical_json(payload).encode(),
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        body = json.loads(response.read().decode("utf-8"))
    latency = time.monotonic() - started
    if not served_model_matches(model, body.get("model")):
        raise ValueError(f"served model differs: {body.get('model')!r}")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("PathVQA judge response must contain one choice")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("PathVQA judge response has no text content")
    judgment = parse_judgment(content)
    metadata = {
        "id": body.get("id"),
        "served_model": body.get("model"),
        "usage": body.get("usage"),
        "latency_seconds": latency,
    }
    return judgment, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-paid-requests", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=90)
    parser.add_argument(
        "--indices",
        help="Optional comma-separated prediction indices for calibration/smoke",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_paid_requests <= 0:
        raise ValueError("--max-paid-requests must be positive")
    api_key = os.getenv("AIGCBEST_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("AIGCBEST_API_KEY is not set")
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines()]
    selected_indices = None
    if args.indices:
        selected_indices = {int(value.strip()) for value in args.indices.split(",")}
    existing: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            existing[row["cache_key"]] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paid = 0
    with args.output.open("a", encoding="utf-8") as output:
        for row in rows:
            if selected_indices is not None and int(row["index"]) not in selected_indices:
                continue
            answer_type = str(row.get("answer_type") or row.get("score", {}).get("answer_type"))
            if answer_type != "free_form":
                continue
            question, answer, candidate, source = semantic_inputs(row)
            key = cache_key(question, answer, candidate, args.model)
            if key in existing:
                continue
            if paid >= args.max_paid_requests:
                break
            judgment, metadata = request_judgment(
                api_key=api_key,
                model=args.model,
                question=question,
                reference=answer,
                candidate=candidate,
                timeout_seconds=args.timeout_seconds,
            )
            record = {
                "schema_version": 1,
                "prompt_version": PROMPT_VERSION,
                "cache_key": key,
                "index": int(row["index"]),
                "source_record_sha256": row.get("source_record_sha256"),
                "answer_type": "free_form",
                "candidate": candidate,
                "candidate_extraction_source": source,
                "normalized_reference": normalize_short_answer(answer),
                "judgment": judgment,
                "model": args.model,
                **metadata,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            existing[key] = record
            paid += 1
    judged = list(existing.values())
    correct = sum(bool(row["judgment"]["correct"]) for row in judged)
    print(json.dumps({
        "status": "completed_request_cap",
        "paid_requests_this_run": paid,
        "judged_free_form_total": len(judged),
        "semantic_correct": correct,
        "semantic_accuracy": correct / len(judged) if judged else None,
        "model": args.model,
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
