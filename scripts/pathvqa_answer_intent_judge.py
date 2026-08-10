#!/usr/bin/env python3
"""Target-blind LLM extraction for deterministic-unresolved PathVQA yes/no outputs."""

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

from external_vqa_contract import normalize_short_answer, pathvqa_score
from pathvqa_llm_judge import (
    API_URL,
    JudgeRequestFailure,
    TRANSIENT_HTTP_STATUSES,
    canonical_json,
    served_model_matches,
)


PROMPT_VERSION = "pathvqa_yesno_answer_intent_extractor_v1"
DEFAULT_MODEL = "gpt-4.1-mini-2025-04-14"
SYSTEM_PROMPT = """You extract the answer intended by another model; you do not solve the medical question.
You receive a binary QUESTION and that model's COMPLETION. Determine whether the completion clearly
intends Yes, clearly intends No, or does not provide one unambiguous final answer. Do not use outside
medical knowledge. A cut-off argument, contradictory conclusion, list of possibilities, or text with
no clear commitment is unresolved. Negated language can express an answer even without the literal
words Yes or No. A statement that the completion does not discuss, mention, or answer the feature is
not evidence that the feature is absent; classify that as unresolved/no_answer.

Return exactly one JSON object:
{"answer":"yes|no|unresolved","evidence_span":"exact substring","reason_code":"explicit_final|entailed_conclusion|ambiguous|no_answer"}

For yes or no, evidence_span must be a non-empty exact substring copied from COMPLETION that directly
supports the intended answer. For unresolved, evidence_span may be empty. explicit_final and
entailed_conclusion are valid only with yes/no; ambiguous and no_answer are valid only with unresolved.

Example: QUESTION: Does the sample contain the feature? COMPLETION: The description makes no statement
about that feature. OUTPUT: {"answer":"unresolved","evidence_span":"makes no statement about that feature","reason_code":"no_answer"}
"""
ANSWERS = {"yes", "no", "unresolved"}
REASON_CODES = {"explicit_final", "entailed_conclusion", "ambiguous", "no_answer"}


def cache_key(question: str, completion: str, model: str) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "question": question,
        "completion": completion,
        "model": model,
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def parse_intent_judgment(content: str, completion: str) -> dict[str, str]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    value = json.loads(text)
    required = {"answer", "evidence_span", "reason_code"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("intent judge returned an invalid field set")
    answer = str(value["answer"]).lower()
    evidence = value["evidence_span"]
    reason = str(value["reason_code"])
    if answer not in ANSWERS or reason not in REASON_CODES:
        raise ValueError("intent judge returned an invalid enum")
    # Exact-substring validation already bounds the evidence by the completion.
    # Do not reject a valid intent merely because the gateway copied a long
    # answer sentence (PathVQA questions and generated answer tags can exceed
    # an arbitrary 240-character cutoff).
    if not isinstance(evidence, str) or len(evidence) > len(completion):
        raise ValueError("intent judge evidence span is invalid")
    if answer in {"yes", "no"}:
        if reason not in {"explicit_final", "entailed_conclusion"}:
            raise ValueError("resolved answer has an unresolved reason code")
        if not evidence or evidence not in completion:
            raise ValueError("resolved evidence is not an exact completion substring")
    else:
        if reason not in {"ambiguous", "no_answer"}:
            raise ValueError("unresolved answer has a resolved reason code")
        if evidence and evidence not in completion:
            raise ValueError("unresolved evidence is not an exact completion substring")
    return {"answer": answer, "evidence_span": evidence, "reason_code": reason}


def build_payload(model: str, question: str, completion: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"QUESTION:\n{question}\n\nCOMPLETION:\n{completion}",
            },
        ],
        "temperature": 0,
        "max_tokens": 128,
        "response_format": {"type": "json_object"},
        "stream": False,
    }


def request_json_object(
    *, api_key: str, model: str, payload: dict[str, Any], parse,
    timeout_seconds: float, max_attempts: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Shared fail-closed JSON request helper for intent extraction and verification."""
    attempts: list[dict[str, Any]] = []
    for attempt_number in range(1, max_attempts + 1):
        if attempt_number > 1:
            time.sleep(2 if attempt_number == 2 else 5)
        request = urllib.request.Request(
            API_URL, data=canonical_json(payload).encode(), method="POST",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8", errors="replace")
                status = int(response.status)
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            attempts.append({"attempt": attempt_number, "http_status": status, "error": f"HTTP {status}"})
            if status in TRANSIENT_HTTP_STATUSES and attempt_number < max_attempts:
                continue
            raise JudgeRequestFailure(f"JSON judge HTTP {status}", attempts=attempts, terminal=status not in TRANSIENT_HTTP_STATUSES) from exc
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            attempts.append({"attempt": attempt_number, "http_status": None, "error": f"{type(exc).__name__}: {exc}"})
            if attempt_number < max_attempts:
                continue
            raise JudgeRequestFailure(f"JSON judge transport failed: {exc}", attempts=attempts, terminal=False) from exc
        attempt: dict[str, Any] = {"attempt": attempt_number, "http_status": status, "latency_seconds": time.monotonic() - started}
        try:
            body = json.loads(raw)
            if not served_model_matches(model, body.get("model")):
                raise RuntimeError(f"served model differs: {body.get('model')!r}")
            choices = body.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("response does not contain exactly one choice")
            choice = choices[0]
            if choice.get("finish_reason") != "stop":
                raise ValueError(f"unexpected finish reason: {choice.get('finish_reason')!r}")
            content = choice.get("message", {}).get("content")
            if not isinstance(content, str):
                raise ValueError("response has no text content")
            value = parse(content)
        except RuntimeError as exc:
            attempt.update({"error": str(exc), "raw_response": raw[:4000]})
            attempts.append(attempt)
            raise JudgeRequestFailure(str(exc), attempts=attempts, terminal=True) from exc
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            attempt.update({"error": f"{type(exc).__name__}: {exc}", "raw_response": raw[:4000]})
            attempts.append(attempt)
            if attempt_number < max_attempts:
                continue
            raise JudgeRequestFailure("JSON judge response validation failed", attempts=attempts, terminal=False) from exc
        attempt["status"] = "completed"
        attempts.append(attempt)
        return value, {"request_id": body.get("id"), "served_model": body.get("model"), "usage": body.get("usage"), "attempts": attempts}
    raise AssertionError("unreachable")


def request_intent(
    *, api_key: str, model: str, question: str, completion: str,
    timeout_seconds: float, max_attempts: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    payload = build_payload(model, question, completion)
    return request_json_object(
        api_key=api_key, model=model, payload=payload,
        parse=lambda content: parse_intent_judgment(content, completion),
        timeout_seconds=timeout_seconds, max_attempts=max_attempts,
    )


def deterministic_answer(row: dict[str, Any]) -> str | None:
    value = row.get("parsed_answer")
    if normalize_short_answer(str(value or "")) in {"yes", "no"}:
        return normalize_short_answer(str(value))
    completion = str(row.get("completion") or "")
    score = pathvqa_score(completion, "yes")
    value = normalize_short_answer(score["contract_aligned_answer"])
    return value if value in {"yes", "no"} else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-paid-requests", required=True, type=int)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=60)
    args = parser.parse_args()
    if args.max_paid_requests <= 0 or args.max_attempts <= 0:
        raise ValueError("request and attempt caps must be positive")
    api_key = os.getenv("AIGCBEST_API_KEY")
    if not api_key:
        raise RuntimeError("AIGCBEST_API_KEY is not set")
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines()]
    existing = {}
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            existing[row["cache_key"]] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures_path = args.output.with_name(args.output.stem + "_failures.jsonl")
    physical_requests = 0
    with args.output.open("a", encoding="utf-8") as output:
        for row in rows:
            if deterministic_answer(row) is not None:
                continue
            question = str(row["question"])
            completion = str(row["completion"])
            key = cache_key(question, completion, args.model)
            if key in existing:
                continue
            remaining = args.max_paid_requests - physical_requests
            if remaining <= 0:
                break
            try:
                judgment, metadata = request_intent(
                    api_key=api_key,
                    model=args.model,
                    question=question,
                    completion=completion,
                    timeout_seconds=args.timeout_seconds,
                    max_attempts=min(args.max_attempts, remaining),
                )
            except JudgeRequestFailure as exc:
                physical_requests += len(exc.attempts)
                with failures_path.open("a", encoding="utf-8") as failures:
                    failures.write(json.dumps({
                        "index": row.get("index"), "cache_key": key,
                        "error": str(exc), "terminal": exc.terminal,
                        "attempts": exc.attempts,
                    }, ensure_ascii=False) + "\n")
                    failures.flush()
                    os.fsync(failures.fileno())
                if exc.terminal:
                    raise
                continue
            physical_requests += len(metadata["attempts"])
            result = {
                "schema_version": 1,
                "index": row.get("index"),
                "panel_index": row.get("panel_index"),
                "source_index": row.get("source_index"),
                "source_record_sha256": row.get("source_record_sha256"),
                "completion_sha256": hashlib.sha256(completion.encode()).hexdigest(),
                "cache_key": key,
                "requested_model": args.model,
                "judgment": judgment,
                "metadata": metadata,
                "reference_answer_sent": False,
                "image_sent": False,
                "model_identity_sent": False,
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            existing[key] = result
    print(json.dumps({
        "status": "completed",
        "physical_requests_this_run": physical_requests,
        "cached_or_completed_rows": len(existing),
    }))


if __name__ == "__main__":
    main()
