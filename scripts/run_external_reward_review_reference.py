#!/usr/bin/env python3
"""Create an independent, resumable external reference for reward-review cases.

The output is intentionally separate from the blinded expert packet.  It may be
shown only after an expert has saved their independent first-pass ratings.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_URL = "https://api2.aigcbest.top/v1/chat/completions"
PROMPT_VERSION = "independent_reward_review_reference_v1"
MODEL = "claude-sonnet-4-6"
TRANSIENT = {408, 429, 500, 502, 503, 504, 520, 529}
BOOLEAN_FIELDS = (
    "image_feature_analysis_present",
    "option_elimination_present",
    "medical_knowledge_support_present",
    "histological_definition_error",
    "logical_contradiction",
    "outdated_or_incorrect_pathology_criterion",
    "hallucination_present",
    "reference_answer_ambiguous_or_noisy",
)
SCORE_FIELDS = (
    "pathology_reasoning_correctness_1_to_5",
    "image_grounding_1_to_5",
    "logic_consistency_1_to_5",
    "reviewer_confidence_1_to_5",
)
SYSTEM_PROMPT = """You are an independent pathology review assistant. Evaluate one candidate
answer using the supplied pathology image, question/options, and dataset reference answer. The
reference may itself be noisy; do not copy or trust it blindly. Return only one JSON object.

Required fields:
image_feature_analysis_present, option_elimination_present,
medical_knowledge_support_present, histological_definition_error,
logical_contradiction, outdated_or_incorrect_pathology_criterion,
hallucination_present, reference_answer_ambiguous_or_noisy,
pathology_reasoning_correctness_1_to_5, image_grounding_1_to_5,
logic_consistency_1_to_5, reviewer_confidence_1_to_5, overall_reason.

The eight event fields must be exactly true, false, or "uncertain". The four scores are integers
1 through 5. overall_reason must be concise, identify concrete evidence/errors, and be under 800
characters. This is advisory only; do not claim to be a human pathology expert."""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def parse_reference(text: str) -> dict[str, Any]:
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    value = json.loads(cleaned)
    expected = {*BOOLEAN_FIELDS, *SCORE_FIELDS, "overall_reason"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"invalid field set: {set(value) if isinstance(value, dict) else type(value)}")
    for field in BOOLEAN_FIELDS:
        if value[field] not in (True, False, "uncertain"):
            raise ValueError(f"invalid {field}: {value[field]!r}")
    for field in SCORE_FIELDS:
        if isinstance(value[field], bool) or not isinstance(value[field], int) or not 1 <= value[field] <= 5:
            raise ValueError(f"invalid {field}: {value[field]!r}")
    if not isinstance(value["overall_reason"], str) or not value["overall_reason"].strip():
        raise ValueError("missing overall_reason")
    return value


def request_once(api_key: str, case: dict[str, Any], image: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    user_text = (
        "QUESTION AND OPTIONS:\n" + case["question_and_options"]
        + "\n\nDATASET REFERENCE ANSWER (may be noisy):\n" + case["dataset_reference_answer"]
        + "\n\nCANDIDATE RESPONSE TO REVIEW:\n" + case["candidate_completion"]
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_url(image)}},
            ]},
        ],
        "temperature": 0,
        "max_tokens": 700,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    request = urllib.request.Request(
        API_URL,
        data=canonical(payload).encode(),
        method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=240) as response:
        body = json.loads(response.read().decode("utf-8", errors="replace"))
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("response lacks exactly one choice")
    served = body.get("model")
    if served != MODEL:
        raise ValueError(f"served model mismatch: {served!r}")
    choice = choices[0]
    result = parse_reference(choice.get("message", {}).get("content", ""))
    usage = body.get("usage") or {}
    prompt = int(usage.get("prompt_tokens", 0))
    completion = int(usage.get("completion_tokens", 0))
    metadata = {
        "response_id": body.get("id"),
        "served_model": served,
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
        "estimated_cost_usd": (prompt * 3.0 + completion * 15.0) / 1_000_000,
        "latency_seconds": time.monotonic() - started,
    }
    return result, metadata


def call_with_retry(api_key: str, case: dict[str, Any], image: Path, retries: int) -> dict[str, Any]:
    errors: list[str] = []
    for attempt in range(1, retries + 1):
        try:
            reference, metadata = request_once(api_key, case, image)
            return {
                "schema_version": 1,
                "case_id": case["case_id"],
                "prompt_version": PROMPT_VERSION,
                "requested_model": MODEL,
                "reference": reference,
                "metadata": metadata,
                "attempt": attempt,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
            if attempt == retries:
                break
            time.sleep(min(30, 2 ** (attempt - 1)) + random.random())
    raise RuntimeError(f"{case['case_id']} failed: {' | '.join(errors)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()
    api_key = os.environ.get("AIGCBEST_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("AIGCBEST_API_KEY is not set")
    cases_path = args.input_dir / "cases.jsonl"
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line]
    if len(cases) != 60:
        raise ValueError(f"expected 60 cases, got {len(cases)}")
    if args.limit is not None:
        cases = cases[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "references.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[row["case_id"]] = row
    pending = [case for case in cases if case["case_id"] not in existing]
    completed = dict(existing)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for case in pending:
            image = args.input_dir / case["image_file"]
            futures[pool.submit(call_with_retry, api_key, case, image, args.retries)] = case["case_id"]
        for future in as_completed(futures):
            row = future.result()
            completed[row["case_id"]] = row
            ordered = [completed[key] for key in sorted(completed)]
            temp = output_path.with_suffix(".jsonl.tmp")
            temp.write_text("".join(canonical(item) + "\n" for item in ordered), encoding="utf-8")
            os.replace(temp, output_path)
            print(json.dumps({"completed": len(completed), "target": len(cases), "case_id": row["case_id"]}), flush=True)

    selected = [completed[case["case_id"]] for case in cases]
    total_cost = sum(float(row["metadata"]["estimated_cost_usd"]) for row in selected)
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "scientific_boundary": "external model advice for second-pass expert review; not human ground truth",
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "case_count": len(selected),
        "source_cases": str(cases_path.resolve()),
        "source_cases_sha256": sha256_file(cases_path),
        "references_sha256": sha256_file(output_path),
        "estimated_cost_usd": total_cost,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
