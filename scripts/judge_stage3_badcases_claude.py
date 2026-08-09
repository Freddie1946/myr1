#!/usr/bin/env python3
"""Independent visual audit of paired Stage2/Stage3 PathMMU flips."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


API_URL = "https://api2.aigcbest.top/v1/chat/completions"
MODEL = "claude-sonnet-4-6"
INPUT_USD_PER_MILLION = 3.0
OUTPUT_USD_PER_MILLION = 15.0
CATEGORIES = {
    "none", "visual_feature_misread", "pathology_knowledge_error", "option_mapping_error",
    "unsupported_hallucination", "insufficient_reasoning", "reference_or_image_ambiguity", "other",
}
SYSTEM_PROMPT = """You are an independent pathology VQA auditor. Compare an anonymous BEFORE
response and AFTER response for the same image and multiple-choice question. The reference answer
is provided but may itself be imperfect; inspect the image and explicitly flag ambiguity. Identify
why the answer changed. Do not judge by wording similarity alone.

Return exactly one JSON object:
{"effect":"improvement|regression|ambiguous", "before_error":"category",
 "after_error":"category", "reference_ambiguous":boolean, "image_evidence":"string",
 "explanation":"string"}

Each error category must be one of: none, visual_feature_misread, pathology_knowledge_error,
option_mapping_error, unsupported_hallucination, insufficient_reasoning,
reference_or_image_ambiguity, other. image_evidence and explanation must each be at most 300
characters. Use none only when that response is adequately supported."""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def image_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def parse(content: str) -> dict[str, Any]:
    text = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    value = json.loads(text)
    expected = {"effect", "before_error", "after_error", "reference_ambiguous", "image_evidence", "explanation"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("invalid field set")
    if value["effect"] not in {"improvement", "regression", "ambiguous"}:
        raise ValueError("invalid effect")
    if value["before_error"] not in CATEGORIES or value["after_error"] not in CATEGORIES:
        raise ValueError("invalid error category")
    if type(value["reference_ambiguous"]) is not bool:
        raise ValueError("reference_ambiguous is not boolean")
    for key in ("image_evidence", "explanation"):
        if not isinstance(value[key], str) or not value[key].strip() or len(value[key]) > 600:
            raise ValueError(f"invalid {key}")
    return value


def request(api_key: str, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    text = (
        f"QUESTION:\n{row['problem']}\n\nREFERENCE:\n{row['reference']}\n\n"
        f"BEFORE RESPONSE (Stage2):\n{row['stage2_completion']}\n\n"
        f"AFTER RESPONSE (Stage3):\n{row['stage3_completion']}"
    )
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_url(Path(row["image"]))}},
            ]},
        ],
        "temperature": 0,
        "max_tokens": 480,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    req = urllib.request.Request(
        API_URL, data=canonical(payload).encode(), method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=180) as response:
        body = json.loads(response.read().decode("utf-8", errors="replace"))
    if body.get("model") != MODEL:
        raise ValueError(f"served-model mismatch: {body.get('model')!r}")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("response lacks exactly one choice")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("response content missing")
    verdict = parse(content)
    usage = body.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("usage missing")
    cost = (
        int(usage["prompt_tokens"]) * INPUT_USD_PER_MILLION
        + int(usage["completion_tokens"]) * OUTPUT_USD_PER_MILLION
    ) / 1_000_000
    return verdict, {
        "served_model": body.get("model"), "response_id": body.get("id"),
        "usage": usage, "cost_usd": cost, "latency_seconds": time.monotonic() - started,
    }


def atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-direction", type=int, default=4)
    parser.add_argument("--budget-limit-usd", type=float, default=0.15)
    args = parser.parse_args()
    api_key = os.getenv("AIGCBEST_API_KEY", "")
    if not api_key:
        raise RuntimeError("AIGCBEST_API_KEY is not set")
    rows = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    selected = []
    for direction in ("regressions", "improvements"):
        selected.extend([row for row in rows if row["direction"] == direction][:args.per_direction])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "visual_badcase_judgments.jsonl"
    existing = {}
    if results_path.exists():
        existing = {int(row["index"]): row for row in map(json.loads, results_path.open())}
    ledger_path = output / "budget_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        "limit_usd": args.budget_limit_usd, "spent_usd": 0.0, "attempts": []
    }
    if float(ledger["limit_usd"]) != args.budget_limit_usd:
        raise ValueError("budget limit differs from existing ledger")
    for row in selected:
        if int(row["index"]) in existing:
            continue
        last_error = None
        for attempt in range(1, 4):
            if args.budget_limit_usd - float(ledger["spent_usd"]) < 0.025:
                raise RuntimeError("badcase judge budget reserve reached")
            if attempt > 1:
                time.sleep((2, 8)[attempt - 2])
            try:
                verdict, metadata = request(api_key, row)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                accounted = 0.01
                ledger["attempts"].append({"index": row["index"], "attempt": attempt, "status": "failed", "accounted_cost_usd": accounted, "error": f"{type(exc).__name__}: {exc}"})
                ledger["spent_usd"] = sum(x["accounted_cost_usd"] for x in ledger["attempts"])
                atomic(ledger_path, ledger)
                continue
            result = {**row, "judge": MODEL, "verdict": verdict, **metadata}
            with results_path.open("a", encoding="utf-8") as handle:
                handle.write(canonical(result) + "\n")
                handle.flush(); os.fsync(handle.fileno())
            ledger["attempts"].append({"index": row["index"], "attempt": attempt, "status": "passed", "accounted_cost_usd": metadata["cost_usd"]})
            ledger["spent_usd"] = sum(x["accounted_cost_usd"] for x in ledger["attempts"])
            atomic(ledger_path, ledger)
            print(row["index"], verdict["effect"], verdict["after_error"], f"${metadata['cost_usd']:.5f}", flush=True)
            break
        else:
            raise RuntimeError(f"case {row['index']} failed after retries: {last_error}")


if __name__ == "__main__":
    main()
