#!/usr/bin/env python3
"""Run one non-training GPT-4o Stage3 judge smoke through AIGCBest.

The script deliberately makes at most one paid chat-completions request per
invocation.  It has no retry path and refuses to overwrite an existing record.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage3_openrouter_judge import (
    SCHEMA_VERSION,
    SYSTEM_PROMPT,
    response_schema,
    score_events,
    validate_events,
)


BASE_URL = "https://api2.aigcbest.top"
MODEL = "gpt-4o-2024-08-06"
MODEL_RATIO = 1.25
COMPLETION_RATIO = 4
BASE_INPUT_USD_PER_MILLION = 2.0
MAX_TOKENS = 320


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def get_json(url: str, *, key: str | None = None) -> dict[str, Any]:
    headers = {"User-Agent": "PathVLM-R1 GPT-4o Stage3 smoke"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"non-object response from {url}")
    return value


def usage_snapshot(key: str) -> dict[str, Any]:
    try:
        data = get_json(BASE_URL + "/api/usage/token", key=key).get("data")
        if not isinstance(data, dict):
            raise ValueError("usage endpoint lacks object data")
        return {
            "status": "available",
            **{
                name: data.get(name)
                for name in ("total_granted", "total_used", "total_available", "unlimited_quota")
            },
        }
    except Exception as exc:  # supplementary telemetry must not trigger a paid retry
        return {"status": "unavailable", "error_type": type(exc).__name__, "error": str(exc)}


def parse_case(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {"smoke_kind", "image", "problem", "solution", "completion"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"case keys must be exactly {sorted(expected)}")
    if value["smoke_kind"] not in {"synthetic", "pathmmu_validation"}:
        raise ValueError("smoke_kind must be synthetic or pathmmu_validation")
    if any(not isinstance(value[name], str) or not value[name].strip() for name in expected):
        raise ValueError("all case fields must be non-empty strings")
    return value


def make_payload(
    case: dict[str, str],
    image_data_url: str,
    *,
    model: str = MODEL,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    text = (
        "QUESTION:\n"
        + case["problem"]
        + "\n\nREFERENCE ANSWER (context only; do not infer process quality from answer match):\n"
        + case["solution"]
        + "\n\nCANDIDATE COMPLETION TO AUDIT:\n"
        + case["completion"]
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": SCHEMA_VERSION,
                "strict": True,
                "schema": response_schema(),
            },
        },
        "temperature": 0,
        "seed": 42,
        "stream": False,
        "max_tokens": MAX_TOKENS,
    }
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    return payload


def parse_success(
    body: dict[str, Any], *, model: str = MODEL
) -> tuple[dict[str, Any], dict[str, Any]]:
    if body.get("model") != model:
        raise ValueError(f"served model mismatch: {body.get('model')!r}")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("response must contain exactly one choice")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ValueError(f"non-terminal finish_reason: {choice.get('finish_reason')!r}")
    content = choice.get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("response content is not a string")
    events = validate_events(json.loads(content))
    return events, score_events(events)


def verify_catalog_and_pricing(
    key: str,
    *,
    model: str = MODEL,
    model_ratio: float = MODEL_RATIO,
    completion_ratio: float = COMPLETION_RATIO,
) -> dict[str, Any]:
    catalog = get_json(BASE_URL + "/v1/models", key=key).get("data", [])
    matches = [row for row in catalog if isinstance(row, dict) and row.get("id") == model]
    if len(matches) != 1:
        raise RuntimeError(f"expected one authenticated catalog row for {model}, got {len(matches)}")
    pricing = get_json(BASE_URL + "/api/pricing").get("data", [])
    price_rows = [row for row in pricing if isinstance(row, dict) and row.get("model_name") == model]
    if len(price_rows) != 1:
        raise RuntimeError(f"expected one public pricing row for {model}, got {len(price_rows)}")
    price_row = price_rows[0]
    if (
        price_row.get("model_ratio") != model_ratio
        or price_row.get("completion_ratio") != completion_ratio
    ):
        raise RuntimeError(f"public {model} price ratio changed: {price_row}")
    return price_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--expected-model-ratio", type=float)
    parser.add_argument("--expected-completion-ratio", type=float)
    parser.add_argument(
        "--reasoning-effort", choices=("minimal", "low", "medium", "high")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to repeat existing paid smoke: {args.output}")
    key = os.getenv("AIGCBEST_API_KEY")
    if not key:
        raise RuntimeError("AIGCBEST_API_KEY is not set")
    if not args.model.strip():
        raise ValueError("model must be non-empty")
    custom_ratio_count = sum(
        value is not None
        for value in (args.expected_model_ratio, args.expected_completion_ratio)
    )
    if custom_ratio_count == 1:
        raise ValueError("both expected pricing ratios must be supplied together")
    if args.model != MODEL and custom_ratio_count != 2:
        raise ValueError("non-default model requires both expected pricing ratios")
    model_ratio = (
        MODEL_RATIO if args.expected_model_ratio is None else args.expected_model_ratio
    )
    completion_ratio = (
        COMPLETION_RATIO
        if args.expected_completion_ratio is None
        else args.expected_completion_ratio
    )
    if model_ratio <= 0 or completion_ratio <= 0:
        raise ValueError("pricing ratios must be positive")
    input_usd_per_million = BASE_INPUT_USD_PER_MILLION * model_ratio
    output_usd_per_million = input_usd_per_million * completion_ratio
    case = parse_case(args.case_json)
    image_path = Path(case["image"]).resolve()
    image_bytes = image_path.read_bytes()
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    if not mime.startswith("image/"):
        raise ValueError(f"unsupported image MIME: {mime}")
    image_data_url = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    price_row = verify_catalog_and_pricing(
        key,
        model=args.model,
        model_ratio=model_ratio,
        completion_ratio=completion_ratio,
    )
    before = usage_snapshot(key)
    payload = make_payload(
        case,
        image_data_url,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
    )
    request = urllib.request.Request(
        BASE_URL + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    started = time.monotonic()
    base_record: dict[str, Any] = {
        "schema_version": 1,
        "recorded_at": now_iso(),
        "formal_result": False,
        "training_call": False,
        "provider": "aigcbest",
        "base_url": BASE_URL,
        "requested_model": args.model,
        "smoke_kind": case["smoke_kind"],
        "case_json": str(args.case_json.resolve()),
        "case_sha256": hashlib.sha256(args.case_json.read_bytes()).hexdigest(),
        "image_path": str(image_path),
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "image_bytes": len(image_bytes),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "response_schema": SCHEMA_VERSION,
        "temperature": 0,
        "seed": 42,
        "max_tokens": MAX_TOKENS,
        "reasoning_effort": args.reasoning_effort,
        "public_pricing_snapshot": price_row,
        "input_usd_per_million_tokens": input_usd_per_million,
        "output_usd_per_million_tokens": output_usd_per_million,
        "token_usage_before": before,
        "paid_request_limit_for_invocation": 1,
        "retry_count": 0,
    }
    response_observation: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            status = int(response.status)
            raw_body = response.read().decode("utf-8")
            body = json.loads(raw_body)
            safe_headers = {
                name.lower(): value
                for name, value in response.headers.items()
                if name.lower() in {"x-request-id", "x-oneapi-request-id", "content-type"}
            }
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        cost = (
            prompt_tokens * input_usd_per_million
            + completion_tokens * output_usd_per_million
        ) / 1_000_000
        response_observation = {
            "http_status": status,
            "safe_response_headers": safe_headers,
            "response": body,
            "served_model": body.get("model"),
            "response_id": body.get("id"),
            "usage": usage,
            "estimated_request_cost_usd": cost,
        }
        events, scores = parse_success(body, model=args.model)
        record = {
            **base_record,
            "status": "passed",
            "latency_seconds": time.monotonic() - started,
            **response_observation,
            "events": events,
            "scores_at_penalty_0_4": scores,
            "token_usage_after": usage_snapshot(key),
        }
        atomic_json(args.output, record)
        print(json.dumps({
            "status": "passed",
            "smoke_kind": case["smoke_kind"],
            "served_model": body.get("model"),
            "usage": usage,
            "estimated_request_cost_usd": cost,
            "process_score_at_0_4": scores["process"],
            "output": str(args.output.resolve()),
        }, ensure_ascii=False, sort_keys=True))
    except Exception as exc:
        record = {
            **base_record,
            "status": "failed",
            "latency_seconds": time.monotonic() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
            **response_observation,
            "token_usage_after": usage_snapshot(key),
        }
        if isinstance(exc, urllib.error.HTTPError):
            record["http_status"] = exc.code
            record["error_response_body"] = exc.read().decode("utf-8", errors="replace")[:20000]
        atomic_json(args.output, record)
        raise


if __name__ == "__main__":
    main()
