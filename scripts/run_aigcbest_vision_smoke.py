#!/usr/bin/env python3
"""One auditable AIGCBest vision smoke with before/after usage and price estimate."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_URL = "https://api2.aigcbest.top"
MODEL = "qwen-vl-plus"
INPUT_USD_PER_MILLION = 0.8
OUTPUT_USD_PER_MILLION = 2.0


def get_json(url: str, *, key: str | None = None) -> dict[str, Any]:
    headers = {"User-Agent": "PathVLM-R1 AIGCBest smoke"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"non-object response from {url}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--question", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to repeat existing paid smoke: {args.output}")
    key = os.getenv("AIGCBEST_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("AIGCBEST_API_KEY is not set")
    image = args.image.resolve()
    raw = image.read_bytes()
    mime = mimetypes.guess_type(image.name)[0] or "image/jpeg"
    if not mime.startswith("image/"):
        raise ValueError(f"unsupported image MIME: {mime}")
    data_url = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"

    catalog = get_json(BASE_URL + "/v1/models", key=key).get("data", [])
    if not any(isinstance(row, dict) and row.get("id") == MODEL for row in catalog):
        raise RuntimeError(f"smoke model is absent from authenticated catalog: {MODEL}")
    pricing = get_json(BASE_URL + "/api/pricing")
    price_rows = [row for row in pricing.get("data", []) if row.get("model_name") == MODEL]
    if len(price_rows) != 1:
        raise RuntimeError(f"expected one public pricing row for {MODEL}, got {len(price_rows)}")
    price_row = price_rows[0]
    if price_row.get("model_ratio") != 0.4 or price_row.get("completion_ratio") != 2.5:
        raise RuntimeError(f"public {MODEL} price ratio changed: {price_row}")

    before = get_json(BASE_URL + "/api/usage/token", key=key)["data"]
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.question},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 32,
        "stream": False,
    }
    request = urllib.request.Request(
        BASE_URL + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(),
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=120) as response:
        status = int(response.status)
        body = json.loads(response.read().decode("utf-8"))
        safe_headers = {
            name.lower(): value
            for name, value in response.headers.items()
            if name.lower() in {"x-request-id", "x-oneapi-request-id", "content-type"}
        }
    latency = time.monotonic() - started
    after = get_json(BASE_URL + "/api/usage/token", key=key)["data"]
    choices = body.get("choices")
    if status != 200 or not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("AIGCBest smoke did not return one successful choice")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("AIGCBest smoke returned empty content")
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))
    estimated_cost = (
        prompt_tokens * INPUT_USD_PER_MILLION
        + completion_tokens * OUTPUT_USD_PER_MILLION
    ) / 1_000_000
    record = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "formal_result": False,
        "provider": "aigcbest",
        "base_url": BASE_URL,
        "requested_model": MODEL,
        "served_model": body.get("model"),
        "response_id": body.get("id"),
        "http_status": status,
        "safe_response_headers": safe_headers,
        "latency_seconds": latency,
        "image_path": str(image),
        "image_sha256": hashlib.sha256(raw).hexdigest(),
        "image_bytes": len(raw),
        "question": args.question,
        "max_tokens": 32,
        "response_text": content,
        "finish_reason": choices[0].get("finish_reason"),
        "usage": usage,
        "public_pricing_snapshot": price_row,
        "assumed_group": "default",
        "input_usd_per_million_tokens": INPUT_USD_PER_MILLION,
        "output_usd_per_million_tokens": OUTPUT_USD_PER_MILLION,
        "estimated_request_cost_usd": estimated_cost,
        "token_usage_before": {
            key: before.get(key)
            for key in ("total_granted", "total_used", "total_available", "unlimited_quota")
        },
        "token_usage_after": {
            key: after.get(key)
            for key in ("total_granted", "total_used", "total_available", "unlimited_quota")
        },
    }
    atomic_json(args.output, record)
    print(json.dumps({
        "status": "passed",
        "model": body.get("model"),
        "response_id": body.get("id"),
        "usage": usage,
        "estimated_request_cost_usd": estimated_cost,
        "latency_seconds": latency,
        "output": str(args.output.resolve()),
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
