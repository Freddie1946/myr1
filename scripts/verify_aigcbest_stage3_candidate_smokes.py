#!/usr/bin/env python3
"""Verify independent AIGCBest Stage3 candidate smoke records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from stage3_openrouter_judge import SCHEMA_VERSION, validate_events


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(args.input) != args.expected_count or args.expected_count < 2:
        raise ValueError("smoke count does not match the frozen expected count")
    response_ids: set[str] = set()
    rows = []
    for path in args.input:
        value = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "status": "passed",
            "formal_result": False,
            "training_call": False,
            "provider": "aigcbest",
            "requested_model": args.model,
            "served_model": args.model,
            "response_schema": SCHEMA_VERSION,
            "retry_count": 0,
            "http_status": 200,
        }
        bad = {
            key: {"expected": wanted, "actual": value.get(key)}
            for key, wanted in expected.items()
            if value.get(key) != wanted
        }
        if bad:
            raise ValueError(f"candidate smoke contract mismatch in {path}: {bad}")
        response_id = value.get("response_id")
        if not isinstance(response_id, str) or not response_id or response_id in response_ids:
            raise ValueError(f"missing or duplicate response id in {path}")
        response_ids.add(response_id)
        validate_events(value.get("events"))
        latency = float(value["latency_seconds"])
        cost = float(value["estimated_request_cost_usd"])
        if latency <= 0 or cost < 0:
            raise ValueError(f"invalid latency or cost in {path}")
        rows.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "response_id": response_id,
                "latency_seconds": latency,
                "estimated_request_cost_usd": cost,
            }
        )
    output = {
        "schema_version": 1,
        "status": "passed",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "provider": "aigcbest",
        "model": args.model,
        "request_count": len(rows),
        "success_count": len(rows),
        "retry_count": 0,
        "minimum_latency_seconds": min(row["latency_seconds"] for row in rows),
        "maximum_latency_seconds": max(row["latency_seconds"] for row in rows),
        "total_estimated_cost_usd": sum(
            row["estimated_request_cost_usd"] for row in rows
        ),
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, args.output)
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
