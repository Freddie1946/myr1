#!/usr/bin/env python3
"""Verify OmniMedVQA corrective-v3 adapter behavior without gating on accuracy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(
    metrics_path: Path,
    predictions_path: Path,
    expected_data_sha256: str,
    expected_model_config_sha256: str,
) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = {
        "status": "completed",
        "task": "omnimedvqa",
        "split_role": "adapter_smoke",
        "count": 16,
        "generation_contract": "omnimed_corrective_v3_192",
        "max_new_tokens": 192,
        "data_sha256": expected_data_sha256,
        "model_config_sha256": expected_model_config_sha256,
        "predictions_sha256": sha256_file(predictions_path),
        "primary_metric": "strict_final_option_accuracy",
    }
    mismatches = {
        key: {"expected": value, "actual": metrics.get(key)}
        for key, value in expected.items()
        if metrics.get(key) != value
    }
    if mismatches:
        raise ValueError(f"corrective smoke contract mismatch: {mismatches}")
    if len(rows) != 16 or [row.get("index") for row in rows] != list(range(16)):
        raise ValueError("corrective smoke predictions are not 16 contiguous rows")
    nonempty = sum(bool(str(row.get("completion") or "").strip()) for row in rows)
    available = sum(bool(row.get("strict_final_answer_available")) for row in rows)
    caps = sum(bool(row.get("reached_generation_cap")) for row in rows)
    failures = {}
    if nonempty / 16 < 0.95:
        failures["nonempty_rate"] = nonempty / 16
    if available / 16 < 0.75:
        failures["strict_final_answer_coverage"] = available / 16
    if caps / 16 > 0.25:
        failures["generation_cap_hit_rate"] = caps / 16
    if failures:
        raise ValueError(f"corrective smoke behavior failed: {failures}")
    return {
        "schema_version": 1,
        "status": "passed",
        "formal_result": False,
        "count": 16,
        "nonempty_rate": nonempty / 16,
        "strict_final_answer_coverage": available / 16,
        "generation_cap_hit_rate": caps / 16,
        "accuracy_used_as_gate": False,
        "threshold_or_rule_change_permitted_after_smoke": False,
        "metrics_sha256": sha256_file(metrics_path),
        "predictions_sha256": expected["predictions_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-model-config-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(
        args.metrics,
        args.predictions,
        args.expected_data_sha256,
        args.expected_model_config_sha256,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
