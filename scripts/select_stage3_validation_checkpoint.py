#!/usr/bin/env python3
"""Validate three Stage3 validation385 runs and apply the frozen tie-break."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_DATA_SHA256 = "f52c9a412da579db8fb8bf52f6fa320e7c4ae9cb42b6cf1bdf62f77ab158dcf0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_candidate(
    metrics_path: Path, *, expected_step: int, expected_model: Path,
    expected_split_role: str = "validation_0385",
) -> dict[str, Any]:
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse {metrics_path}: {exc}") from exc
    expected = {
        "status": "completed",
        "split_role": expected_split_role,
        "test_accessed": False,
        "count": 385,
        "data_sha256": EXPECTED_DATA_SHA256,
    }
    mismatch = {
        key: {"expected": value, "actual": metrics.get(key)}
        for key, value in expected.items()
        if metrics.get(key) != value
    }
    if mismatch:
        raise ValueError(f"validation contract mismatch in {metrics_path}: {mismatch}")
    if Path(metrics.get("model_path", "")).resolve() != expected_model.resolve():
        raise ValueError(f"model_path mismatch in {metrics_path}")
    predictions = Path(metrics.get("predictions_file") or metrics_path.with_name("predictions.jsonl"))
    if not predictions.is_file():
        raise ValueError(f"predictions file is missing for {metrics_path}: {predictions}")
    if sha256_file(predictions) != metrics.get("predictions_sha256"):
        raise ValueError(f"predictions SHA-256 mismatch for {metrics_path}")
    correct = metrics.get("correct")
    format_correct = metrics.get("format_correct")
    if not isinstance(correct, int) or not 0 <= correct <= 385:
        raise ValueError(f"invalid correct count in {metrics_path}: {correct!r}")
    if not isinstance(format_correct, int) or not 0 <= format_correct <= 385:
        raise ValueError(f"invalid format count in {metrics_path}: {format_correct!r}")
    return {
        "step": expected_step,
        "epoch": expected_step // 500,
        "model_path": str(expected_model.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "metrics_sha256": sha256_file(metrics_path),
        "predictions_path": str(predictions.resolve()),
        "predictions_sha256": metrics["predictions_sha256"],
        "correct": correct,
        "accuracy": metrics["accuracy"],
        "format_correct": format_correct,
        "format_accuracy": metrics.get("format_accuracy", format_correct / 385),
        "choice_extracted": metrics.get("choice_extracted"),
        "generation_cap_hit_count": metrics.get("generation_cap_hit_count"),
    }


def select(
    run_dir: Path, validation_root: Path,
    expected_split_role: str = "validation_0385",
) -> dict[str, Any]:
    candidates = []
    for step in (500, 1000, 1500):
        model = run_dir / "epoch_model_snapshots" / f"checkpoint-{step}"
        metrics = validation_root / f"checkpoint-{step}" / "metrics.json"
        candidates.append(validate_candidate(
            metrics, expected_step=step, expected_model=model,
            expected_split_role=expected_split_role,
        ))
    selected = max(candidates, key=lambda row: (row["correct"], row["format_correct"], -row["step"]))
    return {
        "schema_version": 1,
        "status": "completed",
        "formal_result": False,
        "selection_split": "pathmmu_validation_0385",
        "inference_split_role": expected_split_role,
        "test_accessed": False,
        "tie_break": ["higher_correct", "higher_format_correct", "earlier_epoch"],
        "run_dir": str(run_dir.resolve()),
        "validation_root": str(validation_root.resolve()),
        "candidates": candidates,
        "selected_step": selected["step"],
        "selected_epoch": selected["epoch"],
        "selected_model_path": selected["model_path"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-split-role", choices=("validation_0385", "validation_smoke"),
        default="validation_0385",
    )
    args = parser.parse_args()
    try:
        result = select(
            args.run_dir, args.validation_root,
            expected_split_role=args.expected_split_role,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Stage3 validation selection failed: {exc}") from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
