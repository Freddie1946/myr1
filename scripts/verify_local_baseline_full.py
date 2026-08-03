#!/usr/bin/env python3
"""Fail-closed integrity verification for completed local-baseline full runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object row at {path}:{line_number}")
            rows.append(row)
    return rows


def verify(
    *,
    task: str,
    metrics_path: Path,
    predictions_path: Path,
    run_config_path: Path,
    expected_count: int,
    expected_split_role: str,
    expected_data_sha256: str,
    expected_model_config_sha256: str,
    expected_pathvqa_answer_scope: str | None = None,
) -> dict[str, Any]:
    if task not in {"pathmmu", "pathvqa", "omnimedvqa"}:
        raise ValueError(f"unsupported task: {task}")
    allowed_split = {
        "pathmmu": "test999_development",
        "pathvqa": "external_test",
        "omnimedvqa": "external_test",
    }[task]
    if expected_split_role != allowed_split:
        raise ValueError(
            f"invalid full-run split role for {task}: {expected_split_role}"
        )
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = json.loads(run_config_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict) or not isinstance(config, dict):
        raise ValueError("metrics and run config must be JSON objects")
    rows = load_jsonl(predictions_path)
    predictions_sha256 = sha256_file(predictions_path)

    common = {
        "status": "completed",
        "split_role": expected_split_role,
        "data_sha256": expected_data_sha256,
        "model_config_sha256": expected_model_config_sha256,
    }
    expected_metrics: dict[str, Any] = {
        **common,
        "count": expected_count,
        "predictions_sha256": predictions_sha256,
    }
    expected_config: dict[str, Any] = {
        **common,
        "selected_count": expected_count,
    }
    if task == "pathmmu":
        expected_metrics.update({"diagnostic_only": True, "test_accessed": True})
        expected_config = {
            **common,
            "source_count": expected_count,
            "selected_count": expected_count,
            "diagnostic_only": True,
            "test_accessed": True,
        }
    else:
        expected_metrics["task"] = task
        expected_config["task"] = task
    if expected_pathvqa_answer_scope is not None:
        if task != "pathvqa" or expected_pathvqa_answer_scope not in {"all", "yes_no_only"}:
            raise ValueError("PathVQA answer scope is invalid for this full run")
        expected_metrics["answer_scope"] = expected_pathvqa_answer_scope
        expected_config["answer_scope"] = expected_pathvqa_answer_scope

    mismatches: dict[str, Any] = {}
    for artifact_name, artifact, expected in (
        ("metrics", metrics, expected_metrics),
        ("run_config", config, expected_config),
    ):
        artifact_mismatches = {
            key: {"expected": value, "actual": artifact.get(key)}
            for key, value in expected.items()
            if artifact.get(key) != value
        }
        if artifact_mismatches:
            mismatches[artifact_name] = artifact_mismatches
    if mismatches:
        raise ValueError(f"full-run artifact contract mismatch: {mismatches}")
    if len(rows) != expected_count:
        raise ValueError(
            f"prediction count mismatch: expected {expected_count}, got {len(rows)}"
        )
    indices = [int(row.get("index", -1)) for row in rows]
    if indices != list(range(expected_count)):
        raise ValueError("prediction indices are not contiguous and ordered")
    for row in rows:
        source_sha = str(row.get("source_record_sha256") or "")
        if len(source_sha) != 64:
            raise ValueError(f"invalid source hash at index {row.get('index')}")

    return {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "formal_result": True,
        "task": task,
        "split_role": expected_split_role,
        "count": expected_count,
        "metrics": str(metrics_path.resolve()),
        "metrics_sha256": sha256_file(metrics_path),
        "run_config": str(run_config_path.resolve()),
        "run_config_sha256": sha256_file(run_config_path),
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": predictions_sha256,
        "data_sha256": expected_data_sha256,
        "model_config_sha256": expected_model_config_sha256,
        "individual_answers_modified_or_rejudged": False,
        "pathvqa_answer_scope": expected_pathvqa_answer_scope,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", choices=("pathmmu", "pathvqa", "omnimedvqa"), required=True
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-split-role", required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-model-config-sha256", required=True)
    parser.add_argument(
        "--expected-pathvqa-answer-scope", choices=("all", "yes_no_only")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = verify(
            task=args.task,
            metrics_path=args.metrics,
            predictions_path=args.predictions,
            run_config_path=args.run_config,
            expected_count=args.expected_count,
            expected_split_role=args.expected_split_role,
            expected_data_sha256=args.expected_data_sha256,
            expected_model_config_sha256=args.expected_model_config_sha256,
            expected_pathvqa_answer_scope=args.expected_pathvqa_answer_scope,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Local baseline full verification failed: {exc}") from exc
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, args.output)
    print(text, end="")


if __name__ == "__main__":
    main()
