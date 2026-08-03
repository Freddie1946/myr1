#!/usr/bin/env python3
"""Verify aggregate adapter behavior without gating on answer accuracy."""

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
    rows = []
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


def predicted_choice(row: dict[str, Any], task: str) -> str | None:
    if task == "pathmmu":
        value = row.get("predicted_choice")
    elif task == "omnimedvqa":
        value = row.get("contract_aligned_predicted_choice")
        if value is None:
            value = row.get("official_predicted_choice")
    else:
        answer_type = str(row.get("answer_type") or "")
        if answer_type == "free_form":
            return "FREE_FORM" if str(row.get("completion") or "").strip() else None
        value = row.get("contract_aligned_answer")
        if value is None:
            value = row.get("normalized_completion")
        value = str(value or "").strip().lower()
        return value if value in {"yes", "no"} else None
    value = str(value or "").strip().upper()
    return value if value in {"A", "B", "C", "D"} else None


def verify(
    *,
    task: str,
    metrics_path: Path,
    predictions_path: Path,
    expected_count: int,
    expected_data_sha256: str,
    expected_model_config_sha256: str,
    minimum_nonempty_rate: float,
    minimum_parseable_rate: float,
    maximum_cap_hit_rate: float,
    expected_pathvqa_answer_scope: str | None = None,
) -> dict[str, Any]:
    if task not in {"pathmmu", "pathvqa", "omnimedvqa"}:
        raise ValueError(f"unsupported task: {task}")
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    for name, value in {
        "minimum_nonempty_rate": minimum_nonempty_rate,
        "minimum_parseable_rate": minimum_parseable_rate,
        "maximum_cap_hit_rate": maximum_cap_hit_rate,
    }.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between zero and one")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a JSON object")
    rows = load_jsonl(predictions_path)
    expected_split_role = "validation_smoke" if task == "pathmmu" else "adapter_smoke"
    expected_metrics = {
        "status": "completed",
        "split_role": expected_split_role,
        "count": expected_count,
        "data_sha256": expected_data_sha256,
        "model_config_sha256": expected_model_config_sha256,
        "predictions_sha256": sha256_file(predictions_path),
    }
    # The PathMMU runner predates the explicit ``task`` metrics field.  Its
    # split role and pinned data hash identify the task without weakening the
    # artifact contract.  External-VQA runners must state their task because
    # they share one adapter and one split-role vocabulary.
    if task != "pathmmu":
        expected_metrics["task"] = task
    if expected_pathvqa_answer_scope is not None:
        if task != "pathvqa" or expected_pathvqa_answer_scope not in {"all", "yes_no_only"}:
            raise ValueError("PathVQA answer scope is invalid for this smoke")
        expected_metrics["answer_scope"] = expected_pathvqa_answer_scope
    mismatches = {
        key: {"expected": value, "actual": metrics.get(key)}
        for key, value in expected_metrics.items()
        if metrics.get(key) != value
    }
    if mismatches:
        raise ValueError(f"smoke metrics contract mismatch: {mismatches}")
    if len(rows) != expected_count:
        raise ValueError(f"prediction count mismatch: expected {expected_count}, got {len(rows)}")
    indices = [int(row.get("index", -1)) for row in rows]
    if indices != list(range(expected_count)):
        raise ValueError("smoke prediction indices are not contiguous and ordered")
    for row in rows:
        source_sha = str(row.get("source_record_sha256") or "")
        if len(source_sha) != 64:
            raise ValueError(f"invalid source hash at index {row.get('index')}")

    nonempty = sum(bool(str(row.get("completion") or "").strip()) for row in rows)
    parseable = sum(predicted_choice(row, task) is not None for row in rows)
    cap_hits = sum(bool(row.get("reached_generation_cap")) for row in rows)
    nonempty_rate = nonempty / expected_count
    parseable_rate = parseable / expected_count
    cap_hit_rate = cap_hits / expected_count
    failures = {}
    if nonempty_rate < minimum_nonempty_rate:
        failures["nonempty_rate"] = {
            "required": minimum_nonempty_rate, "actual": nonempty_rate
        }
    if parseable_rate < minimum_parseable_rate:
        failures["parseable_rate"] = {
            "required": minimum_parseable_rate, "actual": parseable_rate
        }
    if cap_hit_rate > maximum_cap_hit_rate:
        failures["cap_hit_rate"] = {
            "maximum": maximum_cap_hit_rate, "actual": cap_hit_rate
        }
    if task == "pathvqa":
        types = {str(row.get("answer_type") or "") for row in rows}
        required_types = (
            {"yes_no"}
            if expected_pathvqa_answer_scope == "yes_no_only"
            else {"yes_no", "free_form"}
        )
        if types != required_types:
            failures["answer_type_coverage"] = {
                "required": sorted(required_types), "actual": sorted(types)
            }
    if failures:
        raise ValueError(f"aggregate smoke behavior failed: {failures}")

    return {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "formal_result": False,
        "task": task,
        "split_role": expected_split_role,
        "count": expected_count,
        "nonempty_count": nonempty,
        "nonempty_rate": nonempty_rate,
        "parseable_count": parseable,
        "parseable_rate": parseable_rate,
        "generation_cap_hit_count": cap_hits,
        "generation_cap_hit_rate": cap_hit_rate,
        "thresholds": {
            "minimum_nonempty_rate": minimum_nonempty_rate,
            "minimum_parseable_rate": minimum_parseable_rate,
            "maximum_cap_hit_rate": maximum_cap_hit_rate,
        },
        "accuracy_used_as_gate": False,
        "individual_wrong_answers_modified_or_rejudged": False,
        "metrics": str(metrics_path.resolve()),
        "metrics_sha256": sha256_file(metrics_path),
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": expected_metrics["predictions_sha256"],
        "data_sha256": expected_data_sha256,
        "model_config_sha256": expected_model_config_sha256,
        "pathvqa_answer_scope": expected_pathvqa_answer_scope,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("pathmmu", "pathvqa", "omnimedvqa"), required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-model-config-sha256", required=True)
    parser.add_argument("--minimum-nonempty-rate", type=float, default=0.80)
    parser.add_argument("--minimum-parseable-rate", type=float, default=0.80)
    parser.add_argument("--maximum-cap-hit-rate", type=float, default=0.20)
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
            expected_count=args.expected_count,
            expected_data_sha256=args.expected_data_sha256,
            expected_model_config_sha256=args.expected_model_config_sha256,
            minimum_nonempty_rate=args.minimum_nonempty_rate,
            minimum_parseable_rate=args.minimum_parseable_rate,
            maximum_cap_hit_rate=args.maximum_cap_hit_rate,
            expected_pathvqa_answer_scope=args.expected_pathvqa_answer_scope,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Local baseline smoke verification failed: {exc}") from exc
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, args.output)
    print(text, end="")


if __name__ == "__main__":
    main()
