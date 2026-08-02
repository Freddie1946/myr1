#!/usr/bin/env python3
"""Fail-closed finalizer for PathVQA yes/no plus free-form semantic scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pathvqa_llm_judge import (
    PROMPT_VERSION,
    cache_key,
    parse_judgment,
    semantic_inputs,
    served_model_matches,
)


class IncompleteEvaluationError(RuntimeError):
    """Inputs are valid so far, but not every required case has a verdict."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path, *, required: bool = True) -> list[dict[str, Any]]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSON row at {path}:{line_number}")
            rows.append(value)
    return rows


def answer_type(row: dict[str, Any]) -> str:
    score = row.get("score")
    value = row.get("answer_type")
    if value is None and isinstance(score, dict):
        value = score.get("answer_type")
    value = str(value or "")
    if value not in {"yes_no", "free_form"}:
        raise ValueError(
            f"invalid answer_type at prediction index {row.get('index')}: {value!r}"
        )
    return value


def numeric_usage_totals(judgments: list[dict[str, Any]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for row in judgments:
        usage = row.get("usage")
        if not isinstance(usage, dict):
            continue
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(name)
            if isinstance(value, int) and value >= 0:
                totals[name] += value
    return dict(totals)


def validate_source_metrics(
    metrics: dict[str, Any], predictions: list[dict[str, Any]]
) -> None:
    passed = sum(row.get("status") == "passed" for row in predictions)
    expected_values = {
        "expected_count": len(predictions),
        "recorded_count": len(predictions),
        "passed_count": passed,
        "failed_count": len(predictions) - passed,
        "yes_no_count": sum(
            row.get("status") == "passed" and answer_type(row) == "yes_no"
            for row in predictions
        ),
        "free_form_count": sum(
            row.get("status") == "passed" and answer_type(row) == "free_form"
            for row in predictions
        ),
    }
    mismatches = {
        name: {"expected": value, "actual": metrics.get(name)}
        for name, value in expected_values.items()
        if metrics.get(name) != value
    }
    if mismatches:
        raise ValueError(f"source metrics disagree with predictions: {mismatches}")


def finalize(
    *,
    predictions_path: Path,
    judgments_path: Path,
    skipped_path: Path,
    failures_path: Path,
    source_metrics_path: Path,
    output_path: Path,
    judge_model: str,
) -> dict[str, Any]:
    predictions = load_jsonl(predictions_path)
    judgments = load_jsonl(judgments_path)
    skipped = load_jsonl(skipped_path, required=False)
    failures = load_jsonl(failures_path, required=False)
    source_metrics = json.loads(source_metrics_path.read_text(encoding="utf-8"))
    if not isinstance(source_metrics, dict):
        raise ValueError("source metrics must be a JSON object")
    validate_source_metrics(source_metrics, predictions)

    prediction_indices = [int(row["index"]) for row in predictions]
    if len(set(prediction_indices)) != len(prediction_indices):
        raise ValueError("prediction indices are not unique")
    if set(prediction_indices) != set(range(len(predictions))):
        raise ValueError("prediction indices are not the exact contiguous evaluation range")

    judgment_by_key: dict[str, dict[str, Any]] = {}
    served_models: set[str] = set()
    for row in judgments:
        if row.get("prompt_version") != PROMPT_VERSION:
            raise ValueError("judgment prompt version mismatch")
        if row.get("model") != judge_model:
            raise ValueError("judgment requested model mismatch")
        if not served_model_matches(judge_model, row.get("served_model")):
            raise ValueError(f"judgment served model mismatch: {row.get('served_model')!r}")
        if row.get("answer_type") != "free_form":
            raise ValueError("non-free-form judgment record")
        parse_judgment(canonical_json(row.get("judgment")))
        key = str(row.get("cache_key") or "")
        if len(key) != 64:
            raise ValueError("invalid judgment cache key")
        if key in judgment_by_key:
            raise ValueError(f"duplicate judgment cache key: {key}")
        judgment_by_key[key] = row
        if row.get("served_model") is not None:
            served_models.add(str(row["served_model"]))

    skipped_identities: set[tuple[int, Any, str]] = set()
    for row in skipped:
        identity = (
            int(row["index"]), row.get("source_record_sha256"), str(row.get("model"))
        )
        if identity in skipped_identities:
            raise ValueError(f"duplicate skipped identity: {identity}")
        if identity[2] != judge_model:
            raise ValueError("skipped record model mismatch")
        skipped_identities.add(identity)

    required_keys: set[str] = set()
    failed_prediction_keys: set[str] = set()
    expected_skips: set[tuple[int, Any, str]] = set()
    case_rows: list[dict[str, Any]] = []
    yes_no_total = yes_no_correct = 0
    free_form_total = free_form_correct = 0
    free_form_judged_cases = 0
    strict_exact_free_form_correct = 0
    error_types: Counter[str] = Counter()

    for row in predictions:
        index = int(row["index"])
        kind = answer_type(row)
        passed = row.get("status") == "passed"
        source_sha = row.get("source_record_sha256")
        if kind == "yes_no":
            yes_no_total += 1
            correct = False
            if passed:
                score = row.get("score")
                if (
                    not isinstance(score, dict)
                    or type(score.get("contract_aligned_exact_match")) is not bool
                ):
                    raise ValueError(f"missing yes/no contract score at index {index}")
                correct = score["contract_aligned_exact_match"]
            yes_no_correct += int(correct)
            case_rows.append({"index": index, "answer_type": kind, "correct": correct})
            continue

        free_form_total += 1
        if not passed:
            identity = (index, source_sha, judge_model)
            expected_skips.add(identity)
            if "completion" in row:
                question, reference, candidate, _ = semantic_inputs(row)
                failed_prediction_keys.add(
                    cache_key(question, reference, candidate, judge_model)
                )
            case_rows.append(
                {
                    "index": index,
                    "answer_type": kind,
                    "correct": False,
                    "status": "prediction_failed_semantic_judge_skipped",
                }
            )
            continue

        question, reference, candidate, extraction_source = semantic_inputs(row)
        key = cache_key(question, reference, candidate, judge_model)
        required_keys.add(key)
        judgment = judgment_by_key.get(key)
        if judgment is None:
            case_rows.append(
                {
                    "index": index,
                    "answer_type": kind,
                    "cache_key": key,
                    "status": "semantic_judgment_pending",
                }
            )
            continue
        if judgment.get("candidate") != candidate:
            raise ValueError(f"judgment candidate mismatch for cache key {key}")
        if judgment.get("candidate_extraction_source") != extraction_source:
            raise ValueError(f"candidate extraction source mismatch for cache key {key}")
        verdict = parse_judgment(canonical_json(judgment["judgment"]))
        correct = verdict["correct"]
        free_form_judged_cases += 1
        free_form_correct += int(correct)
        error_types[verdict["error_type"]] += 1
        score = row.get("score")
        strict_exact = bool(isinstance(score, dict) and score.get("strict_exact_match"))
        strict_exact_free_form_correct += int(strict_exact)
        case_rows.append(
            {
                "index": index,
                "answer_type": kind,
                "cache_key": key,
                "correct": correct,
                "error_type": verdict["error_type"],
            }
        )

    missing_keys = sorted(required_keys - set(judgment_by_key))
    extra_keys = sorted(set(judgment_by_key) - required_keys)
    unexpected_extra_keys = sorted(set(extra_keys) - failed_prediction_keys)
    excluded_failed_prediction_keys = sorted(set(extra_keys) & failed_prediction_keys)
    missing_skips = sorted(expected_skips - skipped_identities)
    extra_skips = sorted(skipped_identities - expected_skips)
    if unexpected_extra_keys or extra_skips:
        raise ValueError(
            "unexpected semantic records: "
            f"extra_keys={len(unexpected_extra_keys)}, "
            f"extra_skips={len(extra_skips)}"
        )
    if missing_keys or missing_skips:
        raise IncompleteEvaluationError(
            f"semantic coverage incomplete: missing_keys={len(missing_keys)}, "
            f"missing_skips={len(missing_skips)}"
        )
    if free_form_judged_cases + len(expected_skips) != free_form_total:
        raise RuntimeError("free-form case accounting invariant failed")

    if source_metrics.get("yes_no_contract_aligned_correct") != yes_no_correct:
        raise ValueError(
            "yes/no correct-count mismatch: "
            f"source={source_metrics.get('yes_no_contract_aligned_correct')}, "
            f"recomputed={yes_no_correct}"
        )

    total_correct = yes_no_correct + free_form_correct
    successful_attempts = sum(
        len(row.get("attempts", [])) if isinstance(row.get("attempts"), list) else 0
        for row in judgments
    )
    failure_attempts = sum(
        len(row.get("attempts", [])) if isinstance(row.get("attempts"), list) else 0
        for row in failures
    )
    result = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "formal_result": bool(source_metrics.get("formal_result")),
        "result_class": "authorized_full_contemporary_baseline_evaluation_semantic_final",
        "status": "completed",
        "model": source_metrics.get("model"),
        "task": "pathvqa",
        "expected_count": len(predictions),
        "total_correct": total_correct,
        "overall_accuracy": total_correct / len(predictions),
        "yes_no": {
            "total_cases": yes_no_total,
            "correct": yes_no_correct,
            "accuracy": yes_no_correct / yes_no_total if yes_no_total else None,
        },
        "free_form": {
            "total_cases": free_form_total,
            "passed_prediction_cases_judged": free_form_judged_cases,
            "failed_prediction_cases_scored_wrong": len(expected_skips),
            "unique_semantic_cache_keys": len(required_keys),
            "correct": free_form_correct,
            "accuracy_all_cases": free_form_correct / free_form_total if free_form_total else None,
            "accuracy_judged_predictions": (
                free_form_correct / free_form_judged_cases if free_form_judged_cases else None
            ),
            "strict_exact_correct_diagnostic": strict_exact_free_form_correct,
            "error_type_case_counts": dict(sorted(error_types.items())),
        },
        "semantic_judge": {
            "requested_model": judge_model,
            "served_models": sorted(served_models),
            "prompt_version": PROMPT_VERSION,
            "unique_judgment_records": len(judgments),
            "used_unique_judgments": len(required_keys),
            "excluded_historical_judgments_for_failed_predictions": len(
                excluded_failed_prediction_keys
            ),
            "successful_physical_attempts": successful_attempts,
            "historical_failure_records": len(failures),
            "historical_failed_physical_attempts": failure_attempts,
            "skipped_failed_predictions": len(skipped),
            "usage_totals_over_unique_judgments": numeric_usage_totals(judgments),
        },
        "artifacts": {
            "predictions": str(predictions_path.resolve()),
            "predictions_sha256": sha256_file(predictions_path),
            "source_metrics": str(source_metrics_path.resolve()),
            "source_metrics_sha256": sha256_file(source_metrics_path),
            "semantic_judgments": str(judgments_path.resolve()),
            "semantic_judgments_sha256": sha256_file(judgments_path),
            "skipped_predictions": str(skipped_path.resolve()),
            "skipped_predictions_sha256": (
                sha256_file(skipped_path) if skipped_path.is_file() else None
            ),
            "failure_audit": str(failures_path.resolve()),
            "failure_audit_sha256": (
                sha256_file(failures_path) if failures_path.is_file() else None
            ),
            "case_result_sha256": hashlib.sha256(
                canonical_json(case_rows).encode("utf-8")
            ).hexdigest(),
        },
        "pathmmu_test999_accessed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--source-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--skipped", type=Path)
    parser.add_argument("--failures", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skipped = args.skipped or args.judgments.with_name(
        args.judgments.stem + "_skipped_predictions.jsonl"
    )
    failures = args.failures or args.judgments.with_name(
        args.judgments.stem + "_failures.jsonl"
    )
    try:
        result = finalize(
            predictions_path=args.predictions,
            judgments_path=args.judgments,
            skipped_path=skipped,
            failures_path=failures,
            source_metrics_path=args.source_metrics,
            output_path=args.output,
            judge_model=args.model,
        )
    except IncompleteEvaluationError as exc:
        print(json.dumps({"status": "incomplete", "error": str(exc)}), file=sys.stderr)
        raise SystemExit(3) from exc
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
