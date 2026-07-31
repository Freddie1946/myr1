#!/usr/bin/env python3
"""Re-score cached Stage3 six-event judgments under alternative penalties.

This is a strictly offline diagnostic: it reads immutable judge cache records and
never imports or invokes a network client.  Smoke records are excluded by default.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def process_score(missing_integrity: int, present_errors: int, penalty: float) -> float:
    integrity = max(0.0, 1.0 - penalty * missing_integrity)
    knowledge = max(0.0, 1.0 - penalty * present_errors)
    return (integrity + knowledge) / 2.0


def midranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_sum = sum((a - left_mean) ** 2 for a in left)
    right_sum = sum((b - right_mean) ** 2 for b in right)
    denominator = math.sqrt(left_sum * right_sum)
    return None if denominator == 0 else numerator / denominator


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(midranks(left), midranks(right))


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    location = probability * (len(ordered) - 1)
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (location - lower) * (ordered[upper] - ordered[lower])


def parse_run(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("--run must be NAME=PATH")
    return name, Path(path)


def parse_coefficients(value: str) -> list[float]:
    result = [float(item) for item in value.split(",")]
    if not result or any(not 0.0 < item <= 1.0 for item in result):
        raise argparse.ArgumentTypeError("coefficients must lie in (0, 1]")
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("coefficients must be unique")
    return result


def load_records(cache_root: Path) -> tuple[list[dict[str, Any]], int]:
    files = sorted(cache_root.glob("*/*.json"))
    if not files:
        files = sorted(cache_root.rglob("*.json"))
    records: list[dict[str, Any]] = []
    smoke_count = 0
    seen_cache_keys: set[str] = set()
    for path in files:
        value = json.loads(path.read_text(encoding="utf-8"))
        source = value.get("source")
        scores = value.get("scores")
        if not isinstance(source, dict) or not isinstance(scores, dict):
            raise ValueError(f"cache record lacks source/scores: {path}")
        if source.get("smoke") is True:
            smoke_count += 1
            continue
        cache_key = value.get("cache_key")
        if not isinstance(cache_key, str) or cache_key in seen_cache_keys:
            raise ValueError(f"missing or duplicate cache key: {path}")
        seen_cache_keys.add(cache_key)
        missing = scores.get("missing_integrity_count")
        errors = scores.get("present_knowledge_error_count")
        if type(missing) is not int or type(errors) is not int:
            raise ValueError(f"event counts are not integers: {path}")
        if missing not in range(4) or errors not in range(4):
            raise ValueError(f"event counts outside 0..3: {path}")
        records.append(
            {
                "cache_key": cache_key,
                "missing": missing,
                "errors": errors,
                "group": (
                    str(source.get("training_segment", "unspecified")),
                    int(source.get("record_index", -1)),
                    str(source.get("problem_sha256", "")),
                ),
            }
        )
    if not records:
        raise ValueError(f"no non-smoke cache records found under {cache_root}")
    return records, smoke_count


def sign(value: float, tolerance: float = 1e-12) -> int:
    if abs(value) <= tolerance:
        return 0
    return 1 if value > 0 else -1


def summarize_groups(
    records: list[dict[str, Any]], scores: list[float], reference: list[float]
) -> dict[str, Any]:
    groups: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[record["group"]].append(index)
    eligible = [indices for indices in groups.values() if len(indices) >= 2]
    score_zero_variance = 0
    reference_zero_variance = 0
    sign_equal = 0
    sign_total = 0
    jaccards: list[float] = []
    for indices in eligible:
        current = [scores[index] for index in indices]
        baseline = [reference[index] for index in indices]
        current_mean = statistics.fmean(current)
        baseline_mean = statistics.fmean(baseline)
        if max(current) == min(current):
            score_zero_variance += 1
        if max(baseline) == min(baseline):
            reference_zero_variance += 1
        for left, right in zip(current, baseline):
            sign_equal += sign(left - current_mean) == sign(right - baseline_mean)
            sign_total += 1
        current_top = {i for i, value in enumerate(current) if value == max(current)}
        baseline_top = {i for i, value in enumerate(baseline) if value == max(baseline)}
        jaccards.append(len(current_top & baseline_top) / len(current_top | baseline_top))
    return {
        "group_key": ["training_segment", "record_index", "problem_sha256"],
        "all_group_count": len(groups),
        "eligible_group_count_size_ge_2": len(eligible),
        "group_size_distribution": {
            str(size): sum(len(indices) == size for indices in groups.values())
            for size in sorted({len(indices) for indices in groups.values()})
        },
        "zero_variance_group_count": score_zero_variance,
        "reference_zero_variance_group_count": reference_zero_variance,
        "centered_advantage_sign_agreement_with_reference": (
            sign_equal / sign_total if sign_total else None
        ),
        "mean_top_reward_set_jaccard_with_reference": (
            statistics.fmean(jaccards) if jaccards else None
        ),
    }


def summarize_run(
    name: str,
    cache_root: Path,
    coefficients: Iterable[float],
    reference_coefficient: float,
) -> dict[str, Any]:
    records, smoke_count = load_records(cache_root)
    coefficients = list(coefficients)
    scores_by_coefficient = {
        coefficient: [
            process_score(record["missing"], record["errors"], coefficient)
            for record in records
        ]
        for coefficient in coefficients
    }
    reference = scores_by_coefficient[reference_coefficient]
    summaries: dict[str, Any] = {}
    for coefficient, values in scores_by_coefficient.items():
        summaries[f"{coefficient:g}"] = {
            "mean": statistics.fmean(values),
            "population_stddev": statistics.pstdev(values),
            "minimum": min(values),
            "q25": quantile(values, 0.25),
            "median": quantile(values, 0.5),
            "q75": quantile(values, 0.75),
            "maximum": max(values),
            "zero_fraction": sum(value == 0.0 for value in values) / len(values),
            "one_fraction": sum(value == 1.0 for value in values) / len(values),
            "unique_score_count": len(set(values)),
            "spearman_with_reference": spearman(values, reference),
            "group_robustness": summarize_groups(records, values, reference),
        }
    event_count_distribution: dict[str, int] = defaultdict(int)
    for record in records:
        event_count_distribution[f"missing={record['missing']},errors={record['errors']}"] += 1
    return {
        "name": name,
        "cache_root": str(cache_root.resolve()),
        "non_smoke_record_count": len(records),
        "excluded_smoke_record_count": smoke_count,
        "event_count_distribution": dict(sorted(event_count_distribution.items())),
        "coefficients": summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--coefficients", type=parse_coefficients, default=parse_coefficients("0.2,0.3,0.4,0.5,0.6"))
    parser.add_argument("--reference-coefficient", type=float, default=0.4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reference_coefficient not in args.coefficients:
        raise ValueError("reference coefficient must be among --coefficients")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite existing analysis: {args.output}")
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_type": "offline_cached_six_event_penalty_sensitivity",
        "network_calls": 0,
        "paid_calls": 0,
        "reference_coefficient": args.reference_coefficient,
        "coefficients": args.coefficients,
        "runs": [
            summarize_run(name, path, args.coefficients, args.reference_coefficient)
            for name, path in args.run
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "status": "complete",
        "runs": {run["name"]: run["non_smoke_record_count"] for run in payload["runs"]},
        "output": str(args.output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
