#!/usr/bin/env python3
"""Aggregate completed blind multi-judge reasoning evaluations."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRICS = ("r_acc", "k_acc", "rigor", "professionalism", "clarity", "conciseness")


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["judge"]), str(row["candidate_label"]), int(row["index"]), int(row["run"]))
            if key in seen:
                raise ValueError(f"duplicate judgment key: {key}")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError("no judgments found")
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap(values: list[float], *, seed: int, replicates: int) -> dict[str, float]:
    rng = random.Random(seed)
    n = len(values)
    draws = [mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(replicates)]
    return {
        "mean_delta": mean(values),
        "ci95_low": percentile(draws, 0.025),
        "ci95_high": percentile(draws, 0.975),
    }


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left) * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def icc_one_way(values: list[list[float]]) -> float | None:
    """ICC(1,1) for equal-size repeated ratings; may legitimately be negative."""
    if len(values) < 2 or not values or len({len(row) for row in values}) != 1:
        return None
    k = len(values[0])
    if k < 2:
        return None
    unit_means = [mean(row) for row in values]
    grand = mean([value for row in values for value in row])
    ms_between = k * sum((value - grand) ** 2 for value in unit_means) / (len(values) - 1)
    ms_within = sum(
        sum((value - unit_mean) ** 2 for value in row)
        for row, unit_mean in zip(values, unit_means)
    ) / (len(values) * (k - 1))
    denominator = ms_between + (k - 1) * ms_within
    return (ms_between - ms_within) / denominator if denominator else None


def analyze(
    rows: list[dict[str, Any]], *, stage2_label: str, stage3_label: str,
    expected_per_judge: int, bootstrap_replicates: int, seed: int,
) -> dict[str, Any]:
    judges = sorted({str(row["judge"]) for row in rows})
    labels = sorted({str(row["candidate_label"]) for row in rows})
    judge_counts = {judge: sum(str(row["judge"]) == judge for row in rows) for judge in judges}
    primary = [row for row in rows if int(row["run"]) == 0]

    model_scores: dict[str, Any] = {}
    for judge in judges:
        model_scores[judge] = {}
        for label in labels:
            subset = [row for row in primary if row["judge"] == judge and row["candidate_label"] == label]
            if not subset:
                continue
            metric_means = {metric: mean([float(row["scores"][metric]) for row in subset]) for metric in METRICS}
            metric_means["six_metric_macro"] = mean(list(metric_means.values()))
            model_scores[judge][label] = {"n": len(subset), "means": metric_means}

    paired: dict[str, Any] = {}
    for judge_number, judge in enumerate(judges):
        maps = {
            label: {int(row["index"]): row for row in primary if row["judge"] == judge and row["candidate_label"] == label}
            for label in (stage2_label, stage3_label)
        }
        common = sorted(set(maps[stage2_label]) & set(maps[stage3_label]))
        if not common:
            continue
        paired[judge] = {"n": len(common), "stage3_minus_stage2": {}}
        for metric_number, metric in enumerate((*METRICS, "six_metric_macro")):
            deltas = []
            for index in common:
                before = maps[stage2_label][index]["scores"]
                after = maps[stage3_label][index]["scores"]
                if metric == "six_metric_macro":
                    before_value = mean([float(before[name]) for name in METRICS])
                    after_value = mean([float(after[name]) for name in METRICS])
                else:
                    before_value, after_value = float(before[metric]), float(after[metric])
                deltas.append(after_value - before_value)
            paired[judge]["stage3_minus_stage2"][metric] = paired_bootstrap(
                deltas, seed=seed + judge_number * 100 + metric_number, replicates=bootstrap_replicates
            )

    repeated: dict[str, Any] = {}
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["judge"]), str(row["candidate_label"]), int(row["index"]))].append(row)
    for judge in judges:
        units = [sorted(value, key=lambda row: int(row["run"])) for key, value in grouped.items() if key[0] == judge and len(value) == 3]
        repeated[judge] = {"three_run_units": len(units), "metrics": {}}
        for metric in METRICS:
            matrix = [[float(row["scores"][metric]) for row in unit] for unit in units]
            pair_diffs = [abs(values[a] - values[b]) for values in matrix for a, b in ((0, 1), (0, 2), (1, 2))]
            repeated[judge]["metrics"][metric] = {
                "mean_pairwise_absolute_difference": mean(pair_diffs) if pair_diffs else None,
                "icc_1_1": icc_one_way(matrix),
            }

    interjudge: dict[str, Any] = {}
    primary_map = {
        (str(row["judge"]), str(row["candidate_label"]), int(row["index"])): row for row in primary
    }
    for i, left in enumerate(judges):
        for right in judges[i + 1:]:
            common_keys = sorted(
                {(label, index) for judge, label, index in primary_map if judge == left}
                & {(label, index) for judge, label, index in primary_map if judge == right}
            )
            pair_name = f"{left} vs {right}"
            interjudge[pair_name] = {"n": len(common_keys), "pearson": {}}
            for metric in METRICS:
                left_values = [float(primary_map[(left, label, index)]["scores"][metric]) for label, index in common_keys]
                right_values = [float(primary_map[(right, label, index)]["scores"][metric]) for label, index in common_keys]
                interjudge[pair_name]["pearson"][metric] = pearson(left_values, right_values)

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "complete": all(count == expected_per_judge for count in judge_counts.values()),
        "expected_per_judge": expected_per_judge,
        "judge_result_counts": judge_counts,
        "labels": labels,
        "primary_run_model_scores": model_scores,
        "paired_stage3_vs_stage2": paired,
        "three_run_reliability": repeated,
        "interjudge_agreement": interjudge,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": seed,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = ["# Blind multi-judge reasoning evaluation", ""]
    state = "complete" if result["complete"] else "INCOMPLETE / interim"
    lines += [f"Status: **{state}**. Expected {result['expected_per_judge']} rows per judge.", ""]
    lines += ["| Judge | Rows |", "|---|---:|"]
    for judge, count in result["judge_result_counts"].items():
        lines.append(f"| {judge} | {count} |")
    lines += ["", "## Primary run means", ""]
    lines += ["| Judge | Model | n | R-Acc | K-Acc | Rigor | Professionalism | Clarity | Conciseness | Macro |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for judge, models in result["primary_run_model_scores"].items():
        for label, item in models.items():
            values = item["means"]
            lines.append(
                f"| {judge} | {label} | {item['n']} | "
                + " | ".join(f"{values[name]:.4f}" for name in (*METRICS, "six_metric_macro")) + " |"
            )
    lines += ["", "## Paired Stage3 minus Stage2", ""]
    lines += ["| Judge | n | Metric | Mean delta | Bootstrap 95% CI |", "|---|---:|---|---:|---:|"]
    for judge, item in result["paired_stage3_vs_stage2"].items():
        for metric, estimate in item["stage3_minus_stage2"].items():
            lines.append(
                f"| {judge} | {item['n']} | {metric} | {estimate['mean_delta']:+.4f} | "
                f"[{estimate['ci95_low']:+.4f}, {estimate['ci95_high']:+.4f}] |"
            )
    lines += ["", "## Three-run judge reliability", ""]
    lines.append("ICC(1,1) can be negative when within-item variation exceeds between-item variation.")
    lines += ["", "| Judge | Three-run units | Metric | Mean pairwise abs. diff | ICC(1,1) |", "|---|---:|---|---:|---:|"]
    for judge, item in result["three_run_reliability"].items():
        for metric, values in item["metrics"].items():
            icc = values["icc_1_1"]
            icc_text = f"{icc:.4f}" if icc is not None else "NA"
            lines.append(
                f"| {judge} | {item['three_run_units']} | {metric} | "
                f"{values['mean_pairwise_absolute_difference']:.4f} | {icc_text} |"
            )
    lines += ["", "## Inter-judge agreement", ""]
    lines += ["| Judge pair | n | Metric | Pearson r |", "|---|---:|---|---:|"]
    for pair, item in result["interjudge_agreement"].items():
        for metric, value in item["pearson"].items():
            lines.append(f"| {pair} | {item['n']} | {metric} | {value:.4f} |" if value is not None else f"| {pair} | {item['n']} | {metric} | NA |")
    lines += ["", "Primary inference uses run 0 only. Repeated runs measure judge stability and are not pooled as independent cases.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgments", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--stage2-label", default="Stage2-outcome-GRPO")
    parser.add_argument("--stage3-label", default="Stage3-GPT4o-selected")
    parser.add_argument("--expected-per-judge", type=int, default=800)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    result = analyze(
        load_rows(args.judgments), stage2_label=args.stage2_label, stage3_label=args.stage3_label,
        expected_per_judge=args.expected_per_judge, bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(result), encoding="utf-8")


if __name__ == "__main__":
    main()
