#!/usr/bin/env python3
"""Paired accuracy, McNemar test and image-cluster bootstrap for closed benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from external_vqa_contract import omnimed_score, pathvqa_score


def load(path: Path, filter_field: str | None, filter_value: str | None) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if filter_field:
        rows = [row for row in rows if str(row.get(filter_field)) == filter_value]
    mapping = {str(row["source_record_sha256"]): row for row in rows}
    if len(mapping) != len(rows):
        raise ValueError(f"duplicate source hashes in {path}")
    return mapping


def mcnemar_exact(left_only: int, right_only: int) -> float:
    n = left_only + right_only
    if n == 0:
        return 1.0
    observed = min(left_only, right_only)
    cdf = sum(math.comb(n, value) for value in range(observed + 1)) / (2 ** n)
    return min(1.0, 2 * cdf)


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def is_correct(row: dict[str, Any], field: str, task: str | None) -> bool:
    if field in row:
        return bool(row[field])
    if task == "pathvqa" and field == "contract_aligned_exact_match":
        return bool(pathvqa_score(row["completion"], row["answer"])[field])
    if task == "omnimedvqa" and field == "contract_aligned_correct":
        record = {
            "question": row["question"], "gt_answer": row["gt_answer"],
            **{f"option_{letter}": row["option_texts"].get(letter) for letter in "ABCD"},
        }
        return bool(omnimed_score(row["completion"], record)[field])
    raise ValueError(f"missing correctness field {field!r} and no compatible task rescorer")


def analyze(
    left: dict[str, dict[str, Any]], right: dict[str, dict[str, Any]], *,
    correctness_field: str, expected_count: int, replicates: int, seed: int,
    task: str | None = None,
) -> dict[str, Any]:
    common = sorted(set(left) & set(right))
    if len(common) != expected_count or len(left) != expected_count or len(right) != expected_count:
        raise ValueError(
            f"expected {expected_count} aligned records; left={len(left)} right={len(right)} common={len(common)}"
        )
    transitions = {"both_correct": 0, "both_wrong": 0, "left_only": 0, "right_only": 0}
    clusters: dict[str, list[str]] = defaultdict(list)
    deltas: dict[str, float] = {}
    left_correct = right_correct = 0
    for key in common:
        if left[key].get("image") != right[key].get("image"):
            raise ValueError(f"image mismatch: {key}")
        a = is_correct(left[key], correctness_field, task)
        b = is_correct(right[key], correctness_field, task)
        left_correct += a
        right_correct += b
        name = "both_correct" if a and b else "left_only" if a else "right_only" if b else "both_wrong"
        transitions[name] += 1
        image = str(Path(left[key]["image"]).resolve())
        clusters[image].append(key)
        deltas[key] = float(b) - float(a)
    names = sorted(clusters)
    rng = random.Random(seed)
    boot = []
    for _ in range(replicates):
        selected = [rng.choice(names) for _ in names]
        keys = [key for image in selected for key in clusters[image]]
        boot.append(sum(deltas[key] for key in keys) / len(keys))
    observed = (right_correct - left_correct) / expected_count
    return {
        "count": expected_count,
        "left_correct": left_correct,
        "right_correct": right_correct,
        "left_accuracy": left_correct / expected_count,
        "right_accuracy": right_correct / expected_count,
        "right_minus_left": observed,
        "transition_counts": transitions,
        "mcnemar_exact_two_sided_p": mcnemar_exact(transitions["left_only"], transitions["right_only"]),
        "unique_image_clusters": len(names),
        "clusters_with_multiple_questions": sum(len(values) > 1 for values in clusters.values()),
        "image_cluster_bootstrap_95_ci": [percentile(boot, 0.025), percentile(boot, 0.975)],
        "image_cluster_bootstrap_two_sided_p": min(
            1.0,
            2 * min(sum(value <= 0 for value in boot) / replicates, sum(value >= 0 for value in boot) / replicates),
        ),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--correctness-field", required=True)
    parser.add_argument("--task", choices=("pathvqa", "omnimedvqa"))
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--filter-field")
    parser.add_argument("--filter-value")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.filter_field) != bool(args.filter_value):
        raise ValueError("filter field and value must be provided together")
    if args.replicates < 1000:
        raise ValueError("at least 1000 bootstrap replicates are required")
    result = analyze(
        load(args.left, args.filter_field, args.filter_value),
        load(args.right, args.filter_field, args.filter_value),
        correctness_field=args.correctness_field, expected_count=args.expected_count,
        replicates=args.replicates, seed=args.seed, task=args.task,
    )
    result.update({
        "schema_version": 1, "benchmark": args.benchmark,
        "left_label": args.left_label, "right_label": args.right_label,
        "left_predictions": str(args.left.resolve()), "right_predictions": str(args.right.resolve()),
        "correctness_field": args.correctness_field,
        "filter": {args.filter_field: args.filter_value} if args.filter_field else None,
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    low, high = result["image_cluster_bootstrap_95_ci"]
    lines = [
        f"# {args.benchmark}: paired {args.left_label} versus {args.right_label}", "",
        f"- {args.left_label}: {result['left_correct']}/{result['count']} ({result['left_accuracy']:.2%})",
        f"- {args.right_label}: {result['right_correct']}/{result['count']} ({result['right_accuracy']:.2%})",
        f"- Difference: {result['right_minus_left']:+.2%}",
        f"- Paired flips (right improvements / regressions): {result['transition_counts']['right_only']} / {result['transition_counts']['left_only']}",
        f"- Exact McNemar p: {result['mcnemar_exact_two_sided_p']:.4g}",
        f"- Image-cluster bootstrap 95% CI: [{low:+.2%}, {high:+.2%}]",
        f"- Image-cluster bootstrap p: {result['image_cluster_bootstrap_two_sided_p']:.4g}",
        f"- Unique image clusters: {result['unique_image_clusters']}", "",
    ]
    (args.output_dir / "paired_results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
