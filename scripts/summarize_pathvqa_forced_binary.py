#!/usr/bin/env python3
"""Paired summary for a directory of PathVQA forced-binary model runs."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from external_vqa_contract import sha256_file


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def bootstrap(base: np.ndarray, model: np.ndarray, seed: int, draws: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    for start in range(0, draws, 500):
        size = min(500, draws - start)
        indices = rng.integers(0, len(base), size=(size, len(base)))
        values.extend((model[indices].mean(axis=1) - base[indices].mean(axis=1)).tolist())
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--base-name", default="base")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = sorted(args.run_root.glob("*/original/predictions.jsonl"))
    if not paths:
        raise ValueError("no completed forced-binary predictions found")
    runs = {path.parent.parent.name: (path, load(path)) for path in paths}
    if args.base_name not in runs:
        raise ValueError("base run is missing")
    base_path, base_rows = runs[args.base_name]
    base_correct = np.asarray([row["forced_binary_correct"] for row in base_rows], dtype=float)
    summaries = {}
    for name, (path, rows) in runs.items():
        if len(rows) != len(base_rows) or any(
            row["source_record_sha256"] != reference["source_record_sha256"]
            for row, reference in zip(rows, base_rows)
        ):
            raise ValueError(f"run alignment failure: {name}")
        correct = np.asarray([row["forced_binary_correct"] for row in rows], dtype=float)
        base_only = int(np.sum((base_correct == 1) & (correct == 0)))
        model_only = int(np.sum((base_correct == 0) & (correct == 1)))
        discordant = base_only + model_only
        summaries[name] = {
            "predictions": str(path.resolve()),
            "predictions_sha256": sha256_file(path),
            "count": len(rows),
            "coverage": 1.0,
            "correct": int(correct.sum()),
            "accuracy": float(correct.mean()),
            "predicted_yes_rate": sum(row["forced_binary_answer"] == "yes" for row in rows) / len(rows),
            "mean_target_margin": float(np.mean([row["target_margin"] for row in rows])),
            "base_correct_model_wrong": base_only,
            "base_wrong_model_correct": model_only,
            "mcnemar_exact_two_sided_p_vs_base": (
                float(binomtest(min(base_only, model_only), discordant, 0.5).pvalue)
                if discordant else 1.0
            ),
            "accuracy_difference_vs_base": float(correct.mean() - base_correct.mean()),
            "paired_bootstrap_95_ci_difference_vs_base": bootstrap(base_correct, correct, args.seed),
        }
    pairwise = {}
    for left, right in combinations(sorted(runs), 2):
        left_correct = np.asarray(
            [row["forced_binary_correct"] for row in runs[left][1]], dtype=float
        )
        right_correct = np.asarray(
            [row["forced_binary_correct"] for row in runs[right][1]], dtype=float
        )
        left_only = int(np.sum((left_correct == 1) & (right_correct == 0)))
        right_only = int(np.sum((left_correct == 0) & (right_correct == 1)))
        discordant = left_only + right_only
        pairwise[f"{left}__to__{right}"] = {
            "right_minus_left_accuracy": float(right_correct.mean() - left_correct.mean()),
            "paired_bootstrap_95_ci": bootstrap(left_correct, right_correct, args.seed),
            "left_correct_right_wrong": left_only,
            "left_wrong_right_correct": right_only,
            "mcnemar_exact_two_sided_p": (
                float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue)
                if discordant else 1.0
            ),
        }
    payload = {
        "schema_version": 1,
        "status": "completed",
        "formal_result": False,
        "result_role": "format_neutral_forced_binary_sensitivity",
        "scientific_boundary": "This compares constrained Yes/No next-token scores and is not a free-generation metric or a visual attribution method.",
        "run_root": str(args.run_root.resolve()),
        "base_name": args.base_name,
        "target_yes_rate": sum(row["target"] == "yes" for row in base_rows) / len(base_rows),
        "bootstrap_seed": args.seed,
        "models": summaries,
        "all_pairwise_post_hoc": pairwise,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({name: {"accuracy": value["accuracy"], "p": value["mcnemar_exact_two_sided_p_vs_base"]} for name, value in summaries.items()}))


if __name__ == "__main__":
    main()
