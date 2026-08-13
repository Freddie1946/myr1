#!/usr/bin/env python3
"""Summarize fixed-seed repeated inference without conflating it with case bootstrap."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


T975 = {2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582,
        6: 2.446912, 7: 2.364624, 8: 2.306004, 9: 2.262157}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-runs", type=int, default=5)
    args = parser.parse_args()
    paths = sorted(args.root.glob("seed_*/metrics.json"))
    if len(paths) != args.expected_runs:
        raise ValueError(f"expected {args.expected_runs} completed runs, got {len(paths)}")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    seeds = [int(row["seed"]) for row in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate repeat seeds")
    accuracies = [float(row["accuracy"]) for row in rows]
    mean = statistics.fmean(accuracies)
    sd = statistics.stdev(accuracies)
    n = len(accuracies)
    critical = T975.get(n - 1)
    if critical is None:
        raise ValueError("supported repeated-run count is 3 through 10")
    half = critical * sd / math.sqrt(n)
    result = {
        "schema_version": 1,
        "method": "fixed-seed stochastic inference repeats; Student-t CI across run accuracies",
        "run_count": n,
        "seeds": seeds,
        "accuracies": accuracies,
        "mean_accuracy": mean,
        "sample_standard_deviation": sd,
        "student_t_95_ci": [max(0.0, mean - half), min(1.0, mean + half)],
        "generation_contract": {
            "do_sample": True, "temperature": 0.7, "top_p": 0.9,
            "top_k": 0, "max_new_tokens": 1024,
        },
        "important_boundary": (
            "This interval quantifies decoding-seed variability. It is separate from the "
            "case/bootstrap uncertainty of benchmark accuracy."
        ),
        "runs": [{"path": str(path.resolve()), "metrics": row} for path, row in zip(paths, rows)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
