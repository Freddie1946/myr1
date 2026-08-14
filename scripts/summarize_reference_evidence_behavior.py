#!/usr/bin/env python3
"""Summarize paired reference/neighbor/random evidence-deletion behavior."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap(values: np.ndarray, seed: int, samples: int) -> dict:
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(samples)])
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "positive_count": int((values > 0).sum()),
            "bootstrap_mean_ci95": [float(x) for x in np.quantile(means, [.025, .975])]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--behavior", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--bootstrap-samples", type=int, default=100000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    panel_raw = args.panel.read_bytes(); panel = json.loads(panel_raw)
    expected = {int(case["panel_index"]) for case in panel["cases"]}
    rows = {}; sources = {}
    for path in args.behavior:
        sources[str(path.resolve())] = sha256_file(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line); index = int(row["panel_index"])
            if index in rows:
                raise ValueError(f"duplicate behavior row {index}")
            rows[index] = row
    if set(rows) != expected:
        raise ValueError("behavior coverage does not match panel")
    primary = [row for row in rows.values()
               if row["clean_reasoning_correct"] and row["clean_reasoning_forced_agreement"]]
    effective = [row for row in primary if row["reference_target_margin_drop"] > 0
                 and row["reference_minus_random_extra_drop"] > 0]
    keys = ["reference_target_margin_drop", "random_mean_target_margin_drop",
            "neighbor_target_margin_drop", "reference_minus_random_extra_drop"]
    output = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "status": "completed",
        "panel": str(args.panel.resolve()), "panel_sha256": hashlib.sha256(panel_raw).hexdigest(),
        "behavior_sources": sources, "case_count": len(rows),
        "clean_reasoning_correct_count": sum(row["clean_reasoning_correct"] for row in rows.values()),
        "reasoning_forced_agreement_count_all": sum(row["clean_reasoning_forced_agreement"] for row in rows.values()),
        "primary_clean_correct_and_agreement_count": len(primary),
        "reference_reasoning_flip_correct_to_wrong_count": sum(
            not row["conditions"]["reference_deleted"]["reasoning_correct"] for row in primary),
        "reference_forced_choice_flip_away_count": sum(
            row["conditions"]["reference_deleted"]["forced_choice"] != row["target_choice"] for row in primary),
        "strict_effective_rule": "clean correct and generation/score agreement and reference margin drop > 0 and reference-minus-random extra drop > 0",
        "strict_effective_count": len(effective),
        "strict_effective_choice_counts": dict(Counter(row["target_choice"] for row in effective)),
        "primary_paired_statistics": {
            key: bootstrap(np.array([row[key] for row in primary], dtype=float),
                           args.seed + offset, args.bootstrap_samples)
            for offset, key in enumerate(keys)
        },
        "interpretation_boundary": "The reference-minus-random aggregate CI determines the general matched-control claim; the strict effective panel is a case panel and must not replace the aggregate analysis.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"case_count": len(rows), "primary_count": len(primary),
                      "strict_effective_count": len(effective)}, sort_keys=True))


if __name__ == "__main__":
    main()
