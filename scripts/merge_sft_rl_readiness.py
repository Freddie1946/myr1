#!/usr/bin/env python3
"""Merge and summarize eight deterministic RL-readiness shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from run_pathmmu_qwen_diagnostic import record_sha256, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--shard-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    records = json.loads(args.panel.read_text(encoding="utf-8"))
    rows = []
    shard_hashes = []
    for shard in range(8):
        path = args.shard_root / f"shard_{shard}" / "rollouts.jsonl"
        shard_hashes.append(sha256_file(path))
        rows.extend(json.loads(line) for line in path.open(encoding="utf-8"))
    rows.sort(key=lambda row: row["index"])
    if [row["index"] for row in rows] != list(range(300)):
        raise ValueError("merged shards do not cover indices 0..299 exactly")
    for record, row in zip(records, rows, strict=True):
        if row["source_record_sha256"] != record_sha256(record):
            raise ValueError(f"record mismatch at {row['index']}")
        if len(row["accuracy_rewards"]) != 8:
            raise ValueError(f"rollout count mismatch at {row['index']}")
    args.output_dir.mkdir(parents=True)
    merged = args.output_dir / "rollouts.jsonl"
    with merged.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    passes = [sum(float(value) for value in row["accuracy_rewards"]) for row in rows]
    rates = [value / 8 for value in passes]
    group_stds = [math.sqrt(rate * (1.0 - rate)) for rate in rates]
    metrics = {
        "schema_version": 1,
        "status": "completed",
        "prompt_count": 300,
        "rollouts_per_prompt": 8,
        "rollout_count": 2400,
        "mean_accuracy_reward": statistics.fmean(rates),
        "mean_group_reward_std": statistics.fmean(group_stds),
        "mixed_reward_group_fraction": sum(0 < value < 8 for value in passes) / 300,
        "all_zero_group_fraction": sum(value == 0 for value in passes) / 300,
        "all_one_group_fraction": sum(value == 8 for value in passes) / 300,
        "mean_format_reward": statistics.fmean(
            float(value) for row in rows for value in row["format_rewards"]
        ),
        "generation_cap_hit_count": sum(
            bool(value) for row in rows for value in row["generation_cap_hits"]
        ),
        "panel_sha256": sha256_file(args.panel),
        "source_shard_sha256s": shard_hashes,
        "merged_rollouts_sha256": sha256_file(merged),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
