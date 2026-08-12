#!/usr/bin/env python3
"""Merge and summarize the eight PathMMU split-audit shards."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

from run_pathmmu_qwen_diagnostic import sha256_file
from run_pathmmu_split_passk_shard import digest, selected_records


def summarize(rows: list[dict], k: int = 8) -> dict:
    passes = [sum(float(value) for value in row["accuracy_rewards"]) for row in rows]
    rates = [value / k for value in passes]
    histogram = Counter(int(value) for value in passes)
    return {
        "prompt_count": len(rows), "rollout_count": len(rows) * k,
        "mean_rollout_accuracy": statistics.fmean(rates),
        "pass_at_8_any_correct": sum(value > 0 for value in passes) / len(rows),
        "all_zero_fraction": sum(value == 0 for value in passes) / len(rows),
        "all_one_fraction": sum(value == k for value in passes) / len(rows),
        "mixed_fraction": sum(0 < value < k for value in passes) / len(rows),
        "too_hard_0_or_1_fraction": sum(value <= 1 for value in passes) / len(rows),
        "sweet_spot_2_to_6_fraction": sum(2 <= value <= 6 for value in passes) / len(rows),
        "too_easy_7_or_8_fraction": sum(value >= 7 for value in passes) / len(rows),
        "mean_group_reward_std": statistics.fmean(
            math.sqrt(rate * (1 - rate)) for rate in rates
        ),
        "mean_format_reward": statistics.fmean(
            float(value) for row in rows for value in row["format_rewards"]
        ),
        "generation_cap_hit_count": sum(
            bool(value) for row in rows for value in row["generation_cap_hits"]
        ),
        "correct_rollout_histogram": {str(i): histogram[i] for i in range(k + 1)},
    }


def bootstrap_difference(left: list[dict], right: list[dict], seed: int = 42) -> dict:
    a = [statistics.fmean(map(float, row["accuracy_rewards"])) for row in left]
    b = [statistics.fmean(map(float, row["accuracy_rewards"])) for row in right]
    rng = random.Random(seed)
    values = []
    for _ in range(10000):
        am = statistics.fmean(a[rng.randrange(len(a))] for _ in a)
        bm = statistics.fmean(b[rng.randrange(len(b))] for _ in b)
        values.append(bm - am)
    values.sort()
    return {
        "quantity": "rl1000_minus_sft3000_mean_rollout_accuracy",
        "point_difference": statistics.fmean(b) - statistics.fmean(a),
        "bootstrap_95_ci": [values[249], values[9749]],
        "bootstrap_replicates": 10000,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-data", required=True, type=Path)
    parser.add_argument("--rl-data", required=True, type=Path)
    parser.add_argument("--shard-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit-per-split", type=int)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    sft = json.loads(args.sft_data.read_text(encoding="utf-8"))
    rl = json.loads(args.rl_data.read_text(encoding="utf-8"))
    records = selected_records(sft, rl, args.limit_per_split)
    rows, shard_hashes = [], []
    for shard in range(8):
        path = args.shard_root / f"shard_{shard}" / "rollouts.jsonl"
        shard_hashes.append(sha256_file(path))
        rows.extend(json.loads(line) for line in path.open(encoding="utf-8"))
    rows.sort(key=lambda row: row["index"])
    if [row["index"] for row in rows] != list(range(len(records))):
        raise ValueError("merged shards do not cover the selected panel exactly")
    for record, row in zip(records, rows, strict=True):
        expected = digest({key: record[key] for key in ("image", "problem", "solution")})
        if row["source_record_sha256"] != expected or len(row["accuracy_rewards"]) != 8:
            raise ValueError(f"record mismatch at {row['index']}")
    args.output_dir.mkdir(parents=True)
    merged = args.output_dir / "rollouts.jsonl"
    with merged.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    sft_rows = [row for row in rows if row["split"] == "sft3000"]
    rl_rows = [row for row in rows if row["split"] == "rl1000"]
    metrics = {
        "schema_version": 1, "status": "completed", "rollouts_per_prompt": 8,
        "sampling_contract": {
            "do_sample": True, "temperature": 0.9, "top_p": 1.0,
            "top_k": 0, "max_new_tokens": 384,
        },
        "splits": {"sft3000": summarize(sft_rows), "rl1000": summarize(rl_rows)},
        "split_difference": bootstrap_difference(sft_rows, rl_rows),
        "limit_per_split": args.limit_per_split,
        "sft_data_sha256": sha256_file(args.sft_data),
        "rl_data_sha256": sha256_file(args.rl_data),
        "source_shard_sha256s": shard_hashes,
        "merged_rollouts_sha256": sha256_file(merged),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
