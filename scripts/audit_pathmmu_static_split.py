#!/usr/bin/env python3
"""Model-independent audit of the frozen PathMMU SFT3000/RL1000 split."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from pathmmu_rewards import choice_letter
from run_pathmmu_qwen_diagnostic import sha256_file


def describe(rows: list[dict]) -> dict:
    images = [row["image"] for row in rows]
    counts = Counter(images)
    question_lengths = [len(row["problem"]) for row in rows]
    solution_lengths = [len(row["solution"]) for row in rows]
    option_lengths = [len(re.findall(r"(?m)^[A-D]\)", row["problem"])) for row in rows]
    return {
        "count": len(rows), "unique_images": len(counts),
        "mean_questions_per_image": len(rows) / len(counts),
        "max_questions_per_image": max(counts.values()),
        "answer_distribution": dict(sorted(Counter(
            choice_letter(row["solution"]) for row in rows
        ).items())),
        "question_chars": {
            "mean": statistics.fmean(question_lengths),
            "median": statistics.median(question_lengths),
            "min": min(question_lengths), "max": max(question_lengths),
        },
        "solution_chars": {
            "mean": statistics.fmean(solution_lengths),
            "median": statistics.median(solution_lengths),
            "min": min(solution_lengths), "max": max(solution_lengths),
        },
        "four_option_fraction": sum(value == 4 for value in option_lengths) / len(rows),
        "image_extensions": dict(sorted(Counter(
            Path(value).suffix.lower() for value in images
        ).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-data", required=True, type=Path)
    parser.add_argument("--rl-data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sft = json.loads(args.sft_data.read_text(encoding="utf-8"))
    rl = json.loads(args.rl_data.read_text(encoding="utf-8"))
    if len(sft) != 3000 or len(rl) != 1000:
        raise ValueError("frozen split counts are not 3000/1000")
    overlap = set(row["image"] for row in sft) & set(row["image"] for row in rl)
    result = {
        "schema_version": 1, "status": "completed",
        "sft3000": describe(sft), "rl1000": describe(rl),
        "cross_split_image_overlap_count": len(overlap),
        "sft_data_sha256": sha256_file(args.sft_data),
        "rl_data_sha256": sha256_file(args.rl_data),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
