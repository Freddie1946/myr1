#!/usr/bin/env python3
"""Freeze a model-blind, answer-balanced PathVQA validation calibration panel."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from external_vqa_contract import normalize_short_answer, sha256_file


def select_panel(
    records: list[dict[str, Any]], *, per_answer: int, seed: int
) -> list[dict[str, Any]]:
    if per_answer <= 0:
        raise ValueError("per-answer count must be positive")
    randomizer = random.Random(seed)
    selected: list[dict[str, Any]] = []
    used_images: set[str] = set()
    for answer in ("yes", "no"):
        candidates = [
            row for row in records if normalize_short_answer(row["answer"]) == answer
        ]
        randomizer.shuffle(candidates)
        chosen = []
        for row in candidates:
            image_sha = str(row["image_sha256"])
            if image_sha in used_images:
                continue
            chosen.append(dict(row))
            used_images.add(image_sha)
            if len(chosen) == per_answer:
                break
        if len(chosen) != per_answer:
            raise ValueError(f"not enough unique-image {answer} records")
        selected.extend(chosen)
    randomizer.shuffle(selected)
    for panel_index, row in enumerate(selected):
        row["panel_index"] = panel_index
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-answer", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = json.loads(args.records.read_text(encoding="utf-8"))
    panel = select_panel(records, per_answer=args.per_answer, seed=args.seed)
    payload = {
        "schema_version": 1,
        "status": "frozen_before_calibration_inference",
        "split_role": "pathvqa_validation_prompt_parser_calibration",
        "formal_test": False,
        "source_records": str(args.records.resolve()),
        "source_records_sha256": sha256_file(args.records),
        "seed": args.seed,
        "selection_uses_model_outputs": False,
        "per_answer": args.per_answer,
        "count": len(panel),
        "unique_image_contents": len({row["image_sha256"] for row in panel}),
        "records": panel,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "passed", "count": len(panel), "output": str(args.output)}))


if __name__ == "__main__":
    main()
