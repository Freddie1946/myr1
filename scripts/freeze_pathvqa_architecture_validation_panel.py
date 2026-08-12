#!/usr/bin/env python3
"""Freeze an output-blind, balanced, image-unique PathVQA validation panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from external_vqa_contract import normalize_short_answer, sha256_file


SALT = "pathvqa_architecture_validation_v1_20260811"


def rank_key(row: dict) -> str:
    value = "\0".join(
        (
            SALT,
            str(row.get("image_sha256") or row["image"]),
            str(row["question"]),
            str(row["answer"]),
            str(row.get("source_index", row.get("index"))),
        )
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def select_panel(rows: list[dict], excluded_source_indices: set[int], per_class: int) -> list[dict]:
    candidates = []
    for row in rows:
        answer = normalize_short_answer(str(row.get("answer") or ""))
        source_index = int(row.get("source_index", row.get("index")))
        if answer not in {"yes", "no"} or source_index in excluded_source_indices:
            continue
        candidates.append((rank_key(row), answer, source_index, row))
    selected = []
    counts = Counter()
    used_images = set()
    for _, answer, source_index, row in sorted(candidates):
        image_id = str(row.get("image_sha256") or row["image"])
        if counts[answer] >= per_class or image_id in used_images:
            continue
        selected.append({**row, "architecture_panel_index": len(selected)})
        counts[answer] += 1
        used_images.add(image_id)
        if counts == Counter({"yes": per_class, "no": per_class}):
            break
    if counts != Counter({"yes": per_class, "no": per_class}):
        raise ValueError(f"insufficient balanced image-unique records: {dict(counts)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclude-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=256)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite frozen panel artifacts")
    if args.per_class < 1:
        raise ValueError("per-class count must be positive")
    rows = json.loads(args.source.read_text(encoding="utf-8"))
    excluded_rows = [
        json.loads(line)
        for line in args.exclude_predictions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    excluded = {int(row["source_index"]) for row in excluded_rows}
    panel = select_panel(rows, excluded, args.per_class)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(panel, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "frozen_before_model_evaluation",
        "formal_result": False,
        "split_role": "architecture_selection_validation_only",
        "selection_method": "SHA256 rank with fixed salt; balanced Yes/No; one record per image; no model outputs or labels beyond balancing",
        "salt": SALT,
        "source": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "excluded_prior_panel_predictions": str(args.exclude_predictions.resolve()),
        "excluded_prior_panel_predictions_sha256": sha256_file(args.exclude_predictions),
        "excluded_source_index_count": len(excluded),
        "record_count": len(panel),
        "yes_count": sum(normalize_short_answer(str(row["answer"])) == "yes" for row in panel),
        "no_count": sum(normalize_short_answer(str(row["answer"])) == "no" for row in panel),
        "unique_image_count": len({str(row.get("image_sha256") or row["image"]) for row in panel}),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "test_accessed": False,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
