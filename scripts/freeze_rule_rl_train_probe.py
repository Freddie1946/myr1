#!/usr/bin/env python3
"""Freeze a balanced, image-unique GRPO train probe before mechanism training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pathmmu_rewards import choice_letter


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_key(row: dict) -> str:
    value = f"{row.get('image', '')}\0{row.get('problem', '')}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def freeze(records: list[dict], count: int) -> list[dict]:
    if count < 4 or count % 4:
        raise ValueError("count must be a positive multiple of four")
    quota = count // 4
    groups = {letter: [] for letter in "ABCD"}
    for source_index, record in enumerate(records):
        letter = choice_letter(str(record.get("solution", "")))
        if letter not in groups:
            raise ValueError(f"record {source_index} has no A/B/C/D solution")
        row = dict(record)
        row["source_index"] = source_index
        groups[letter].append(row)
    for letter in groups:
        groups[letter].sort(key=rank_key)

    selected: list[dict] = []
    used_images: set[str] = set()
    for letter in "ABCD":
        for row in groups[letter]:
            image = str(row.get("image", ""))
            if not image or image in used_images:
                continue
            selected.append(row)
            used_images.add(image)
            if sum(choice_letter(str(item["solution"])) == letter for item in selected) == quota:
                break
        actual = sum(choice_letter(str(item["solution"])) == letter for item in selected)
        if actual != quota:
            raise RuntimeError(f"insufficient image-unique rows for {letter}: {actual}/{quota}")
    selected.sort(key=lambda row: (rank_key(row), int(row["source_index"])))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=256)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    records = json.loads(args.source.read_text(encoding="utf-8"))
    selected = freeze(records, args.count)
    args.output_dir.mkdir(parents=True)
    data_path = args.output_dir / f"records_n{args.count:04d}.json"
    data_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = {letter: 0 for letter in "ABCD"}
    for row in selected:
        counts[choice_letter(str(row["solution"]))] += 1
    manifest = {
        "schema_version": 1,
        "status": "frozen_before_mechanism_training",
        "purpose": "fixed_greedy_rule_rl_train_probe",
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "source_count": len(records),
        "records": str(data_path.resolve()),
        "records_sha256": sha256(data_path),
        "count": len(selected),
        "answer_counts": counts,
        "unique_images": len({str(row["image"]) for row in selected}),
        "selection": "SHA256(image\\0problem), exact A/B/C/D balance, globally image-unique",
        "test_accessed": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
