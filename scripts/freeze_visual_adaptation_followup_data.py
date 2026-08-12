#!/usr/bin/env python3
"""Freeze an answer-balanced, image-unique PathMMU SFT follow-up panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


ANSWER = re.compile(r"<answer>\s*([A-D])(?:\)|\b)", re.I)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(record: dict, seed: int) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{seed}\0{payload}".encode()).hexdigest()


def answer_letter(record: dict) -> str:
    match = ANSWER.search(record["solution"])
    if not match:
        raise ValueError("record has no A-D answer tag")
    return match.group(1).upper()


def select_records(records: list[dict], count: int, seed: int) -> list[dict]:
    if count % 4:
        raise ValueError("count must be divisible by four")
    quota = count // 4
    ranked = {
        letter: sorted(
            (record for record in records if answer_letter(record) == letter),
            key=lambda record: stable_key(record, seed),
        )
        for letter in "ABCD"
    }
    selected: list[dict] = []
    used_images: set[str] = set()
    for letter in "ABCD":
        arm: list[dict] = []
        for record in ranked[letter]:
            if record["image"] in used_images:
                continue
            arm.append(record)
            used_images.add(record["image"])
            if len(arm) == quota:
                break
        if len(arm) < quota:
            raise ValueError(f"not enough image-unique records for answer {letter}")
        selected.extend(arm)
    selected.sort(key=lambda record: stable_key(record, seed + 1))
    if len({record["image"] for record in selected}) != count:
        raise AssertionError("selection is not image unique")
    return selected


def to_sharegpt(record: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": "<image>" + record["problem"]},
            {"role": "assistant", "content": record["solution"]},
        ],
        "images": [record["image"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=1500)
    parser.add_argument("--selection-seed", type=int, default=20260811)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    records = json.loads(args.source.read_text(encoding="utf-8"))
    selected = select_records(records, args.count, args.selection_seed)
    missing = [record["image"] for record in selected if not Path(record["image"]).is_file()]
    if missing:
        raise FileNotFoundError(missing[0])

    args.output_dir.mkdir(parents=True)
    raw_path = args.output_dir / f"records_n{args.count:04d}.json"
    data_name = f"pathvlm_sft_stratified_n{args.count:04d}"
    data_path = args.output_dir / f"{data_name}.json"
    info_path = args.output_dir / "dataset_info.json"
    raw_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    data_path.write_text(
        json.dumps([to_sharegpt(record) for record in selected], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    info = {
        data_name: {
            "file_name": data_path.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    }
    info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    counts = Counter(answer_letter(record) for record in selected)
    manifest = {
        "schema_version": 1,
        "status": "frozen_before_training",
        "purpose": "L_vs_A_architecture_followup_only",
        "source": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "selection_seed": args.selection_seed,
        "count": len(selected),
        "answer_counts": dict(sorted(counts.items())),
        "unique_images": len({record["image"] for record in selected}),
        "selection": "SHA256-ranked within answer; globally image-unique; A/B/C/D exact balance",
        "records": str(raw_path.resolve()),
        "records_sha256": sha256_file(raw_path),
        "llamafactory_data": str(data_path.resolve()),
        "llamafactory_data_sha256": sha256_file(data_path),
        "dataset_info_sha256": sha256_file(info_path),
        "test_accessed": False,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
