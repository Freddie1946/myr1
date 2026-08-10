#!/usr/bin/env python3
"""Prepare a content-addressed PathVQA validation set for prompt/parser calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from external_vqa_contract import pathvqa_answer_type, sha256_file


def build_records(parquet_files: list[Path], image_root: Path) -> list[dict[str, Any]]:
    image_root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for parquet_path in parquet_files:
        table = pq.read_table(parquet_path, columns=["image", "question", "answer"])
        for raw in table.to_pylist():
            image_bytes = raw["image"]["bytes"]
            image_sha = hashlib.sha256(image_bytes).hexdigest()
            suffix = Path(raw["image"]["path"]).suffix.lower() or ".jpg"
            image_path = image_root / f"{image_sha}{suffix}"
            if image_path.exists():
                if sha256_file(image_path) != image_sha:
                    raise ValueError(f"content-address collision: {image_path}")
            else:
                image_path.write_bytes(image_bytes)
            answer = str(raw["answer"])
            rows.append(
                {
                    "source_index": len(rows),
                    "image": str(image_path.resolve()),
                    "image_sha256": image_sha,
                    "question": str(raw["question"]),
                    "answer": answer,
                    "answer_type": pathvqa_answer_type(answer),
                }
            )
    return rows


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    parquet_files = sorted((args.source_root / "data").glob("validation-*.parquet"))
    if len(parquet_files) != 3:
        raise ValueError(f"expected three validation parquet shards, got {len(parquet_files)}")
    args.output_root.mkdir(parents=True)
    records = build_records(parquet_files, args.output_root / "images_by_sha256")
    if len(records) != 6259:
        raise ValueError(f"expected 6259 validation records, got {len(records)}")
    yes_no = [row for row in records if row["answer_type"] == "yes_no"]
    if len(yes_no) != 3125:
        raise ValueError(f"expected 3125 yes/no records, got {len(yes_no)}")
    for index, row in enumerate(yes_no):
        row["index"] = index
    all_path = args.output_root / "pathvqa_validation_6259.json"
    yes_no_path = args.output_root / "pathvqa_validation_yesno_3125.json"
    write_json(all_path, records)
    write_json(yes_no_path, yes_no)
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "split_role": "adapter_prompt_parser_calibration_only",
        "formal_test": False,
        "source_root": str(args.source_root.resolve()),
        "source_parquet_sha256": {
            str(path.resolve()): sha256_file(path) for path in parquet_files
        },
        "record_count": len(records),
        "yes_no_count": len(yes_no),
        "free_form_count": len(records) - len(yes_no),
        "unique_image_contents": len({row["image_sha256"] for row in records}),
        "all_records": str(all_path.resolve()),
        "all_records_sha256": sha256_file(all_path),
        "yes_no_records": str(yes_no_path.resolve()),
        "yes_no_records_sha256": sha256_file(yes_no_path),
        "test_output_rule_selection_forbidden": True,
    }
    write_json(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
