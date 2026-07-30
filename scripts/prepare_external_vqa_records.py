#!/usr/bin/env python3
"""Create content-addressed PathVQA records and a fixed four-source OmniMedVQA index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from external_vqa_contract import omnimed_target_choice, pathvqa_answer_type, sha256_file


OMNI_SOURCES = (
    "Chest CT Scan",
    "Diabetic Retinopathy",
    "ISIC2020",
    "Retinal OCT-C8",
)


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathvqa-root", required=True, type=Path)
    parser.add_argument("--omnimed-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)

    image_root = args.output_root / "pathvqa_images_by_sha256"
    image_root.mkdir()
    pathvqa_records = []
    parquet_files = sorted((args.pathvqa_root / "data").glob("test-*.parquet"))
    if len(parquet_files) != 3:
        raise ValueError(f"expected three PathVQA parquet shards, got {len(parquet_files)}")
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
            pathvqa_records.append(
                {
                    "index": len(pathvqa_records),
                    "image": str(image_path.resolve()),
                    "image_sha256": image_sha,
                    "question": raw["question"],
                    "answer": raw["answer"],
                    "answer_type": pathvqa_answer_type(raw["answer"]),
                }
            )
    if len(pathvqa_records) != 6719:
        raise ValueError(f"expected 6719 PathVQA records, got {len(pathvqa_records)}")
    if len({row["image_sha256"] for row in pathvqa_records}) != 858:
        raise ValueError("expected 858 unique PathVQA image contents")

    omni_records = []
    omni_hashes = {}
    for source in OMNI_SOURCES:
        qa_path = args.omnimed_root / "QA_information" / "Open-access" / f"{source}.json"
        source_rows = json.loads(qa_path.read_text(encoding="utf-8"))
        omni_hashes[source] = sha256_file(qa_path)
        for raw in source_rows:
            image_path = args.omnimed_root / raw["image_path"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            row = dict(raw)
            row["index"] = len(omni_records)
            row["image"] = str(image_path.resolve())
            row["target_choice"] = omnimed_target_choice(row)
            omni_records.append(row)
    if len(omni_records) != 8518:
        raise ValueError(f"expected 8518 OmniMedVQA records, got {len(omni_records)}")

    pathvqa_path = args.output_root / "pathvqa_test_6719.json"
    omni_path = args.output_root / "omnimedvqa_four_sources_8518.json"
    write_json(pathvqa_path, pathvqa_records)
    write_json(omni_path, omni_records)
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "pathvqa": {
            "source_root": str(args.pathvqa_root.resolve()),
            "source_parquet_sha256": {
                str(path.resolve()): sha256_file(path) for path in parquet_files
            },
            "record_count": len(pathvqa_records),
            "unique_image_contents": len(
                {row["image_sha256"] for row in pathvqa_records}
            ),
            "yes_no_count": sum(
                row["answer_type"] == "yes_no" for row in pathvqa_records
            ),
            "free_form_count": sum(
                row["answer_type"] == "free_form" for row in pathvqa_records
            ),
            "records_path": str(pathvqa_path.resolve()),
            "records_sha256": sha256_file(pathvqa_path),
        },
        "omnimedvqa": {
            "source_root": str(args.omnimed_root.resolve()),
            "source_json_sha256": omni_hashes,
            "source_names": list(OMNI_SOURCES),
            "record_count": len(omni_records),
            "records_path": str(omni_path.resolve()),
            "records_sha256": sha256_file(omni_path),
        },
    }
    write_json(args.output_root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
