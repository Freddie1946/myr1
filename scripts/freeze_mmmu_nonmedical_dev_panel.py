#!/usr/bin/env python3
"""Freeze the official MMMU non-medical multiple-choice dev panel before inference."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


MEDICAL_SUBJECTS = {
    "Basic_Medical_Science",
    "Clinical_Medicine",
    "Diagnostics_and_Laboratory_Medicine",
    "Pharmacy",
    "Public_Health",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_records(source_root: Path, image_root: Path) -> list[dict]:
    records: list[dict] = []
    for parquet_path in sorted(source_root.glob("*/dev-*.parquet")):
        subject = parquet_path.parent.name
        if subject in MEDICAL_SUBJECTS:
            continue
        for row in pq.read_table(parquet_path).to_pylist():
            if row["question_type"] != "multiple-choice":
                continue
            options = ast.literal_eval(row["options"])
            if not isinstance(options, list) or not 2 <= len(options) <= 9:
                raise ValueError(f"invalid options for {row['id']}")
            answer = str(row["answer"]).strip().upper()
            if answer not in "ABCDEFGHI"[: len(options)]:
                raise ValueError(f"invalid answer for {row['id']}: {answer}")
            images: list[str] = []
            for image_index in range(1, 8):
                image = row.get(f"image_{image_index}")
                if image is None:
                    continue
                value = image["bytes"]
                suffix = Path(image.get("path") or ".png").suffix.lower() or ".png"
                digest = sha256_bytes(value)
                destination = image_root / f"{digest}{suffix}"
                if destination.exists():
                    if sha256_file(destination) != digest:
                        raise ValueError(f"image hash collision at {destination}")
                else:
                    destination.write_bytes(value)
                images.append(str(destination.resolve()))
            if not images:
                raise ValueError(f"no image for {row['id']}")
            records.append(
                {
                    "id": row["id"],
                    "subject": subject,
                    "question": row["question"],
                    "options": options,
                    "answer": answer,
                    "images": images,
                    "image_count": len(images),
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    image_root = args.output_root / "images_by_sha256"
    image_root.mkdir()
    records = select_records(args.source_root, image_root)
    if len(records) != 116:
        raise ValueError(f"expected 116 non-medical multiple-choice dev records, got {len(records)}")
    panel_path = args.output_root / "panel.json"
    panel_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    parquet_hashes = {
        str(path.relative_to(args.source_root)): sha256_file(path)
        for path in sorted(args.source_root.glob("*/dev-*.parquet"))
    }
    manifest = {
        "schema_version": 1,
        "status": "frozen_before_inference",
        "formal_result": False,
        "purpose": "small_nonmedical_general_vlm_retention_sanity_check",
        "source": "MMMU/MMMU",
        "source_revision": args.revision,
        "source_split": "dev",
        "excluded_subjects": sorted(MEDICAL_SUBJECTS),
        "excluded_question_types": ["open"],
        "record_count": len(records),
        "panel": str(panel_path.resolve()),
        "panel_sha256": sha256_file(panel_path),
        "source_parquet_sha256": parquet_hashes,
        "test_accessed": False,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
