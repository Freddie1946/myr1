#!/usr/bin/env python3
"""Validate prepared external evaluation assets without running model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


OMNIMED_SOURCES = (
    "Chest CT Scan",
    "ISIC2020",
    "Retinal OCT-C8",
    "Diabetic Retinopathy",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def load_formal_hashes(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = payload.get("images")
    if not isinstance(images, dict):
        raise ValueError(f"formal content manifest has no image mapping: {path}")
    values = set(images.values())
    if not all(isinstance(value, str) and len(value) == 64 for value in values):
        raise ValueError(f"invalid SHA-256 value in formal content manifest: {path}")
    return values


def audit_omnimed(root: Path, formal_hashes: set[str]) -> tuple[dict[str, Any], set[str]]:
    source_reports: dict[str, Any] = {}
    all_hashes: set[str] = set()

    for source in OMNIMED_SOURCES:
        metadata_path = root / "QA_information" / "Open-access" / f"{source}.json"
        records = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"OmniMedVQA metadata is not a list: {metadata_path}")

        question_ids: set[str] = set()
        referenced_paths: set[Path] = set()
        missing_paths: list[str] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"non-object record {index} in {metadata_path}")
            for field in ("question_id", "image_path", "question", "gt_answer"):
                if field not in record:
                    raise ValueError(f"missing {field!r} in record {index} of {metadata_path}")
            question_ids.add(str(record["question_id"]))
            image_path = contained_path(root, str(record["image_path"]))
            referenced_paths.add(image_path)
            if not image_path.is_file():
                missing_paths.append(str(image_path))

        content_paths: dict[str, list[str]] = {}
        for path in sorted(referenced_paths):
            if not path.is_file():
                continue
            digest = sha256_file(path)
            content_paths.setdefault(digest, []).append(str(path.relative_to(root)))
        hashes = set(content_paths)
        duplicate_groups = {
            digest: paths for digest, paths in content_paths.items() if len(paths) > 1
        }
        all_hashes.update(hashes)
        source_reports[source] = {
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "qa_count": len(records),
            "unique_question_id_count": len(question_ids),
            "unique_referenced_image_count": len(referenced_paths),
            "existing_referenced_image_count": len(referenced_paths) - len(missing_paths),
            "unique_image_content_count": len(hashes),
            "internal_exact_duplicate_path_count": len(referenced_paths) - len(hashes),
            "internal_exact_duplicate_content_group_count": len(duplicate_groups),
            "internal_exact_duplicate_content_groups": duplicate_groups,
            "missing_image_count": len(missing_paths),
            "missing_images": missing_paths,
            "formal_pathmmu_exact_content_overlap_count": len(hashes & formal_hashes),
        }

    return source_reports, all_hashes


def image_payload_bytes(value: Any, parquet_path: Path) -> bytes:
    if not isinstance(value, dict):
        raise ValueError(f"unexpected image value in {parquet_path}: {type(value).__name__}")
    payload = value.get("bytes")
    if isinstance(payload, bytes):
        return payload
    relative_path = value.get("path")
    if isinstance(relative_path, str):
        candidate = contained_path(parquet_path.parent, relative_path)
        return candidate.read_bytes()
    raise ValueError(f"image has neither bytes nor path in {parquet_path}")


def audit_pathvqa(
    root: Path, formal_hashes: set[str], omnimed_hashes: set[str]
) -> tuple[dict[str, Any], set[str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("PathVQA audit requires pyarrow in the selected environment") from exc

    parquet_paths = sorted((root / "data").glob("test-*.parquet"))
    if len(parquet_paths) != 3:
        raise ValueError(f"expected exactly 3 PathVQA test shards, found {len(parquet_paths)}")

    total_rows = 0
    image_hashes: set[str] = set()
    shards: list[dict[str, Any]] = []
    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        names = parquet_file.schema_arrow.names
        if names != ["image", "question", "answer"]:
            raise ValueError(f"unexpected PathVQA schema in {parquet_path}: {names}")
        rows = parquet_file.metadata.num_rows
        total_rows += rows
        for batch in parquet_file.iter_batches(columns=["image"], batch_size=128):
            for value in batch.column(0).to_pylist():
                image_hashes.add(sha256_bytes(image_payload_bytes(value, parquet_path)))
        shards.append(
            {
                "path": str(parquet_path),
                "sha256": sha256_file(parquet_path),
                "rows": rows,
                "columns": names,
            }
        )

    return (
        {
            "root": str(root),
            "shard_count": len(parquet_paths),
            "qa_count": total_rows,
            "unique_image_content_count": len(image_hashes),
            "formal_pathmmu_exact_content_overlap_count": len(image_hashes & formal_hashes),
            "omnimedvqa_exact_content_overlap_count": len(image_hashes & omnimed_hashes),
            "shards": shards,
        },
        image_hashes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--omnimed-root", type=Path, required=True)
    parser.add_argument("--pathvqa-root", type=Path, required=True)
    parser.add_argument("--formal-content-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formal_hashes = load_formal_hashes(args.formal_content_manifest)
    omnimed_report, omnimed_hashes = audit_omnimed(args.omnimed_root, formal_hashes)
    pathvqa_report, _ = audit_pathvqa(args.pathvqa_root, formal_hashes, omnimed_hashes)

    passed = (
        all(report["missing_image_count"] == 0 for report in omnimed_report.values())
        and all(
            report["formal_pathmmu_exact_content_overlap_count"] == 0
            for report in omnimed_report.values()
        )
        and pathvqa_report["formal_pathmmu_exact_content_overlap_count"] == 0
        and pathvqa_report["omnimedvqa_exact_content_overlap_count"] == 0
    )
    report = {
        "schema_version": 1,
        "formal_result": False,
        "scope": "preparation-only external asset integrity and exact-content overlap audit",
        "formal_content_manifest": {
            "path": str(args.formal_content_manifest),
            "sha256": sha256_file(args.formal_content_manifest),
            "unique_content_count": len(formal_hashes),
        },
        "omnimedvqa": {
            "root": str(args.omnimed_root),
            "sources": omnimed_report,
            "all_source_unique_content_count": len(omnimed_hashes),
        },
        "pathvqa_test": pathvqa_report,
        "gates": {
            "all_referenced_omnimedvqa_images_exist": all(
                item["missing_image_count"] == 0 for item in omnimed_report.values()
            ),
            "no_external_formal_pathmmu_exact_content_overlap": all(
                item["formal_pathmmu_exact_content_overlap_count"] == 0
                for item in omnimed_report.values()
            )
            and pathvqa_report["formal_pathmmu_exact_content_overlap_count"] == 0,
            "no_pathvqa_omnimedvqa_exact_content_overlap": (
                pathvqa_report["omnimedvqa_exact_content_overlap_count"] == 0
            ),
            "model_inference_performed": False,
            "formal_split_labels_read": False,
            "passed": passed,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["gates"], sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
