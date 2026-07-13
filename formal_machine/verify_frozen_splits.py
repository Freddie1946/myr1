#!/usr/bin/env python3
"""Verify frozen PathMMU split hashes and image-disjoint data invariants."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PRIMARY = {
    "sft": ("splits/sft_train_3000_without_cot.json", 3000, 2121),
    "rl": ("splits/rl_train_1000_without_cot.json", 1000, 708),
    "validation": ("splits/validation_385_without_cot.json", 385, 272),
    "test": ("splits/test_1000_without_cot.json", 1000, 708),
}
NESTED = {
    "sft": ((500, 1000), (1000, 2000), (2000, 3000)),
    "rl": ((250, 500), (500, 1000)),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_crlf(data: bytes) -> bytes:
    normalized = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return normalized.replace(b"\n", b"\r\n")


def load(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list: {path}")
    return payload


def image_name(record: dict) -> str:
    return Path(str(record["image"])).name


def record_key(record: dict) -> tuple[str, str, str]:
    return image_name(record), str(record["problem"]), str(record["solution"])


def check_nested(split_root: Path, family: str, small: int, large: int) -> dict:
    directory = split_root / "subsets" / family
    small_rows = load(directory / f"{family}_{small:04d}_without_cot.json")
    large_rows = load(directory / f"{family}_{large:04d}_without_cot.json")
    large_keys = {record_key(row) for row in large_rows}
    nested = all(record_key(row) in large_keys for row in small_rows)

    small_images = {image_name(row) for row in small_rows}
    large_by_image: dict[str, int] = {}
    small_by_image: dict[str, int] = {}
    for row in large_rows:
        large_by_image[image_name(row)] = large_by_image.get(image_name(row), 0) + 1
    for row in small_rows:
        small_by_image[image_name(row)] = small_by_image.get(image_name(row), 0) + 1
    image_complete = all(small_by_image[name] == large_by_image[name] for name in small_images)
    return {
        "small": small,
        "large": large,
        "nested": nested,
        "image_complete": image_complete,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    split_root = args.split_root.resolve()
    recorded = json.loads((split_root / "validation_report.json").read_text(encoding="utf-8"))

    hash_checks = []
    for relative, expected in recorded["files"].items():
        normalized_relative = relative.replace("\\", "/")
        if "/" not in normalized_relative:
            normalized_relative = f"splits/{normalized_relative}"
        path = split_root / normalized_relative
        data = path.read_bytes()
        hash_checks.append({
            "path": normalized_relative,
            "checkout_sha256": sha256(data),
            "canonical_crlf_sha256": sha256(canonical_crlf(data)),
            "recorded_crlf_sha256": expected,
            "matches_recorded": sha256(canonical_crlf(data)) == expected,
        })

    primary_rows: dict[str, list[dict]] = {}
    primary_checks = {}
    image_sets = {}
    for name, (relative, qa_count, unique_images) in PRIMARY.items():
        rows = load(split_root / relative)
        images = {image_name(row) for row in rows}
        primary_rows[name] = rows
        image_sets[name] = images
        primary_checks[name] = {
            "qa_count": len(rows),
            "expected_qa_count": qa_count,
            "unique_images": len(images),
            "expected_unique_images": unique_images,
            "passed": len(rows) == qa_count and len(images) == unique_images,
        }

    overlaps = {}
    names = list(image_sets)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlaps[f"{left}__{right}"] = len(image_sets[left].intersection(image_sets[right]))

    nested_checks = [
        check_nested(split_root, family, small, large)
        for family, pairs in NESTED.items()
        for small, large in pairs
    ]
    all_records = [record_key(row) for rows in primary_rows.values() for row in rows]
    gates = {
        "all_recorded_crlf_hashes_match": all(item["matches_recorded"] for item in hash_checks),
        "primary_counts_match": all(item["passed"] for item in primary_checks.values()),
        "primary_qa_union_is_5385": len(all_records) == 5385 and len(set(all_records)) == 5385,
        "all_pairwise_image_overlaps_zero": all(value == 0 for value in overlaps.values()),
        "nested_subsets_valid": all(item["nested"] and item["image_complete"] for item in nested_checks),
        "picked_json_absent": not (split_root / "picked.json").exists(),
    }
    result = {
        "data_version": "pathmmu_image_disjoint_v1",
        "split_root": str(split_root),
        "hash_policy": recorded.get("hash_policy"),
        "hash_checks": hash_checks,
        "primary_checks": primary_checks,
        "pairwise_image_overlaps": overlaps,
        "nested_checks": nested_checks,
        "gates": gates,
        "passed": all(gates.values()),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
