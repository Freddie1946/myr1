#!/usr/bin/env python3
"""Verify frozen PathMMU v2 hashes, exact-content isolation, and nested subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


EXPECTED_VERSION = "pathmmu_image_disjoint_v2"
EXPECTED_COUNTS = {"sft": 3000, "rl": 1000, "validation": 385, "test": 999}
EXPECTED_IMAGES = {"sft": 2121, "rl": 708, "validation": 272, "test": 707}
NESTED = {
    "sft": ((500, 1000), (1000, 2000), (2000, 3000)),
    "rl": ((250, 500), (500, 1000)),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list: {path}")
    return payload


def image_name(record: dict) -> str:
    return Path(str(record["image"])).name


def record_key(record: dict) -> tuple[str, str, str]:
    return image_name(record), str(record["problem"]), str(record["solution"])


def membership_key(record: dict) -> tuple[str, str]:
    return image_name(record), str(record["problem"])


def check_nested(split_root: Path, family: str, small: int, large: int) -> dict:
    directory = split_root / "subsets" / family
    small_rows = load(directory / f"{family}_{small:04d}_without_cot.json")
    large_rows = load(directory / f"{family}_{large:04d}_without_cot.json")
    large_keys = {record_key(row) for row in large_rows}
    nested = all(record_key(row) in large_keys for row in small_rows)
    small_counts = Counter(image_name(row) for row in small_rows)
    large_counts = Counter(image_name(row) for row in large_rows)
    image_complete = all(small_counts[name] == large_counts[name] for name in small_counts)
    return {"family": family, "small": small, "large": large,
            "nested": nested, "image_complete": image_complete}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--parent-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    split_root = args.split_root.resolve()
    image_root = args.image_root.resolve()
    manifest = json.loads((split_root / "manifest.json").read_text(encoding="utf-8"))
    recorded = json.loads((split_root / "validation_report.json").read_text(encoding="utf-8"))
    content_manifest_path = split_root / manifest["content_manifest"]
    content_manifest = json.loads(content_manifest_path.read_text(encoding="utf-8"))
    content_map: dict[str, str] = content_manifest["images"]

    hash_checks = []
    for relative, expected in recorded["files"].items():
        path = split_root / relative
        actual = sha256(path)
        hash_checks.append({"path": relative, "actual_sha256": actual,
                            "recorded_sha256": expected, "matches_recorded": actual == expected})

    primary_rows = {}
    primary_checks = {}
    basename_sets = {}
    content_sets = {}
    representation_checks = {}
    for split, spec in manifest["primary_splits"].items():
        without_rows = load(split_root / spec["without_cot"])
        with_rows = load(split_root / spec["with_cot"])
        without_membership = Counter(membership_key(row) for row in without_rows)
        with_membership = Counter(membership_key(row) for row in with_rows)
        representation_checks[split] = {
            "without_count": len(without_rows),
            "with_count": len(with_rows),
            "same_image_question_membership": without_membership == with_membership,
        }
        images = {image_name(row) for row in without_rows}
        missing_content = sorted(images.difference(content_map))
        primary_rows[split] = without_rows
        basename_sets[split] = images
        content_sets[split] = {content_map[name] for name in images if name in content_map}
        primary_checks[split] = {
            "qa_count": len(without_rows), "expected_qa_count": EXPECTED_COUNTS[split],
            "unique_image_basenames": len(images),
            "expected_unique_image_basenames": EXPECTED_IMAGES[split],
            "missing_content_hashes": missing_content,
            "passed": (len(without_rows) == EXPECTED_COUNTS[split]
                       and len(images) == EXPECTED_IMAGES[split] and not missing_content),
        }

    basename_overlaps = {}
    content_overlaps = {}
    names = list(primary_rows)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            key = f"{left}__{right}"
            basename_overlaps[key] = len(basename_sets[left].intersection(basename_sets[right]))
            content_overlaps[key] = len(content_sets[left].intersection(content_sets[right]))

    referenced_images = set().union(*basename_sets.values())
    physical_checks = []
    for name in sorted(referenced_images):
        path = image_root / name
        actual = sha256(path) if path.is_file() else None
        expected = content_map.get(name)
        physical_checks.append({"image": name, "exists": path.is_file(),
                                "expected_sha256": expected, "actual_sha256": actual,
                                "matches": actual == expected})

    nested_checks = [
        check_nested(split_root, family, small, large)
        for family, pairs in NESTED.items() for small, large in pairs
    ]
    all_records = [record_key(row) for rows in primary_rows.values() for row in rows]

    parent_checks = {"requested": args.parent_root is not None, "passed": True, "files": []}
    if args.parent_root is not None:
        parent = args.parent_root.resolve()
        changed = {"splits/test_0999_without_cot.json", "splits/test_0999_with_cot.json",
                   "manifest.json", "validation_report.json", "image_content_sha256.json"}
        for relative in recorded["files"]:
            if relative in changed:
                continue
            parent_path = parent / relative
            same = parent_path.is_file() and sha256(parent_path) == sha256(split_root / relative)
            parent_checks["files"].append({"path": relative, "unchanged": same})
        parent_checks["passed"] = all(item["unchanged"] for item in parent_checks["files"])

    removed_name = manifest["removal"]["opaque_image_basename"]
    gates = {
        "version_is_v2": manifest.get("version") == EXPECTED_VERSION
                         and recorded.get("version") == EXPECTED_VERSION,
        "all_recorded_hashes_match": all(item["matches_recorded"] for item in hash_checks),
        "primary_counts_match": all(item["passed"] for item in primary_checks.values()),
        "with_without_membership_matches": all(
            item["same_image_question_membership"] for item in representation_checks.values()
        ),
        "primary_qa_union_is_5384": len(all_records) == 5384 and len(set(all_records)) == 5384,
        "all_pairwise_basename_overlaps_zero": all(value == 0 for value in basename_overlaps.values()),
        "all_pairwise_content_overlaps_zero": all(value == 0 for value in content_overlaps.values()),
        "content_manifest_covers_exact_referenced_images": set(content_map) == referenced_images,
        "all_physical_image_hashes_match": all(item["matches"] for item in physical_checks),
        "nested_subsets_valid": all(item["nested"] and item["image_complete"] for item in nested_checks),
        "removed_test_image_absent": removed_name not in referenced_images,
        "parent_unchanged_files_match": parent_checks["passed"],
        "picked_json_absent": not (split_root / "picked.json").exists(),
        "test_decision_blinded": manifest["removal"].get("question_or_answer_inspected_for_decision") is False
                                 and manifest["removal"].get("model_output_or_performance_used_for_decision") is False,
    }
    result = {
        "data_version": EXPECTED_VERSION,
        "split_root": str(split_root),
        "image_root": str(image_root),
        "hash_checks": hash_checks,
        "primary_checks": primary_checks,
        "representation_checks": representation_checks,
        "pairwise_basename_overlaps": basename_overlaps,
        "pairwise_content_overlaps": content_overlaps,
        "content_counts": {"referenced_basenames": len(referenced_images),
                           "unique_exact_contents": len(set(content_map.values()))},
        "physical_image_check_count": len(physical_checks),
        "nested_checks": nested_checks,
        "parent_checks": parent_checks,
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
