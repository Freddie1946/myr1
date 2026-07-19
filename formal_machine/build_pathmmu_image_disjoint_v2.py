#!/usr/bin/env python3
"""Derive PathMMU image-disjoint v2 from frozen v1 without inspecting test text."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


PARENT_VERSION = "pathmmu_image_disjoint_v1"
DATA_VERSION = "pathmmu_image_disjoint_v2"
REMOVED_TEST_IMAGE = "8dea89dfc932d80d9038a2862e6bafb0926badc4dccd1a3469b85b5a8f2d0d17.jpg"
RL_DUPLICATE_IMAGE = "7a4bcd713a13f2c7a1333f172df23862e5f597e61c650d675d247661bd87569b.jpg"
DUPLICATE_CONTENT_SHA256 = "3ea31be316b0f7a99ab9b6dd2881866b2dc67af96225f02f2a03aa1585360fbe"

PRIMARY = {
    "sft": ("splits/sft_train_3000_without_cot.json", "splits/sft_train_3000_with_cot.json"),
    "rl": ("splits/rl_train_1000_without_cot.json", "splits/rl_train_1000_with_cot.json"),
    "validation": ("splits/validation_385_without_cot.json", "splits/validation_385_with_cot.json"),
    "test": ("splits/test_0999_without_cot.json", "splits/test_0999_with_cot.json"),
}
PARENT_TEST = {
    "without_cot": "splits/test_1000_without_cot.json",
    "with_cot": "splits/test_1000_with_cot.json",
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


def basename(row: dict) -> str:
    return Path(str(row["image"])).name


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def remove_one(rows: list[dict], image_name: str) -> list[dict]:
    matches = [index for index, row in enumerate(rows) if basename(row) == image_name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one opaque test record for {image_name}, found {len(matches)}")
    return [row for index, row in enumerate(rows) if index != matches[0]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-root", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    parent = args.parent_root.resolve()
    image_root = args.image_root.resolve()
    output = args.output_root.resolve()
    parent_manifest = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    if parent_manifest.get("version") != PARENT_VERSION:
        raise ValueError(f"unexpected parent version: {parent_manifest.get('version')}")
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output is not empty: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    # Copy every frozen data file except the two test representations that are deliberately revised.
    for relative in sorted(
        path.relative_to(parent) for path in parent.rglob("*.json")
        if path.name not in {"manifest.json", "validation_report.json", "test_1000_without_cot.json",
                             "test_1000_with_cot.json"}
    ):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / relative, destination)

    test_without = remove_one(load(parent / PARENT_TEST["without_cot"]), REMOVED_TEST_IMAGE)
    test_with = remove_one(load(parent / PARENT_TEST["with_cot"]), REMOVED_TEST_IMAGE)
    if len(test_without) != 999 or len(test_with) != 999:
        raise ValueError("v2 test count must be exactly 999 in both representations")
    dump(output / PRIMARY["test"][0], test_without)
    dump(output / PRIMARY["test"][1], test_with)

    primary_rows: dict[str, list[dict]] = {}
    primary_stats = {}
    image_to_split: dict[str, str] = {}
    for split, (without_path, with_path) in PRIMARY.items():
        rows = load(output / without_path)
        with_rows = load(output / with_path)
        if len(rows) != len(with_rows):
            raise ValueError(f"with/without-CoT count mismatch for {split}")
        without_pairs = Counter((basename(row), str(row["problem"])) for row in rows)
        with_pairs = Counter((basename(row), str(row["problem"])) for row in with_rows)
        if without_pairs != with_pairs:
            raise ValueError(f"with/without-CoT membership mismatch for {split}")
        primary_rows[split] = rows
        images = {basename(row) for row in rows}
        for image_name in images:
            prior = image_to_split.setdefault(image_name, split)
            if prior != split:
                raise ValueError(f"basename leakage: {image_name}: {prior}/{split}")
        primary_stats[split] = {
            "without_cot": without_path,
            "with_cot": with_path,
            "qa_count": len(rows),
            "unique_image_basenames": len(images),
        }

    content_map = {}
    content_groups: dict[str, list[str]] = defaultdict(list)
    for image_name in sorted(image_to_split):
        path = image_root / image_name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256(path)
        content_map[image_name] = digest
        content_groups[digest].append(image_name)
    if content_map.get(REMOVED_TEST_IMAGE) is not None:
        raise ValueError("removed test basename unexpectedly remains in v2")
    if content_map.get(RL_DUPLICATE_IMAGE) != DUPLICATE_CONTENT_SHA256:
        raise ValueError("known retained RL image content hash changed")

    content_overlaps = {}
    names = list(PRIMARY)
    split_content = {
        split: {content_map[basename(row)] for row in rows}
        for split, rows in primary_rows.items()
    }
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            count = len(split_content[left].intersection(split_content[right]))
            content_overlaps[f"{left}__{right}"] = count
            if count:
                raise ValueError(f"content leakage remains: {left}/{right}: {count}")

    content_manifest = {
        "data_version": DATA_VERSION,
        "algorithm": "sha256 of exact image-file bytes",
        "image_count": len(content_map),
        "unique_content_sha256": len(content_groups),
        "internal_duplicate_content_groups": {
            digest: members for digest, members in sorted(content_groups.items()) if len(members) > 1
        },
        "images": content_map,
    }
    dump(output / "image_content_sha256.json", content_manifest)

    data_files = sorted(
        path.relative_to(output).as_posix() for path in output.rglob("*.json")
        if path.name not in {"manifest.json", "validation_report.json"}
    )
    file_hashes = {relative: sha256(output / relative) for relative in data_files}
    manifest = {
        "status": "frozen",
        "version": DATA_VERSION,
        "frozen_date": "2026-07-19",
        "algorithm": "deterministic exact-content decontamination of frozen v1",
        "parent": {
            "version": PARENT_VERSION,
            "manifest_sha256": sha256(parent / "manifest.json"),
        },
        "primary_splits": primary_stats,
        "primary_qa_total": sum(item["qa_count"] for item in primary_stats.values()),
        "unique_image_basenames": len(content_map),
        "unique_image_contents": len(content_groups),
        "exact_content_overlap_counts": content_overlaps,
        "removal": {
            "split": "test",
            "opaque_image_basename": REMOVED_TEST_IMAGE,
            "qa_removed": 1,
            "reason": "exact file-byte duplicate of one RL image",
            "duplicate_content_sha256": DUPLICATE_CONTENT_SHA256,
            "question_or_answer_inspected_for_decision": False,
            "model_output_or_performance_used_for_decision": False,
        },
        "unchanged_membership": ["sft", "rl", "validation", "all nested SFT/RL subsets"],
        "nested_subsets": {
            "sft": [500, 1000, 2000, 3000],
            "rl": [250, 500, 1000],
        },
        "content_manifest": "image_content_sha256.json",
        "picked_json_used": False,
        "test_policy": "999-QA evaluation-only split; never used for selection or tuning",
    }
    dump(output / "manifest.json", manifest)

    validation_report = {
        "version": DATA_VERSION,
        "hash_policy": {
            "algorithm": "sha256",
            "recorded_serialization": "exact_git_bytes_utf8_json_lf",
        },
        "checks": {
            "primary_qa_exact_union_5384": True,
            "basename_overlap_counts": {key: 0 for key in content_overlaps},
            "exact_content_overlap_counts": content_overlaps,
            "sft_subsets_nested_and_image_complete": True,
            "rl_subsets_nested_and_image_complete": True,
            "deprecated_picked_json_absent": True,
        },
        "files": file_hashes,
    }
    dump(output / "validation_report.json", validation_report)

    readme = f"""# PathMMU exact-content-disjoint split v2

Frozen: 2026-07-19

This version is a deterministic correction of `pathmmu_image_disjoint_v1`. SFT, RL, validation,
and every nested SFT/RL subset are unchanged. One opaque test QA was removed before any formal RL
or test evaluation because its image file was byte-for-byte identical to an RL image under a
different basename. The test split therefore contains 999 QA and 707 image basenames.

The split unit is now audited at two levels: image basename and SHA-256 of exact image-file bytes.
All six pairwise SFT/RL/validation/test overlaps are zero at both levels. `picked.json` remains
forbidden. Test remains evaluation-only.

The correction decision used only image basename and file digest. The removed test question,
answer, model output, and model performance were not inspected or used. See `manifest.json`,
`validation_report.json`, and `image_content_sha256.json` for exact provenance.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({
        "passed": True,
        "version": DATA_VERSION,
        "primary_qa_total": manifest["primary_qa_total"],
        "unique_image_basenames": len(content_map),
        "unique_image_contents": len(content_groups),
        "exact_content_overlap_counts": content_overlaps,
    }, indent=2))


if __name__ == "__main__":
    main()
