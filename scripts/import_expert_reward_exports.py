#!/usr/bin/env python3
"""Verify and import distributed expert reward-review ZIP exports.

The archive is treated as untrusted input: paths, manifest hashes, reviewer ID,
case IDs, and the exact six-boolean event schema are checked before any record
is copied into the analysis directory. Existing non-identical files are never
overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


EVENTS = (
    "image_feature_analysis_present",
    "option_elimination_present",
    "medical_knowledge_support_present",
    "histological_definition_error",
    "logical_contradiction",
    "outdated_or_incorrect_pathology_criterion",
)
REVIEWER_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}")
CASE_RE = re.compile(r"HRA-(?:00[1-9]|0[1-5][0-9]|060)")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe ZIP member: {name}")


def validate_rating(data: bytes, reviewer_id: str, expected_case: str) -> None:
    row = json.loads(data)
    if row.get("reviewer_id") != reviewer_id or row.get("case_id") != expected_case:
        raise ValueError(f"rating identity mismatch: {expected_case}")
    events = row.get("events")
    if not isinstance(events, dict) or set(events) != set(EVENTS):
        raise ValueError(f"invalid event fields: {expected_case}")
    if any(type(events[event]) is not bool for event in EVENTS):
        raise ValueError(f"non-boolean event: {expected_case}")


def import_archive(archive_path: Path, output_root: Path, allow_partial: bool) -> dict[str, Any]:
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        for name in names:
            validate_member_name(name)
        manifest_names = [name for name in names if PurePosixPath(name).name == "EXPORT_MANIFEST.json"]
        if len(manifest_names) != 1:
            raise ValueError(f"expected one EXPORT_MANIFEST.json in {archive_path}")
        manifest_name = manifest_names[0]
        prefix = PurePosixPath(manifest_name).parent
        manifest = json.loads(archive.read(manifest_name))
        reviewer_id = manifest.get("reviewer_id", "")
        if not REVIEWER_RE.fullmatch(reviewer_id):
            raise ValueError(f"invalid reviewer_id in {archive_path}")
        completed = manifest.get("completed_cases")
        expected = manifest.get("expected_cases")
        complete = manifest.get("complete")
        if expected != 60 or not isinstance(completed, int) or not 1 <= completed <= 60:
            raise ValueError(f"invalid completion counts in {archive_path}")
        if not allow_partial and (complete is not True or completed != 60):
            raise ValueError(f"partial review export rejected: {archive_path}")
        records = manifest.get("files")
        if not isinstance(records, list):
            raise ValueError(f"invalid file manifest in {archive_path}")
        verified: dict[str, bytes] = {}
        for record in records:
            basename = record.get("path", "")
            if PurePosixPath(basename).name != basename:
                raise ValueError(f"non-flat manifest path: {basename}")
            member = str(prefix / basename)
            if member not in names:
                raise ValueError(f"missing ZIP member: {member}")
            data = archive.read(member)
            if len(data) != record.get("bytes") or sha256(data) != record.get("sha256"):
                raise ValueError(f"hash/size mismatch: {member}")
            verified[basename] = data
        rating_names = sorted(name for name in verified if CASE_RE.fullmatch(Path(name).stem) and name.endswith(".json"))
        if len(rating_names) != completed:
            raise ValueError(f"rating count mismatch in {archive_path}")
        if "reward_six_event_ratings.csv" not in verified:
            raise ValueError(f"missing rating CSV in {archive_path}")
        for name in rating_names:
            validate_rating(verified[name], reviewer_id, Path(name).stem)

    destination = output_root / reviewer_id
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in verified.items():
        target = destination / name
        if target.exists():
            if target.read_bytes() != data:
                raise FileExistsError(f"refusing to overwrite divergent rating: {target}")
            continue
        with tempfile.NamedTemporaryFile(dir=destination, prefix=f".{name}.", delete=False) as handle:
            handle.write(data)
            temporary = Path(handle.name)
        os.replace(temporary, target)
    import_record = {
        "schema_version": 1,
        "source_archive": str(archive_path.resolve()),
        "source_archive_sha256": sha256(archive_path.read_bytes()),
        "reviewer_id": reviewer_id,
        "completed_cases": completed,
        "complete": complete,
        "imported_files": sorted(verified),
    }
    record_path = destination / "IMPORT_MANIFEST.json"
    record_path.write_text(json.dumps(import_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return import_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    imported = [import_archive(path, args.output_dir, args.allow_partial) for path in args.archives]
    print(json.dumps({"status": "verified_and_imported", "reviewers": imported}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
