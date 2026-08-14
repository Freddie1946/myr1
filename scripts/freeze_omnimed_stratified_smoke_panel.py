#!/usr/bin/env python3
"""Freeze a deterministic, source-stratified OmniMedVQA smoke panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_SOURCES = (
    "Chest CT Scan",
    "Diabetic Retinopathy",
    "ISIC2020",
    "Retinal OCT-C8",
)


def canonical_record_sha256(record: dict[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def freeze(records: list[dict[str, Any]], per_source: int) -> list[dict[str, Any]]:
    if per_source <= 0:
        raise ValueError("per_source must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("dataset"))].append(record)
    if set(grouped) != set(EXPECTED_SOURCES):
        raise ValueError(
            f"unexpected OmniMedVQA sources: expected {EXPECTED_SOURCES}, got {sorted(grouped)}"
        )
    selected: list[dict[str, Any]] = []
    for source in EXPECTED_SOURCES:
        ordered = sorted(
            grouped[source],
            key=lambda row: (canonical_record_sha256(row), str(row.get("question_id"))),
        )
        if len(ordered) < per_source:
            raise ValueError(f"source {source} has only {len(ordered)} records")
        selected.extend(ordered[:per_source])
    counts = Counter(str(row["dataset"]) for row in selected)
    if counts != Counter({source: per_source for source in EXPECTED_SOURCES}):
        raise AssertionError(f"stratification failure: {counts}")
    if len({str(row["question_id"]) for row in selected}) != len(selected):
        raise ValueError("selected question IDs are not unique")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=16)
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 8518:
        raise ValueError(f"expected the frozen 8518-record pool, got {len(records)}")
    selected = freeze(records, args.per_source)
    atomic_json(args.output, selected)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "formal_result": False,
        "purpose": "cross-source generation-contract and adapter smoke only",
        "selection": "per-source canonical-record-sha256 ascending",
        "source_data": str(args.input.resolve()),
        "source_data_sha256": sha256_file(args.input),
        "source_data_count": len(records),
        "panel": str(args.output.resolve()),
        "panel_sha256": sha256_file(args.output),
        "panel_count": len(selected),
        "per_source": args.per_source,
        "source_counts": dict(sorted(Counter(row["dataset"] for row in selected).items())),
        "question_ids": [row["question_id"] for row in selected],
        "record_sha256": [canonical_record_sha256(row) for row in selected],
        "accuracy_used_as_gate": False,
    }
    atomic_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
