#!/usr/bin/env python3
"""Freeze a model-blind candidate panel for predefined pathology evidence regions.

The panel is deliberately frozen before any target-checkpoint attention, RISE,
or prediction output is inspected.  External visual models may annotate the
resulting cases later; those annotations are metadata, not ground truth until
human pathology review is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prepare_visual_fidelity_panel import (
    EXPECTED_VALIDATION_SHA256,
    canonical_json,
    record_sha256,
    sha256_file,
    target_choice,
)


def select_candidates(
    records: list[dict[str, Any]],
    *,
    excluded: set[str],
    per_choice: int,
    salt: str,
) -> list[dict[str, Any]]:
    if per_choice <= 0:
        raise ValueError("per_choice must be positive")
    buckets: dict[str, list[dict[str, Any]]] = {letter: [] for letter in "ABCD"}
    for index, record in enumerate(records):
        source_hash = record_sha256(record)
        if source_hash in excluded:
            continue
        image = Path(record["image"]).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        letter = target_choice(record["solution"])
        rank = hashlib.sha256(f"{salt}:{source_hash}".encode()).hexdigest()
        buckets[letter].append({
            "index": index,
            "source_record_sha256": source_hash,
            "selection_rank_sha256": rank,
            "target_choice": letter,
            **record,
        })
    selected: list[dict[str, Any]] = []
    used_images: set[str] = set()
    for letter in "ABCD":
        ranked = sorted(buckets[letter], key=lambda row: (row["selection_rank_sha256"], row["index"]))
        for row in ranked:
            image_identity = str(Path(row["image"]).resolve())
            if image_identity in used_images:
                continue
            selected.append(row)
            used_images.add(image_identity)
            if sum(item["target_choice"] == letter for item in selected) >= per_choice:
                break
    counts = {letter: sum(row["target_choice"] == letter for row in selected) for letter in "ABCD"}
    if any(value != per_choice for value in counts.values()):
        raise ValueError(f"could not fill balanced panel: {counts}")
    selected.sort(key=lambda row: row["index"])
    output = []
    for panel_index, row in enumerate(selected):
        image = Path(row["image"]).resolve()
        output.append({
            **row,
            "panel_index": panel_index,
            "image": str(image),
            "image_sha256": sha256_file(image),
            "reference_evidence_status": "pending_external_annotation",
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--exclude-panel", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-choice", type=int, default=20)
    parser.add_argument("--salt", default="pathvlm-r1-predefined-evidence-candidates-v1-20260812")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen panel: {args.output}")
    data_sha = sha256_file(args.data)
    if data_sha != EXPECTED_VALIDATION_SHA256:
        raise ValueError(f"validation SHA-256 mismatch: {data_sha}")
    records = json.loads(args.data.read_text(encoding="utf-8"))
    excluded: set[str] = set()
    excluded_panels = []
    for path in args.exclude_panel:
        panel = json.loads(path.read_text(encoding="utf-8"))
        hashes = {row["source_record_sha256"] for row in panel["cases"]}
        excluded.update(hashes)
        excluded_panels.append({"path": str(path.resolve()), "sha256": sha256_file(path), "count": len(hashes)})
    cases = select_candidates(records, excluded=excluded, per_choice=args.per_choice, salt=args.salt)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_target_model_outputs",
        "selection_uses_model_outputs": False,
        "purpose": "predefined_diagnostically_relevant_evidence_region_candidate_panel",
        "annotation_boundary": "external_model annotations are references, not pathology ground truth before expert review",
        "data_path": str(args.data.resolve()),
        "data_sha256": data_sha,
        "excluded_panels": excluded_panels,
        "excluded_source_record_sha256": sorted(excluded),
        "selection_algorithm": "per-choice lowest salted record hash with unique resolved image paths",
        "selection_salt": args.salt,
        "per_choice": args.per_choice,
        "case_count": len(cases),
        "target_choice_counts": {letter: sum(row["target_choice"] == letter for row in cases) for letter in "ABCD"},
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp-{__import__('os').getpid()}")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(canonical_json({"output": str(args.output.resolve()), "case_count": len(cases), "target_choice_counts": manifest["target_choice_counts"], "sha256": sha256_file(args.output)}))


if __name__ == "__main__":
    main()
