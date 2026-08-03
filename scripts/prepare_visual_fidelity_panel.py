#!/usr/bin/env python3
"""Freeze an outcome-blind, answer-balanced PathMMU visualization panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_VALIDATION_SHA256 = (
    "f52c9a412da579db8fb8bf52f6fa320e7c4ae9cb42b6cf1bdf62f77ab158dcf0"
)
DEFAULT_SALT = "pathvlm-r1-visual-fidelity-panel-v1"
ANSWER_PATTERN = re.compile(r"<answer\b[^>]*>\s*([A-D])\b", re.I)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def record_sha256(record: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()


def target_choice(solution: str) -> str:
    matches = ANSWER_PATTERN.findall(str(solution))
    if not matches:
        raise ValueError("record solution has no A-D answer tag")
    return matches[-1].upper()


def select_panel(
    records: list[dict[str, Any]], *, per_choice: int, salt: str
) -> list[dict[str, Any]]:
    if per_choice <= 0:
        raise ValueError("per_choice must be positive")
    buckets: dict[str, list[dict[str, Any]]] = {choice: [] for choice in "ABCD"}
    for index, record in enumerate(records):
        if set(record) != {"image", "problem", "solution"}:
            raise ValueError(f"unexpected validation schema at index {index}")
        choice = target_choice(record["solution"])
        source_sha = record_sha256(record)
        rank_sha = hashlib.sha256(f"{salt}:{source_sha}".encode("utf-8")).hexdigest()
        buckets[choice].append(
            {
                "index": index,
                "target_choice": choice,
                "selection_rank_sha256": rank_sha,
                "source_record_sha256": source_sha,
                **record,
            }
        )
    selected = []
    counts = {choice: 0 for choice in "ABCD"}
    used_images: set[str] = set()
    ranked = sorted(
        (row for rows in buckets.values() for row in rows),
        key=lambda row: (row["selection_rank_sha256"], row["index"]),
    )
    for row in ranked:
        choice = row["target_choice"]
        image_identity = str(Path(row["image"]).resolve())
        if counts[choice] >= per_choice or image_identity in used_images:
            continue
        selected.append(row)
        counts[choice] += 1
        used_images.add(image_identity)
        if all(count == per_choice for count in counts.values()):
            break
    unfilled = {choice: per_choice - count for choice, count in counts.items() if count < per_choice}
    if unfilled:
        raise ValueError(f"cannot fill answer-balanced unique-image panel: {unfilled}")
    return sorted(selected, key=lambda row: row["index"])


def build_manifest(
    data_path: Path, *, per_choice: int, salt: str
) -> dict[str, Any]:
    data_path = data_path.resolve()
    actual_data_sha = sha256_file(data_path)
    if actual_data_sha != EXPECTED_VALIDATION_SHA256:
        raise ValueError(
            f"validation SHA-256 mismatch: {actual_data_sha} != {EXPECTED_VALIDATION_SHA256}"
        )
    records = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 385:
        raise ValueError("frozen validation input must contain exactly 385 records")
    selected = select_panel(records, per_choice=per_choice, salt=salt)
    cases = []
    for panel_index, row in enumerate(selected):
        image = Path(row["image"]).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        cases.append(
            {
                **row,
                "panel_index": panel_index,
                "image": str(image),
                "image_sha256": sha256_file(image),
            }
        )
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_model_visualization_outputs",
        "selection_uses_model_outputs": False,
        "selection_split": "pathmmu_validation_0385",
        "data_path": str(data_path),
        "data_sha256": actual_data_sha,
        "selection_algorithm": "global_lowest_sha256(salt + ':' + canonical_record_sha256)_with_target_choice_quotas_and_unique_resolved_image_paths",
        "unique_image_paths_required": True,
        "selection_salt": salt,
        "per_choice": per_choice,
        "case_count": len(cases),
        "target_choice_counts": {
            choice: sum(row["target_choice"] == choice for row in cases)
            for choice in "ABCD"
        },
        "primary_display_panel_indices": list(range(0, len(cases), 3)),
        "primary_display_selection_uses_model_outputs": False,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-choice", type=int, default=6)
    parser.add_argument("--salt", default=DEFAULT_SALT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen panel: {args.output}")
    manifest = build_manifest(args.data, per_choice=args.per_choice, salt=args.salt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(canonical_json({
        "output": str(args.output.resolve()),
        "case_count": manifest["case_count"],
        "target_choice_counts": manifest["target_choice_counts"],
        "manifest_sha256": sha256_file(args.output),
    }))


if __name__ == "__main__":
    main()
