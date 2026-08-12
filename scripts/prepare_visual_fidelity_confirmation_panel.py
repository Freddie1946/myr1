#!/usr/bin/env python3
"""Freeze an outcome-blind visual-fidelity confirmation panel.

The existing 24-case pilot is excluded by source-record hash.  Selection is
deterministic and answer balanced, so no model output or saliency result can
influence the confirmation set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from prepare_visual_fidelity_panel import (
    EXPECTED_VALIDATION_SHA256,
    DEFAULT_SALT,
    canonical_json,
    record_sha256,
    select_panel,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pilot-panel", type=Path, required=True)
    parser.add_argument(
        "--exclude-panel", type=Path, action="append", default=[],
        help="additional panel whose source records must not enter the confirmation set",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-choice", type=int, default=12)
    parser.add_argument("--salt", default=DEFAULT_SALT + ":confirmation-v1")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen panel: {args.output}")
    data_sha = sha256_file(args.data)
    if data_sha != EXPECTED_VALIDATION_SHA256:
        raise ValueError(f"validation SHA-256 mismatch: {data_sha}")
    records = json.loads(args.data.read_text(encoding="utf-8"))
    exclusion_panels = [args.pilot_panel, *args.exclude_panel]
    excluded: set[str] = set()
    exclusion_manifest = []
    for panel_path in exclusion_panels:
        panel = json.loads(panel_path.read_text(encoding="utf-8"))
        source_hashes = {row["source_record_sha256"] for row in panel["cases"]}
        excluded.update(source_hashes)
        exclusion_manifest.append({
            "path": str(panel_path.resolve()),
            "sha256": sha256_file(panel_path),
            "source_record_count": len(source_hashes),
        })
    remaining = [row for row in records if record_sha256(row) not in excluded]
    if len(remaining) != len(records) - len(excluded):
        raise ValueError("pilot exclusion did not match unique source records")
    selected = select_panel(remaining, per_choice=args.per_choice, salt=args.salt)
    cases = []
    for panel_index, row in enumerate(selected):
        image = Path(row["image"]).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        cases.append({**row, "panel_index": panel_index, "image": str(image),
                      "image_sha256": sha256_file(image)})
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_model_visualization_outputs",
        "selection_uses_model_outputs": False,
        "selection_split": "pathmmu_validation_0385_minus_visual_fidelity_panel_v1",
        "data_path": str(args.data.resolve()), "data_sha256": data_sha,
        "excluded_panels": exclusion_manifest,
        "excluded_source_record_sha256": sorted(excluded),
        "selection_algorithm": "global_lowest_sha256(salt + ':' + canonical_record_sha256)_with_target_choice_quotas_and_unique_resolved_image_paths_after_pilot_exclusion",
        "selection_salt": args.salt, "per_choice": args.per_choice,
        "case_count": len(cases),
        "target_choice_counts": {c: sum(x["target_choice"] == c for x in cases) for c in "ABCD"},
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(canonical_json({"output": str(args.output.resolve()), "case_count": len(cases),
                          "target_choice_counts": manifest["target_choice_counts"],
                          "manifest_sha256": sha256_file(args.output)}))


if __name__ == "__main__":
    main()
