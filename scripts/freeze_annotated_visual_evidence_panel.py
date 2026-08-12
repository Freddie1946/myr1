#!/usr/bin/env python3
"""Freeze a transparent, model-output-independent-after-annotation evidence panel.

All 80 external annotations remain in the audit file. The selected subset is
chosen only by the predeclared annotation rule: nonempty focal/multifocal boxes,
confidence >= 0.90, redundancy <= 0.50. No target-checkpoint predictions are
used for selection; clean-correct filtering happens later and is reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    rows = {int(json.loads(line)["panel_index"]): json.loads(line) for line in args.annotations.read_text(encoding="utf-8").splitlines() if line.strip()}
    if len(rows) != len(panel["cases"]):
        raise SystemExit(f"need all annotations before freezing: {len(rows)}/{len(panel['cases'])}")
    selected = []
    for case in panel["cases"]:
        row = rows[int(case["panel_index"])]
        ann = row["annotation"]
        if ann["evidence_type"] not in {"focal", "multifocal"}:
            continue
        if not ann["boxes"] or float(ann["confidence"]) < 0.90 or float(ann["redundancy"]) > 0.50:
            continue
        selected.append({**case, "external_annotation": ann, "annotation_model": row["model"], "annotation_attempt": row["attempt"]})
    result = {
        "schema_version": 1,
        "status": "frozen_before_target_model_outputs",
        "selection_uses_target_outputs": False,
        "source_panel": str(args.panel.resolve()),
        "source_panel_sha256": sha256(args.panel),
        "annotations": str(args.annotations.resolve()),
        "annotations_sha256": sha256(args.annotations),
        "selection_rule": "evidence_type in {focal,multifocal} AND nonempty boxes AND confidence >= 0.90 AND redundancy <= 0.50",
        "selected_count": len(selected),
        "candidate_count": len(panel["cases"]),
        "cases": selected,
        "clean_correct_filter": "applied only after this file is frozen; all candidates and exclusions remain auditable",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_count": len(selected), "candidate_count": len(panel["cases"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
