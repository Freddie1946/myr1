#!/usr/bin/env python3
"""Select a balanced, model-blind ROI candidate panel for independent Opus review."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mask_area(regions: list[dict[str, Any]]) -> float:
    mask = np.zeros((1000, 1000), dtype=bool)
    for region in regions:
        x0, y0, x1, y1 = region["box"]
        mask[y0:y1, x0:x1] = True
    return float(mask.mean())


def candidate_metrics(annotation: dict[str, Any]) -> dict[str, Any]:
    regions = annotation["regions"]
    direct = [x for x in regions if x["role"] != "context" and float(x["importance"]) >= 0.5]
    direct_area = mask_area(direct)
    all_area = mask_area(regions)
    eligible = (
        annotation["coverage_complete"]
        and annotation["visual_answerability"] in {"high", "medium"}
        and float(annotation["confidence"]) >= 0.65
        and annotation["evidence_type"] in {"focal", "multifocal", "mixed"}
        and len(direct) >= 1
        and 0.001 <= direct_area <= 0.40
        and all_area <= 0.50
    )
    score = (
        (2.0 if annotation["visual_answerability"] == "high" else 1.0)
        + float(annotation["confidence"])
        + min(len(direct), 6) * 0.08
        + (0.5 if 0.005 <= direct_area <= 0.25 else 0.0)
        + (0.25 if annotation["evidence_type"] in {"multifocal", "mixed"} else 0.0)
    )
    return {"eligible": eligible, "selection_score": score, "direct_region_count": len(direct),
            "all_region_count": len(regions), "direct_union_area_fraction": direct_area,
            "all_union_area_fraction": all_area}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--annotation", type=Path, action="append", required=True)
    p.add_argument("--output-panel", type=Path, required=True)
    p.add_argument("--output-audit", type=Path, required=True)
    p.add_argument("--per-choice", type=int, default=20)
    p.add_argument("--all-eligible", action="store_true")
    args = p.parse_args()
    if args.output_panel.exists() or args.output_audit.exists():
        raise FileExistsError("refusing to overwrite frozen selection outputs")
    panel_raw = args.panel.read_bytes(); panel = json.loads(panel_raw)
    cases = {int(x["panel_index"]): x for x in panel["cases"]}
    annotations = {}
    source_hashes = {}
    for path in args.annotation:
        source_hashes[str(path.resolve())] = sha256_file(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            index = int(row["panel_index"])
            if index in annotations:
                raise ValueError(f"duplicate annotation for panel {index}")
            annotations[index] = row
    if set(annotations) != set(cases):
        missing = sorted(set(cases) - set(annotations)); extra = sorted(set(annotations) - set(cases))
        raise ValueError(f"annotation coverage mismatch missing={missing} extra={extra}")
    if any(x.get("status") != "validated" or x.get("served_model") != "gemini-3.1-pro" for x in annotations.values()):
        raise ValueError("all annotations must be validated exact-model Gemini 3.1 Pro outputs")
    audited = []
    for index, case in cases.items():
        row = annotations[index]
        if row["image_sha256"] != case["image_sha256"]:
            raise ValueError("annotation image mismatch")
        metrics = candidate_metrics(row["annotation"])
        audited.append({"panel_index": index, "target_choice": case["target_choice"],
                        "image_sha256": case["image_sha256"], **metrics,
                        "gemini_annotation": row["annotation"], "gemini_response_id": row.get("response_id")})
    selected = []
    for choice in "ABCD":
        pool = [x for x in audited if x["target_choice"] == choice and x["eligible"]]
        pool.sort(key=lambda x: (-x["selection_score"], hashlib.sha256(
            f"opus-review-v1|{x['panel_index']}|{x['image_sha256']}".encode()).hexdigest()))
        if args.all_eligible:
            selected.extend(pool)
        else:
            if len(pool) < args.per_choice:
                raise ValueError(f"only {len(pool)} eligible Gemini cases for choice {choice}")
            selected.extend(pool[:args.per_choice])
    selected_indices = {x["panel_index"] for x in selected}
    output_cases = [dict(cases[i]) for i in sorted(selected_indices)]
    output = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "status": "frozen",
        "purpose": "blind_independent_claude_opus5_roi_review",
        "selection_uses_target_model_outputs": False, "review_model_can_see_gemini_annotations": False,
        "source_panel": str(args.panel.resolve()), "source_panel_sha256": hashlib.sha256(panel_raw).hexdigest(),
        "gemini_annotation_sources": source_hashes, "candidate_count": len(cases),
        "selected_count": len(output_cases), "per_choice": None if args.all_eligible else args.per_choice,
        "selection_scope": "all_eligible" if args.all_eligible else "balanced_top_per_choice",
        "target_choice_counts": dict(Counter(x["target_choice"] for x in output_cases)), "cases": output_cases,
    }
    args.output_panel.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_panel.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {
        "schema_version": 1, "status": "completed", "selected_count": len(selected),
        "selection_rule": "balanced top eligible Gemini cases; no target-model outputs",
        "thresholds": {"confidence_min": 0.65, "direct_area_min": 0.001,
                       "direct_area_max": 0.40, "all_area_max": 0.50},
        "eligible_count": sum(x["eligible"] for x in audited),
        "eligible_by_choice": dict(Counter(x["target_choice"] for x in audited if x["eligible"])),
        "selected_panel_sha256": sha256_file(args.output_panel), "cases": audited,
    }
    args.output_audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("status", "eligible_count", "eligible_by_choice", "selected_count")}, sort_keys=True))


if __name__ == "__main__":
    main()
