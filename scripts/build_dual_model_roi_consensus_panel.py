#!/usr/bin/env python3
"""Build a target-model-blind Gemini/Claude spatial-consensus ROI panel."""
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


def load_rows(paths: list[Path], expected_model: str) -> tuple[dict[int, dict[str, Any]], dict[str, str]]:
    rows = {}; hashes = {}
    for path in paths:
        hashes[str(path.resolve())] = sha256_file(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line); index = int(row["panel_index"])
            if index in rows:
                raise ValueError(f"duplicate panel index {index}")
            if row.get("status") != "validated" or row.get("served_model") != expected_model:
                raise ValueError(f"invalid {expected_model} row at panel {index}")
            rows[index] = row
    return rows, hashes


def direct_regions(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    return [x for x in annotation["regions"] if x["role"] != "context" and float(x["importance"]) >= 0.5]


def box_overlap_coefficient(a: list[int], b: list[int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy; minimum = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / minimum if minimum else 0.0


def union_area(regions: list[dict[str, Any]]) -> float:
    mask = np.zeros((1000, 1000), bool)
    for region in regions:
        x0, y0, x1, y1 = region["box"]; mask[y0:y1, x0:x1] = True
    return float(mask.mean())


def consensus(gemini: dict[str, Any], opus: dict[str, Any], threshold: float = 0.20) -> dict[str, Any]:
    ga = direct_regions(gemini); oa = direct_regions(opus)
    overlaps = [[box_overlap_coefficient(g["box"], o["box"]) for o in oa] for g in ga]
    matched_g = [i for i, row in enumerate(overlaps) if row and max(row) >= threshold]
    matched_o = [j for j in range(len(oa)) if any(row[j] >= threshold for row in overlaps)] if oa else []
    primary = [ga[i] for i in matched_g]
    gem_coverage = len(matched_g) / len(ga) if ga else 0.0
    opus_coverage = len(matched_o) / len(oa) if oa else 0.0
    primary_area = union_area(primary)
    eligible = (
        gemini["coverage_complete"] and opus["coverage_complete"]
        and gemini["visual_answerability"] in {"high", "medium"}
        and opus["visual_answerability"] in {"high", "medium"}
        and float(gemini["confidence"]) >= 0.65 and float(opus["confidence"]) >= 0.65
        # Exhaustive annotators often decompose the same tissue architecture at
        # different granularities.  The intervention mask contains only Gemini
        # boxes independently corroborated by at least one Opus box; unmatched
        # regions remain in the audit record and are never added to that mask.
        and len(primary) >= 1
        and 0.001 <= primary_area <= 0.35
    )
    return {
        "eligible": eligible, "overlap_threshold": threshold,
        "gemini_direct_count": len(ga), "opus_direct_count": len(oa),
        "matched_gemini_indices": matched_g, "matched_opus_indices": matched_o,
        "gemini_region_coverage": gem_coverage, "opus_region_coverage": opus_coverage,
        "primary_consensus_regions": primary, "primary_consensus_area_fraction": primary_area,
        "gemini_unmatched_regions": [x for i, x in enumerate(ga) if i not in matched_g],
        "opus_unmatched_regions": [x for i, x in enumerate(oa) if i not in matched_o],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--selected-panel", type=Path, required=True)
    p.add_argument("--gemini-annotation", type=Path, action="append", required=True)
    p.add_argument("--opus-annotation", type=Path, action="append", required=True)
    p.add_argument("--output-panel", type=Path, required=True)
    p.add_argument("--output-audit", type=Path, required=True)
    p.add_argument("--allow-missing-opus-index", type=int, action="append", default=[])
    args = p.parse_args()
    if args.output_panel.exists() or args.output_audit.exists():
        raise FileExistsError("refusing to overwrite frozen consensus outputs")
    panel_raw = args.selected_panel.read_bytes(); panel = json.loads(panel_raw)
    cases = {int(x["panel_index"]): x for x in panel["cases"]}
    gemini, gemini_hashes = load_rows(args.gemini_annotation, "gemini-3.1-pro")
    opus, opus_hashes = load_rows(args.opus_annotation, "claude-opus-5")
    allowed_missing_opus = set(args.allow_missing_opus_index)
    if not allowed_missing_opus <= set(cases):
        raise ValueError("allowed missing Opus index is absent from selected panel")
    if not set(cases) <= set(gemini) or set(opus) != set(cases) - allowed_missing_opus:
        raise ValueError("dual-model annotation coverage mismatch")
    audit_cases = []; output_cases = []
    for index in sorted(cases):
        if index not in opus:
            audit_cases.append({"panel_index": index, "target_choice": cases[index]["target_choice"],
                                "eligible": False, "exclusion_reason": "allowed_opus_format_failure"})
            continue
        g = gemini[index]; o = opus[index]; case = cases[index]
        if g["image_sha256"] != case["image_sha256"] or o["image_sha256"] != case["image_sha256"]:
            raise ValueError("image identity mismatch")
        result = consensus(g["annotation"], o["annotation"])
        audit_cases.append({"panel_index": index, "target_choice": case["target_choice"], **result,
                            "gemini_annotation": g["annotation"], "opus_annotation": o["annotation"]})
        if result["eligible"]:
            selected = dict(case)
            selected["external_annotation"] = {
                "annotation_kind": "gemini31pro_regions_blindly_corroborated_by_claude_opus5",
                "boxes": [x["box"] for x in result["primary_consensus_regions"]],
                "regions": result["primary_consensus_regions"], "diagnostically_relevant": True,
                "evidence_type": "focal" if len(result["primary_consensus_regions"]) == 1 else "multifocal",
                "gemini_confidence": g["annotation"]["confidence"],
                "opus_confidence": o["annotation"]["confidence"],
                "all_gemini_regions": g["annotation"]["regions"],
                "all_opus_regions": o["annotation"]["regions"],
                "gemini_unmatched_regions": result["gemini_unmatched_regions"],
                "opus_unmatched_regions": result["opus_unmatched_regions"],
            }
            selected["dual_model_consensus"] = {k: v for k, v in result.items()
                                                 if k not in {"primary_consensus_regions", "gemini_unmatched_regions", "opus_unmatched_regions"}}
            output_cases.append(selected)
    output = {
        "schema_version": 3, "created_at": datetime.now(timezone.utc).isoformat(), "status": "frozen",
        "annotation_boundary": "dual external VLM pseudo-reference; pathology-expert verification remains required",
        "selection_uses_target_outputs": False, "source_selected_panel": str(args.selected_panel.resolve()),
        "source_selected_panel_sha256": hashlib.sha256(panel_raw).hexdigest(),
        "gemini_sources": gemini_hashes, "opus_sources": opus_hashes,
        "selected_for_review_count": len(cases), "reviewed_count": len(opus),
        "allowed_missing_opus_indices": sorted(allowed_missing_opus),
        "consensus_selected_count": len(output_cases),
        "target_choice_counts": dict(Counter(x["target_choice"] for x in output_cases)),
        "cases": output_cases,
    }
    args.output_panel.parent.mkdir(parents=True, exist_ok=True)
    args.output_panel.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {"schema_version": 1, "status": "completed", "selected_for_review_count": len(cases),
             "reviewed_count": len(opus), "allowed_missing_opus_indices": sorted(allowed_missing_opus),
             "eligible_count": len(output_cases), "thresholds": {"per_region_overlap_coefficient": 0.20,
             "minimum_independently_corroborated_regions": 1, "bidirectional_region_coverage": "reported_not_gated",
             "primary_area_min": 0.001,
             "primary_area_max": 0.35, "confidence_min": 0.65}, "cases": audit_cases,
             "output_panel_sha256": sha256_file(args.output_panel)}
    args.output_audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reviewed_count": len(opus), "eligible_count": len(output_cases),
                      "target_choice_counts": output["target_choice_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
