#!/usr/bin/env python3
"""Freeze a target-model-blind causal-deletion panel from v2 annotations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from annotate_visual_evidence_candidates_aigcbest import sha256_file


def eligible(annotation: dict) -> bool:
    return bool(
        annotation["diagnostically_relevant"]
        and annotation["evidence_type"] in {"focal", "multifocal"}
        and annotation["boxes"]
        and annotation["visual_answerability"] in {"high", "medium"}
        and annotation["causal_panel_suitability"] in {"strong", "moderate"}
        and not annotation["requires_global_context"]
        and float(annotation["confidence"]) >= 0.85
        and float(annotation["evidence_concentration"]) >= 0.60
        and float(annotation["redundancy"]) <= 0.50
        and annotation["deletion_prediction"]
        in {"answer_flip_likely", "confidence_drop_likely"}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    rows = {
        int(row["panel_index"]): row
        for row in map(json.loads, args.annotations.read_text(encoding="utf-8").splitlines())
        if row
    }
    if len(rows) != len(panel["cases"]):
        raise ValueError(f"incomplete annotations: {len(rows)}/{len(panel['cases'])}")
    selected = []
    excluded_reasons: dict[str, int] = {}
    for case in panel["cases"]:
        row = rows[int(case["panel_index"])]
        annotation = row["annotation"]
        if eligible(annotation):
            selected.append({
                **case, "external_annotation": annotation,
                "annotation_model": row["model"],
                "annotation_attempt": row["attempt"],
            })
        else:
            key = annotation["causal_panel_suitability"]
            excluded_reasons[key] = excluded_reasons.get(key, 0) + 1
    result = {
        "schema_version": 2,
        "status": "frozen_before_target_model_outputs",
        "selection_uses_target_outputs": False,
        "annotation_boundary": "external VLM reference regions require blinded pathology-expert verification",
        "source_panel": str(args.panel.resolve()),
        "source_panel_sha256": sha256_file(args.panel),
        "annotations": str(args.annotations.resolve()),
        "annotations_sha256": sha256_file(args.annotations),
        "selection_rule": (
            "diagnostically relevant; focal/multifocal nonempty boxes; visual answerability "
            "high/medium; suitability strong/moderate; no required global context; confidence "
            ">=0.85; concentration >=0.60; redundancy <=0.50; predicted confidence drop/flip"
        ),
        "candidate_count": len(panel["cases"]),
        "selected_count": len(selected),
        "excluded_suitability_counts": excluded_reasons,
        "cases": selected,
        "clean_correct_filter": "applied only downstream; exclusions and null intervention effects remain auditable",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "candidate_count": result["candidate_count"],
        "selected_count": result["selected_count"],
        "output": str(args.output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
