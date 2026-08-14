#!/usr/bin/env python3
"""Freeze clean-correct cases with reference-deletion effects beyond controls."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_effective(row: dict) -> bool:
    return bool(
        row["clean_reasoning_correct"]
        and row["clean_reasoning_forced_agreement"]
        and float(row["reference_target_margin_drop"]) > 0
        and float(row["reference_minus_random_extra_drop"]) > 0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--behavior", type=Path, action="append", required=True)
    parser.add_argument("--output-panel", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--minimum-effective", type=int, default=20)
    args = parser.parse_args()
    if args.output_panel.exists() or args.output_summary.exists():
        raise FileExistsError("refusing to overwrite frozen interpretability outputs")
    panel_raw = args.panel.read_bytes(); panel = json.loads(panel_raw)
    cases = {int(case["panel_index"]): case for case in panel["cases"]}
    rows = {}; behavior_hashes = {}
    for path in args.behavior:
        behavior_hashes[str(path.resolve())] = sha256_file(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line); index = int(row["panel_index"])
            if index in rows:
                raise ValueError(f"duplicate behavior row {index}")
            rows[index] = row
    if set(rows) != set(cases):
        missing = sorted(set(cases) - set(rows)); extra = sorted(set(rows) - set(cases))
        raise ValueError(f"behavior coverage mismatch missing={missing} extra={extra}")
    selected = []
    audit = []
    for index in sorted(cases):
        row = rows[index]
        effective = is_effective(row)
        audit.append({"panel_index": index, "effective": effective,
                      "clean_reasoning_correct": row["clean_reasoning_correct"],
                      "clean_reasoning_forced_agreement": row["clean_reasoning_forced_agreement"],
                      "reference_target_margin_drop": row["reference_target_margin_drop"],
                      "reference_minus_random_extra_drop": row["reference_minus_random_extra_drop"]})
        if effective:
            case = dict(cases[index]); case["behavioral_validation"] = audit[-1]
            selected.append(case)
    if len(selected) < args.minimum_effective:
        raise RuntimeError(f"effective case hard gate failed: {len(selected)} < {args.minimum_effective}")
    output = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "status": "frozen",
        "claim_boundary": "target-model clean-correct cases whose predefined dual-model ROI deletion lowers target margin more than area/shape-matched random deletion; expert confirmation remains required",
        "effective_rule": "clean reasoning correct AND reasoning/forced-choice agreement AND reference margin drop > 0 AND reference-minus-random extra drop > 0",
        "minimum_effective": args.minimum_effective, "source_case_count": len(cases),
        "effective_case_count": len(selected), "target_choice_counts": dict(Counter(x["target_choice"] for x in selected)),
        "source_panel": str(args.panel.resolve()), "source_panel_sha256": hashlib.sha256(panel_raw).hexdigest(),
        "behavior_sources": behavior_hashes, "cases": selected,
    }
    args.output_panel.parent.mkdir(parents=True, exist_ok=True)
    args.output_panel.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {"schema_version": 1, "status": "completed", "source_case_count": len(cases),
               "effective_case_count": len(selected), "minimum_effective": args.minimum_effective,
               "gate_passed": True, "output_panel_sha256": sha256_file(args.output_panel), "cases": audit}
    args.output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("status", "source_case_count", "effective_case_count", "gate_passed")}, sort_keys=True))


if __name__ == "__main__":
    main()
