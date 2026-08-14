#!/usr/bin/env python3
"""Freeze cases present in an expanded panel but absent from a completed panel."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--completed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    expanded_raw = args.expanded.read_bytes(); completed_raw = args.completed.read_bytes()
    expanded = json.loads(expanded_raw); completed = json.loads(completed_raw)
    done = {int(case["panel_index"]): case for case in completed["cases"]}
    all_cases = {int(case["panel_index"]): case for case in expanded["cases"]}
    if not set(done) <= set(all_cases):
        raise ValueError("completed panel is not a subset of expanded panel")
    for index, case in done.items():
        if case["image_sha256"] != all_cases[index]["image_sha256"]:
            raise ValueError(f"image identity mismatch at panel {index}")
    difference = [all_cases[index] for index in sorted(set(all_cases) - set(done))]
    output = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "status": "frozen",
        "purpose": "incremental_behavior_evaluation_without_recomputing_completed_cases",
        "expanded_panel": str(args.expanded.resolve()),
        "expanded_panel_sha256": hashlib.sha256(expanded_raw).hexdigest(),
        "completed_panel": str(args.completed.resolve()),
        "completed_panel_sha256": hashlib.sha256(completed_raw).hexdigest(),
        "expanded_count": len(all_cases), "completed_count": len(done),
        "case_count": len(difference), "cases": difference,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"expanded_count": len(all_cases), "completed_count": len(done),
                      "difference_count": len(difference)}, sort_keys=True))


if __name__ == "__main__":
    main()
