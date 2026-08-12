#!/usr/bin/env python3
"""Merge RISE case shards and recompute aggregate statistics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from run_rise_visual_evidence import aggregate
from run_visual_fidelity_experiment import canonical_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    rows = []
    metas = []
    for shard in args.shard:
        metas.append(json.loads((shard / "metrics.json").read_text()))
        rows.extend(json.loads(line) for line in (shard / "case_results.jsonl").open() if line.strip())
    invariant = ("model_label", "model_path", "model_config_sha256", "panel_path", "panel_sha256", "rise_mask_count", "rise_cells", "rise_visibility_probability", "validation_perturbation", "random_control", "random_controls")
    for key in invariant:
        if len({canonical_json(meta[key]) for meta in metas}) != 1:
            raise ValueError(f"shard mismatch: {key}")
    rows.sort(key=lambda row: row["panel_index"])
    if [row["panel_index"] for row in rows] != list(range(24)):
        raise ValueError("expected complete panel indices 0..23")
    args.output_dir.mkdir(parents=True)
    cases = args.output_dir / "case_results.jsonl"
    cases.write_text("".join(canonical_json(row) + "\n" for row in rows))
    first = metas[0]
    metrics = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_exploratory",
        "formal_result": False,
        "method": first["method"],
        "attribution_target": first["attribution_target"],
        **{key: first[key] for key in invariant},
        "panel_indices": list(range(24)),
        "case_results": str(cases.resolve()),
        "case_results_sha256": sha256_file(cases),
        "source_shards": [str(path.resolve()) for path in args.shard],
        **aggregate(rows),
    }
    output = args.output_dir / "metrics.json"
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(canonical_json(metrics))


if __name__ == "__main__":
    main()
