#!/usr/bin/env python3
"""Merge independently executed attention-intervention panel shards."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from run_attention_intervention_experiment import aggregate
from run_visual_fidelity_experiment import canonical_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite merge output: {args.output_dir}")
    metrics = [json.loads((path / "metrics.json").read_text()) for path in args.shard]
    invariant_fields = (
        "model_label",
        "model_path",
        "model_config_sha256",
        "panel_path",
        "panel_sha256",
        "layer_strategies",
        "perturbations",
        "modes",
        "fractions",
        "random_permutations",
    )
    for field in invariant_fields:
        if len({canonical_json(row[field]) for row in metrics}) != 1:
            raise ValueError(f"shard mismatch for {field}")
    records = []
    attentions = []
    seen = set()
    for shard in args.shard:
        for name, destination in (("case_results.jsonl", records), ("attention_maps.jsonl", attentions)):
            with (shard / name).open() as handle:
                destination.extend(json.loads(line) for line in handle if line.strip())
    indices = [row["panel_index"] for row in records]
    if len(indices) != len(set(indices)) or sorted(indices) != list(range(24)):
        raise ValueError(f"expected exactly panel indices 0..23, found {sorted(indices)}")
    if sorted(row["panel_index"] for row in attentions) != list(range(24)):
        raise ValueError("attention shard indices do not match complete panel")
    records.sort(key=lambda row: row["panel_index"])
    attentions.sort(key=lambda row: row["panel_index"])
    args.output_dir.mkdir(parents=True)
    case_path = args.output_dir / "case_results.jsonl"
    attention_path = args.output_dir / "attention_maps.jsonl"
    case_path.write_text("".join(canonical_json(row) + "\n" for row in records), encoding="utf-8")
    attention_path.write_text("".join(canonical_json(row) + "\n" for row in attentions), encoding="utf-8")
    first = metrics[0]
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_exploratory",
        "formal_result": False,
        "method": first["method"],
        "attention_region_selection_uses_answer_logits": False,
        "primary_endpoint": first["primary_endpoint"],
        "secondary_endpoint": first["secondary_endpoint"],
        **{field: first[field] for field in invariant_fields},
        "panel_indices": list(range(24)),
        "attention_maps": str(attention_path.resolve()),
        "attention_maps_sha256": sha256_file(attention_path),
        "case_results": str(case_path.resolve()),
        "case_results_sha256": sha256_file(case_path),
        "source_shards": [str(path.resolve()) for path in args.shard],
        **aggregate(records, first["layer_strategies"]),
    }
    output = args.output_dir / "metrics.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(canonical_json(result))


if __name__ == "__main__":
    main()
