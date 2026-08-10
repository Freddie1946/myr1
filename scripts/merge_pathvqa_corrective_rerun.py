#!/usr/bin/env python3
"""Merge a target-blind capped-row continuation into immutable PathVQA predictions."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
from typing import Any

from external_vqa_contract import sha256_file


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def merge_rows(
    original: list[dict[str, Any]], corrections: list[dict[str, Any]], *, require_prefix: bool = True
) -> list[dict[str, Any]]:
    if len(original) != 3362:
        raise ValueError(f"expected 3362 original yes/no rows, got {len(original)}")
    correction_by_hash = {row["source_record_sha256"]: row for row in corrections}
    if len(correction_by_hash) != len(corrections):
        raise ValueError("duplicate corrective source record hash")
    original_hashes = {row["source_record_sha256"] for row in original}
    if set(correction_by_hash) - original_hashes:
        raise ValueError("corrective rows are not a subset of the original predictions")
    merged = []
    applied = 0
    for index, old in enumerate(original):
        if old.get("index") != index:
            raise ValueError(f"non-contiguous original row at {index}")
        replacement = correction_by_hash.get(old["source_record_sha256"])
        if replacement is None:
            merged.append(old)
            continue
        if not old.get("reached_generation_cap"):
            raise ValueError(f"correction targets a non-capped row at {index}")
        old_completion = str(old["completion"])
        new_completion = str(replacement["completion"])
        prefix_match = new_completion.startswith(old_completion)
        if require_prefix and not prefix_match:
            raise ValueError(f"deterministic continuation prefix mismatch at {index}")
        row = dict(replacement)
        row["index"] = index
        row["corrective_rerun"] = {
            "applied": True,
            "reason": "target_blind_regeneration_of_legacy_64_token_cap_and_v3_unresolved",
            "old_generated_token_count": old["generated_token_count"],
            "old_completion_sha256": hashlib.sha256(old_completion.encode()).hexdigest(),
            "new_max_tokens": 192,
            "old_batch_size": 8,
            "new_batch_size": 1,
            "old_completion_is_prefix": prefix_match,
            "old_prefix_character_similarity": difflib.SequenceMatcher(
                None, old_completion, new_completion[:len(old_completion)]
            ).ratio(),
        }
        merged.append(row)
        applied += 1
    if applied != len(corrections):
        raise ValueError("not every corrective row was applied")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--corrections", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--allow-regeneration-divergence", action="store_true",
        help="Allow fresh greedy outputs to differ because the corrective batch shape differs.",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    merged = merge_rows(
        load_jsonl(args.original), load_jsonl(args.corrections),
        require_prefix=not args.allow_regeneration_divergence,
    )
    args.output_dir.mkdir(parents=True)
    output = args.output_dir / "predictions.jsonl"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged), encoding="utf-8"
    )
    applied = sum(bool(row.get("corrective_rerun", {}).get("applied")) for row in merged)
    divergent = sum(
        bool(
            row.get("corrective_rerun", {}).get("applied")
            and not row["corrective_rerun"]["old_completion_is_prefix"]
        )
        for row in merged
    )
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "formal_result": False,
        "result_role": "post_hoc_target_blind_generation_cap_correction",
        "original": str(args.original.resolve()),
        "original_sha256": sha256_file(args.original),
        "corrections": str(args.corrections.resolve()),
        "corrections_sha256": sha256_file(args.corrections),
        "count": len(merged),
        "corrections_applied": applied,
        "regeneration_prefix_divergence_count": divergent,
        "regeneration_divergence_explicitly_allowed": args.allow_regeneration_divergence,
        "scientific_boundary": (
            "Corrective rows are fresh batch-size-1 greedy regenerations selected without labels; "
            "they are not claimed to be exact continuations of the legacy batch-size-8 outputs."
        ),
        "predictions": str(output.resolve()),
        "predictions_sha256": sha256_file(output),
        "selection_or_tuning_use_forbidden": True,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "completed", "corrections_applied": applied, "output": str(output.resolve())}))


if __name__ == "__main__":
    main()
