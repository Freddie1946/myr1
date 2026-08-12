#!/usr/bin/env python3
"""Uniformly rescore immutable MMMU prediction files with the current parser."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from run_mmmu_retention_diagnostic import extract_choice, record_sha256, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    records = json.loads(args.panel.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.predictions.open(encoding="utf-8")]
    if len(records) != len(rows):
        raise ValueError(f"panel/prediction count mismatch: {len(records)} != {len(rows)}")

    rescored: list[dict] = []
    changes: list[dict] = []
    for index, (record, source) in enumerate(zip(records, rows, strict=True)):
        if source["index"] != index or source["source_record_sha256"] != record_sha256(record):
            raise ValueError(f"source mismatch at index {index}")
        valid = "ABCDEFGHI"[: len(record["options"])]
        predicted = extract_choice(source["completion"], valid)
        row = dict(source)
        row["predicted_choice"] = predicted
        row["correct"] = predicted == record["answer"]
        rescored.append(row)
        if predicted != source.get("predicted_choice"):
            changes.append({
                "index": index,
                "old": source.get("predicted_choice"),
                "new": predicted,
                "target": record["answer"],
            })

    args.output_dir.mkdir(parents=True)
    output_predictions = args.output_dir / "predictions.jsonl"
    with output_predictions.open("w", encoding="utf-8") as handle:
        for row in rescored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
    metrics = {
        "schema_version": 1,
        "status": "completed",
        "formal_result": False,
        "rescore_only": True,
        "count": len(rescored),
        "correct": sum(row["correct"] for row in rescored),
        "accuracy": sum(row["correct"] for row in rescored) / len(rescored),
        "choice_extracted": sum(row["predicted_choice"] is not None for row in rescored),
        "generation_cap_hit_count": sum(row["reached_generation_cap"] for row in rescored),
        "source_predictions": str(args.predictions.resolve()),
        "source_predictions_sha256": digest,
        "panel": str(args.panel.resolve()),
        "panel_sha256": sha256_file(args.panel),
        "rescored_predictions_sha256": sha256_file(output_predictions),
        "parser_changes": changes,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
