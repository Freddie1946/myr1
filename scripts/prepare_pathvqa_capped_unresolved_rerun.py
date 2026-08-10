#!/usr/bin/env python3
"""Freeze a target-blind PathVQA rerun subset from capped, parser-unresolved rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from external_vqa_contract import normalize_short_answer, record_sha256, sha256_file


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def select_records(
    full_records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    item_scores: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    yes_no = [row for row in full_records if row.get("answer_type") == "yes_no"]
    if len(yes_no) != 3362 or len(predictions) != len(yes_no) or len(item_scores) != len(yes_no):
        raise ValueError("expected aligned 3362-row PathVQA yes/no inputs")
    selected: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for index, (record, prediction, score) in enumerate(zip(yes_no, predictions, item_scores)):
        source_hash = record_sha256(record)
        if prediction.get("index") != index or prediction.get("source_record_sha256") != source_hash:
            raise ValueError(f"prediction alignment failure at {index}")
        if score.get("yes_no_index") != index or score.get("source_record_sha256") != source_hash:
            raise ValueError(f"score alignment failure at {index}")
        parsed = normalize_short_answer(str(score.get("v3_extracted_answer") or ""))
        if bool(prediction.get("reached_generation_cap")) and parsed not in {"yes", "no"}:
            selected.append(record)
            selection_rows.append(
                {
                    "yes_no_index": index,
                    "source_record_sha256": source_hash,
                    "old_completion_sha256": score["completion_sha256"],
                    "selection_reason": "generation_cap_and_v3_unresolved",
                }
            )
    if not selected:
        raise ValueError("no capped unresolved rows selected")
    return selected, selection_rows


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--item-scores-v3", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    full_records = json.loads(args.data.read_text(encoding="utf-8"))
    selected, selection_rows = select_records(
        full_records, load_jsonl(args.predictions), load_jsonl(args.item_scores_v3)
    )
    args.output_dir.mkdir(parents=True)
    subset_path = args.output_dir / "records.json"
    selection_path = args.output_dir / "selection.jsonl"
    write_json(subset_path, selected)
    selection_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selection_rows),
        encoding="utf-8",
    )
    write_json(
        args.output_dir / "manifest.json",
        {
            "schema_version": 1,
            "status": "frozen_before_corrective_rerun",
            "selection_is_target_blind": True,
            "selection_rule": "old generation cap hit AND deterministic parser v3 unresolved",
            "data": str(args.data.resolve()),
            "data_sha256": sha256_file(args.data),
            "predictions": str(args.predictions.resolve()),
            "predictions_sha256": sha256_file(args.predictions),
            "item_scores_v3": str(args.item_scores_v3.resolve()),
            "item_scores_v3_sha256": sha256_file(args.item_scores_v3),
            "selected_count": len(selected),
            "records": str(subset_path.resolve()),
            "records_sha256": sha256_file(subset_path),
            "selection": str(selection_path.resolve()),
            "selection_sha256": sha256_file(selection_path),
        },
    )
    print(json.dumps({"selected_count": len(selected), "output_dir": str(args.output_dir.resolve())}))


if __name__ == "__main__":
    main()
