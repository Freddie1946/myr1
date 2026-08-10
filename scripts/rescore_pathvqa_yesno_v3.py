#!/usr/bin/env python3
"""Audit-preserving PathVQA yes/no rescoring under the v3 parser correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from external_vqa_contract import normalize_short_answer, pathvqa_score, sha256_file


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must use NAME=/absolute/path form")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("prediction name and path must be non-empty")
    return name, Path(raw_path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object row at {path}:{line_number}")
            rows.append(row)
    return rows


def select_yes_no(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if str(row.get("answer_type") or "") == "yes_no"
        or normalize_short_answer(str(row.get("answer") or "")) in {"yes", "no"}
    ]
    if len(selected) != 3362:
        raise ValueError(f"expected 3362 yes/no rows, got {len(selected)}")
    return selected


def completion_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def legacy_v2_yes_no_correct(completion: str, answer: str) -> bool:
    """Reproduce the superseded v2 extraction even when old rows lack score fields."""

    tagged = re.findall(
        r"<answer\b[^>]*>\s*(.*?)(?:</answer\s*>|$)", completion, re.I | re.S
    )
    if tagged:
        extracted = tagged[-1].strip().rstrip("</ ")
    else:
        marked = re.findall(
            r"(?:final\s+answer|answer)\s*(?:is|:)\s*\b(yes|no)\b",
            completion,
            re.I,
        )
        if marked:
            extracted = marked[-1].lower()
        else:
            leading = re.match(r"^\s*(yes|no)\b", completion, re.I)
            extracted = leading.group(1).lower() if leading else completion.strip()
    return normalize_short_answer(extracted) == normalize_short_answer(answer)


def rescore(name: str, source: Path, output_root: Path) -> dict[str, Any]:
    rows = select_yes_no(read_jsonl(source))
    item_rows = []
    old_correct = new_correct = old_wrong_new_correct = old_correct_new_wrong = 0
    cap_hits = 0
    parseable = 0
    sources: Counter[str] = Counter()
    for yes_no_index, row in enumerate(rows):
        score = pathvqa_score(str(row["completion"]), str(row["answer"]))
        old = legacy_v2_yes_no_correct(str(row["completion"]), str(row["answer"]))
        new = bool(score["contract_aligned_exact_match"])
        old_correct += old
        new_correct += new
        old_wrong_new_correct += (not old and new)
        old_correct_new_wrong += (old and not new)
        cap_hits += bool(row.get("reached_generation_cap", False))
        extracted = normalize_short_answer(score["contract_aligned_answer"])
        parseable += extracted in {"yes", "no"}
        sources[score["contract_aligned_answer_source"]] += 1
        item_rows.append(
            {
                "yes_no_index": yes_no_index,
                "source_index": row.get("index"),
                "source_record_sha256": row.get("source_record_sha256"),
                "completion_sha256": completion_sha256(str(row["completion"])),
                "target": score["normalized_target"],
                "old_contract_aligned_correct": old,
                "v3_extracted_answer": score["contract_aligned_answer"],
                "v3_extraction_source": score["contract_aligned_answer_source"],
                "v3_correct": new,
                "reached_generation_cap": bool(row.get("reached_generation_cap", False)),
            }
        )

    destination = output_root / name
    destination.mkdir(parents=True, exist_ok=False)
    item_path = destination / "item_scores_v3.jsonl"
    with item_path.open("w", encoding="utf-8") as handle:
        for row in item_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metrics = {
        "schema_version": 3,
        "model": name,
        "task": "pathvqa_yes_no",
        "formal_result": False,
        "result_role": "post_hoc_target_blind_parser_correction_sensitivity",
        "count": len(rows),
        "source_predictions": str(source.resolve()),
        "source_predictions_sha256": sha256_file(source),
        "legacy_contract_aligned_correct": old_correct,
        "legacy_contract_aligned_accuracy": old_correct / len(rows),
        "v3_contract_aligned_correct": new_correct,
        "v3_contract_aligned_accuracy": new_correct / len(rows),
        "v3_parseable_count": parseable,
        "v3_parseable_rate": parseable / len(rows),
        "generation_cap_hit_count": cap_hits,
        "generation_cap_hit_rate": cap_hits / len(rows),
        "legacy_wrong_v3_correct": old_wrong_new_correct,
        "legacy_correct_v3_wrong": old_correct_new_wrong,
        "v3_extraction_source_counts": dict(sorted(sources.items())),
        "item_scores": str(item_path.resolve()),
        "item_scores_sha256": sha256_file(item_path),
        "selection_or_tuning_use_forbidden": True,
    }
    metrics_path = destination / "metrics_v3.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", action="append", required=True, type=parse_named_path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    args.output_root.mkdir(parents=True)
    names = [name for name, _ in args.prediction]
    if len(names) != len(set(names)):
        raise ValueError("prediction names must be unique")
    summaries = [rescore(name, path, args.output_root) for name, path in args.prediction]
    index = {
        "schema_version": 3,
        "task": "pathvqa_yes_no",
        "result_role": "post_hoc_target_blind_parser_correction_sensitivity",
        "selection_or_tuning_use_forbidden": True,
        "runs": summaries,
    }
    index_path = args.output_root / "index_v3.json"
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "passed", "runs": len(summaries), "index": str(index_path)}))


if __name__ == "__main__":
    main()
