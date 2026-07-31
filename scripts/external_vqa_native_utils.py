#!/usr/bin/env python3
"""Shared output/config utilities for native external-VQA backend adapters."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from external_vqa_contract import record_sha256, score_record, sha256_file


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_existing(
    path: Path, records: list[dict[str, Any]], task: str
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for index, row in enumerate(rows):
        if row.get("index") != index:
            raise ValueError(f"non-contiguous prediction at row {index}")
        if index >= len(records):
            raise ValueError("existing predictions exceed selected records")
        if row.get("source_record_sha256") != record_sha256(records[index]):
            raise ValueError(f"source record changed at row {index}")
        row.update(score_record(task, row["completion"], records[index]))
    return rows


def append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def summarize(
    rows: list[dict[str, Any]],
    *,
    task: str,
    common: dict[str, Any],
    predictions_path: Path,
) -> dict[str, Any]:
    lengths = [int(row["generated_token_count"]) for row in rows]
    result = {
        **common,
        "status": "completed",
        "count": len(rows),
        "empty_completion_count": sum(not row["completion"].strip() for row in rows),
        "mean_generated_tokens": statistics.fmean(lengths),
        "median_generated_tokens": statistics.median(lengths),
        "maximum_generated_tokens": max(lengths),
        "generation_cap_hit_count": sum(row["reached_generation_cap"] for row in rows),
        "predictions_sha256": sha256_file(predictions_path),
    }
    if task == "pathvqa":
        yes_no = [row for row in rows if row["answer_type"] == "yes_no"]
        free = [row for row in rows if row["answer_type"] == "free_form"]
        correct = sum(row["exact_match"] for row in rows)
        result.update(
            {
                "primary_metric": "pathvqa_paper_metric_family_by_answer_type",
                "repository_token_overlap_mean_diagnostic": statistics.fmean(
                    row["official_token_overlap_score"] for row in rows
                ),
                "repository_token_f1_mean": statistics.fmean(
                    row["official_token_f1_score"] for row in rows
                ),
                "contract_aligned_exact_correct": sum(
                    row["contract_aligned_exact_match"] for row in rows
                ),
                "contract_aligned_exact_accuracy": sum(
                    row["contract_aligned_exact_match"] for row in rows
                ) / len(rows),
                "legacy_strict_exact_metric": "normalized_whole_completion_exact_match",
                "correct": correct,
                "accuracy": correct / len(rows),
                "yes_no_count": len(yes_no),
                "yes_no_correct": sum(row["exact_match"] for row in yes_no),
                "yes_no_accuracy": sum(row["exact_match"] for row in yes_no)
                / len(yes_no),
                "free_form_count": len(free),
                "free_form_correct": sum(row["exact_match"] for row in free),
                "free_form_accuracy": sum(row["exact_match"] for row in free)
                / len(free),
                "paper_yes_no_contract_aligned_accuracy": sum(
                    row["contract_aligned_exact_match"] for row in yes_no
                ) / len(yes_no),
                "paper_free_form_strict_exact_accuracy": sum(
                    row["strict_exact_match"] for row in free
                ) / len(free),
                "paper_free_form_macro_token_f1": statistics.fmean(
                    row["repository_token_f1_score"] for row in free
                ),
            }
        )
    else:
        official = sum(row["official_most_similar_correct"] for row in rows)
        aligned = sum(row["contract_aligned_correct"] for row in rows)
        strict = sum(row["strict_text_correct"] for row in rows)
        by_source = {}
        for source in sorted({row["dataset"] for row in rows}):
            selected = [row for row in rows if row["dataset"] == source]
            by_source[source] = {
                "count": len(selected),
                "official_correct": sum(
                    row["official_most_similar_correct"] for row in selected
                ),
                "official_accuracy": sum(
                    row["official_most_similar_correct"] for row in selected
                )
                / len(selected),
                "strict_text_correct": sum(row["strict_text_correct"] for row in selected),
                "strict_text_accuracy": sum(
                    row["strict_text_correct"] for row in selected
                )
                / len(selected),
                "contract_aligned_correct": sum(
                    row["contract_aligned_correct"] for row in selected
                ),
                "contract_aligned_accuracy": sum(
                    row["contract_aligned_correct"] for row in selected
                ) / len(selected),
            }
        result.update(
            {
                "primary_metric": "contract_aligned_sequence_matcher_option_accuracy",
                "contract_aligned_correct": aligned,
                "contract_aligned_accuracy": aligned / len(rows),
                "official_raw_completion_metric": "official_sequence_matcher_option_accuracy",
                "official_correct": official,
                "official_accuracy": official / len(rows),
                "strict_text_correct": strict,
                "strict_text_accuracy": strict / len(rows),
                "by_source": by_source,
            }
        )
    return result


def result_row(
    *,
    index: int,
    task: str,
    record: dict[str, Any],
    completion: str,
    token_count: int,
    ended_with_eos: bool | None,
    score: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    row = {
        "index": index,
        "source_record_sha256": record_sha256(record),
        "image": record["image"],
        "question": record["question"],
        "completion": completion,
        "generated_token_count": token_count,
        "ended_with_eos": ended_with_eos,
        "reached_generation_cap": token_count >= max_new_tokens,
        **score,
    }
    if task == "pathvqa":
        row["answer"] = record["answer"]
    else:
        row.update(
            {
                "dataset": record["dataset"],
                "question_id": record["question_id"],
                "question_type": record["question_type"],
                "gt_answer": record["gt_answer"],
                "option_texts": {
                    letter: record[f"option_{letter}"]
                    for letter in "ABCD"
                    if record.get(f"option_{letter}") is not None
                },
            }
        )
    return row
