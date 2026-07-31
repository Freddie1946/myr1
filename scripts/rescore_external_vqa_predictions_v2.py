#!/usr/bin/env python3
"""Rescore existing predictions under the corrected dual-report VQA contract."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from external_vqa_contract import pathvqa_score
from external_vqa_contract import omnimed_score
from external_vqa_contract import sha256_file


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def pathvqa_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [pathvqa_score(row["completion"], row["answer"]) for row in rows]
    return {
        "primary_metric": "pathvqa_paper_metric_family_by_answer_type",
        "repository_token_overlap_mean_diagnostic": statistics.fmean(
            row["official_token_overlap_score"] for row in scores
        ),
        "repository_token_f1_mean": statistics.fmean(
            row["official_token_f1_score"] for row in scores
        ),
        "contract_aligned_exact_correct": sum(
            row["contract_aligned_exact_match"] for row in scores
        ),
        "contract_aligned_exact_accuracy": sum(
            row["contract_aligned_exact_match"] for row in scores
        ) / len(scores),
        "legacy_strict_exact_correct": sum(row["strict_exact_match"] for row in scores),
        "legacy_strict_exact_accuracy": sum(row["strict_exact_match"] for row in scores)
        / len(scores),
        "by_answer_type": {
            kind: {
                "count": len(selected := [s for s in scores if s["answer_type"] == kind]),
                "repository_token_overlap_mean_diagnostic": statistics.fmean(
                    s["official_token_overlap_score"] for s in selected
                ),
                "repository_token_f1_mean": statistics.fmean(
                    s["official_token_f1_score"] for s in selected
                ),
                "contract_aligned_exact_accuracy": sum(
                    s["contract_aligned_exact_match"] for s in selected
                ) / len(selected),
                "paper_metric_role": (
                    "accuracy" if kind == "yes_no" else "strict_exact_and_macro_token_f1"
                ),
            }
            for kind in ("yes_no", "free_form")
        },
    }


def omni_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "question": row["question"],
        "gt_answer": row["gt_answer"],
        **{f"option_{letter}": row["option_texts"].get(letter) for letter in "ABCD"},
    }


def omnimed_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [omnimed_score(row["completion"], omni_record(row)) for row in rows]
    aligned = sum(row["contract_aligned_correct"] for row in scores)
    raw = sum(row["official_most_similar_correct"] for row in scores)
    result: dict[str, Any] = {
        "primary_metric": "contract_aligned_sequence_matcher_option_accuracy",
        "contract_aligned_correct": aligned,
        "contract_aligned_accuracy": aligned / len(scores),
        "official_raw_completion_correct": raw,
        "official_raw_completion_accuracy": raw / len(scores),
        "strict_text_correct": sum(row["strict_text_correct"] for row in scores),
        "strict_text_accuracy": sum(row["strict_text_correct"] for row in scores)
        / len(scores),
        "raw_wrong_aligned_correct": sum(
            not row["official_most_similar_correct"] and row["contract_aligned_correct"]
            for row in scores
        ),
        "raw_correct_aligned_wrong": sum(
            row["official_most_similar_correct"] and not row["contract_aligned_correct"]
            for row in scores
        ),
    }
    by_source = {}
    for source in sorted({row["dataset"] for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["dataset"] == source]
        by_source[source] = {
            "count": len(indices),
            "contract_aligned_accuracy": sum(
                scores[index]["contract_aligned_correct"] for index in indices
            ) / len(indices),
            "official_raw_completion_accuracy": sum(
                scores[index]["official_most_similar_correct"] for index in indices
            ) / len(indices),
        }
    result["by_source"] = by_source
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    index = {"schema_version": 2, "runs": {}}
    for predictions in sorted(args.run_root.glob("*/*/predictions.jsonl")):
        rows = read_jsonl(predictions)
        if not rows or "completion" not in rows[0]:
            continue
        model, task = predictions.relative_to(args.run_root).parts[:2]
        if task == "pathvqa":
            metrics = pathvqa_metrics(rows)
        elif task == "omnimedvqa":
            metrics = omnimed_metrics(rows)
        else:
            continue
        payload = {
            "schema_version": 2,
            "scoring_policy": "official/source-compatible plus contract-aligned extraction; no inference rerun",
            "model": model,
            "task": task,
            "count": len(rows),
            "source_predictions": str(predictions.resolve()),
            "source_predictions_sha256": sha256_file(predictions),
            **metrics,
        }
        destination = args.output_root / model / task / "metrics_v2.json"
        write_json(destination, payload)
        index["runs"][f"{model}/{task}"] = str(destination.resolve())
    write_json(args.output_root / "index.json", index)
    print(json.dumps({"status": "passed", "rescored_runs": len(index["runs"])}))


if __name__ == "__main__":
    main()
