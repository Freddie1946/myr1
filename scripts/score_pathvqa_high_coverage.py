#!/usr/bin/env python3
"""Combine strict parsing with target-blind intent extraction and independent verification."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from external_vqa_contract import normalize_short_answer, pathvqa_score, sha256_file


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def deterministic(row: dict[str, Any]) -> tuple[str | None, str]:
    score = pathvqa_score(str(row["completion"]), "yes")
    answer = normalize_short_answer(str(score["contract_aligned_answer"] or ""))
    return (answer if answer in {"yes", "no"} else None, str(score["contract_aligned_answer_source"]))


def combine(
    predictions: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    verifications: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    judged = {row["index"]: row for row in judgments}
    verified = {row["index"]: row for row in verifications}
    rows, sources, confusion = [], Counter(), Counter()
    for index, source in enumerate(predictions):
        if source.get("index") != index:
            raise ValueError(f"non-contiguous prediction at {index}")
        target = normalize_short_answer(str(source.get("answer") or source.get("target") or ""))
        if target not in {"yes", "no"}:
            raise ValueError(f"invalid target at {index}")
        answer, deterministic_source = deterministic(source)
        recovery = None
        if answer is not None:
            parse_source = f"deterministic:{deterministic_source}"
        else:
            judgment = judged.get(index)
            verification = verified.get(index)
            if judgment is None:
                parse_source = "unresolved:no_judgment"
            elif judgment["judgment"]["answer"] not in {"yes", "no"}:
                parse_source = "unresolved:intent_judge"
            elif verification is None:
                parse_source = "unresolved:no_verification"
            elif verification["verdict"]["verdict"] != "accept":
                parse_source = f"unresolved:verifier_{verification['verdict']['reason_code']}"
                recovery = {"judgment": judgment["judgment"], "verification": verification["verdict"]}
            else:
                answer = judgment["judgment"]["answer"]
                parse_source = "llm_intent_verified"
                recovery = {"judgment": judgment["judgment"], "verification": verification["verdict"]}
        resolved = answer in {"yes", "no"}
        correct = resolved and answer == target
        sources[parse_source] += 1
        confusion[(target, answer or "unresolved")] += 1
        rows.append({
            "index": index,
            "source_record_sha256": source.get("source_record_sha256"),
            "completion": source["completion"],
            "target": target,
            "answer": answer,
            "resolved": resolved,
            "correct": correct,
            "parse_source": parse_source,
            "recovery": recovery,
        })
    resolved_count = sum(row["resolved"] for row in rows)
    correct_count = sum(row["correct"] for row in rows)
    metrics = {
        "count": len(rows),
        "resolved_count": resolved_count,
        "parse_coverage": resolved_count / len(rows),
        "unresolved_count": len(rows) - resolved_count,
        "correct": correct_count,
        "accuracy_unresolved_wrong": correct_count / len(rows),
        "conditional_accuracy_when_resolved": correct_count / resolved_count if resolved_count else None,
        "parse_source_counts": dict(sorted(sources.items())),
        "confusion": {f"target_{target}__pred_{answer}": count for (target, answer), count in sorted(confusion.items())},
    }
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--intent-judgments", required=True, type=Path)
    parser.add_argument("--intent-verifications", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    inputs = [args.predictions, args.intent_judgments, args.intent_verifications]
    rows, metrics = combine(*(load_jsonl(path) for path in inputs))
    args.output_dir.mkdir(parents=True)
    item_path = args.output_dir / "item_scores.jsonl"
    item_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    payload = {
        "schema_version": 1,
        "status": "completed",
        "formal_result": False,
        "result_role": "post_hoc_high_coverage_sensitivity",
        "selection_or_tuning_use_forbidden": True,
        "scientific_boundary": "LLM stages only recover answer intent; they never see the reference answer or image and cannot change medical correctness.",
        "predictions": str(args.predictions.resolve()),
        "predictions_sha256": sha256_file(args.predictions),
        "intent_judgments": str(args.intent_judgments.resolve()),
        "intent_judgments_sha256": sha256_file(args.intent_judgments),
        "intent_verifications": str(args.intent_verifications.resolve()),
        "intent_verifications_sha256": sha256_file(args.intent_verifications),
        "item_scores": str(item_path.resolve()),
        "item_scores_sha256": sha256_file(item_path),
        **metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
