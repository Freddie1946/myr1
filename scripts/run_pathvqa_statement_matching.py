#!/usr/bin/env python3
"""Evaluate pathology image-text encoders on PathVQA yes/no questions.

This is an exploratory image/question/answer statement-matching protocol, not
generative VQA.  The answer token is deliberately placed before the question
so that it cannot be removed by a CLIP-family tokenizer's context truncation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from run_pathmmu_matching_diagnostic import load_backend, record_sha256, sha256_file, write_json


STATEMENT_TEMPLATE = "Answer: {answer}. Question: {question}"


def candidate_statements(question: str) -> dict[str, str]:
    question = " ".join(str(question).strip().split())
    if not question:
        raise ValueError("question must not be empty")
    return {
        "yes": STATEMENT_TEMPLATE.format(answer="Yes", question=question),
        "no": STATEMENT_TEMPLATE.format(answer="No", question=question),
    }


def select_yes_no(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in records if str(row.get("answer_type", "")).lower() == "yes_no"]
    for row in selected:
        answer = str(row.get("answer", "")).strip().lower()
        if answer not in {"yes", "no"}:
            raise ValueError(f"invalid yes/no target at source index {row.get('index')}: {answer!r}")
        if not Path(str(row.get("image", ""))).is_file():
            raise FileNotFoundError(row.get("image"))
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("plip", "conch"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_records = json.loads(args.data.read_text(encoding="utf-8"))
    selected = select_yes_no(all_records)
    if len(all_records) != 6719 or len(selected) != 3362:
        raise ValueError(
            f"unexpected PathVQA test contract: total={len(all_records)}, yes_no={len(selected)}"
        )
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        selected = selected[: args.limit]
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    predictions_path = args.output_dir / "predictions.jsonl"
    config = {
        "schema_version": 1,
        "status": "running",
        "backend": args.backend,
        "task": "exploratory_image_question_answer_statement_matching_v1",
        "scientific_boundary": (
            "exploratory image-text retrieval-style diagnostic; not generative VQA and not "
            "directly comparable to VLM free-generation accuracy"
        ),
        "statement_template": STATEMENT_TEMPLATE,
        "answer_token_precedes_question_to_survive_context_truncation": True,
        "split": "PathVQA test yes_no",
        "test_accessed": True,
        "model_path": str(args.model.resolve()),
        "data_path": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "full_contract_count": len(all_records),
        "full_yes_no_count": 3362,
        "selected_count": len(selected),
        "dtype": "bfloat16",
        "quantization": "none",
    }
    write_json(args.output_dir / "run_config.json", config)
    _, score = load_backend(args.backend, args.model)

    correct = 0
    predicted_yes = 0
    target_counts = {"yes": 0, "no": 0}
    class_correct = {"yes": 0, "no": 0}
    with predictions_path.open("w", encoding="utf-8") as handle:
        for ordinal, row in enumerate(selected):
            statements = candidate_statements(row["question"])
            labels = ["yes", "no"]
            with Image.open(row["image"]) as source:
                similarities = score(source.convert("RGB"), [statements[x] for x in labels])
            prediction = labels[int(similarities.argmax().item())]
            target = str(row["answer"]).strip().lower()
            is_correct = prediction == target
            correct += int(is_correct)
            predicted_yes += int(prediction == "yes")
            target_counts[target] += 1
            class_correct[target] += int(is_correct)
            output = {
                "ordinal": ordinal,
                "source_index": int(row["index"]),
                "source_record_sha256": record_sha256(row),
                "image": row["image"],
                "question": row["question"],
                "target_answer": target,
                "candidate_statements": statements,
                "similarities": {label: float(value) for label, value in zip(labels, similarities.tolist())},
                "predicted_answer": prediction,
                "correct": is_correct,
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()
            if (ordinal + 1) % 50 == 0 or ordinal + 1 == len(selected):
                print(json.dumps({"completed": ordinal + 1, "total": len(selected), "correct": correct}), flush=True)

    recalls = {label: class_correct[label] / target_counts[label] for label in ("yes", "no")}
    metrics = {
        **config,
        "status": "completed",
        "count": len(selected),
        "correct": correct,
        "accuracy": correct / len(selected),
        "predicted_yes_count": predicted_yes,
        "predicted_yes_rate": predicted_yes / len(selected),
        "target_counts": target_counts,
        "class_recall": recalls,
        "balanced_accuracy": sum(recalls.values()) / 2,
        "predictions_sha256": sha256_file(predictions_path),
    }
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "run_config.json", {**config, "status": "completed"})
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
