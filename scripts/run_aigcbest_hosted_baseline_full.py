#!/usr/bin/env python3
"""Resumable full evaluation for smoke-approved AIGCBest vision models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from external_vqa_contract import prompt_for_record, record_sha256
from run_aigcbest_hosted_baseline_smoke import (
    OMNIMEDVQA_DATA,
    PATHMMU_PROMPT,
    PATHVQA_DATA,
    SmokeFailure,
    atomic_json,
    request_case,
    score_case,
    sha256_file,
)


PATHMMU_TEST_DATA = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/"
    "pathmmu_image_disjoint_v2/rewritten_records/test_0999.json"
)
DATASETS = {
    "pathmmu": PATHMMU_TEST_DATA,
    "pathvqa": PATHVQA_DATA,
    "omnimedvqa": OMNIMEDVQA_DATA,
}
WRITE_LOCK = threading.Lock()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_case(task: str, index: int, record: dict[str, Any]) -> dict[str, Any]:
    image = Path(record["image"]).resolve()
    if not image.is_file():
        raise FileNotFoundError(image)
    prompt = (
        PATHMMU_PROMPT.format(problem=record["problem"])
        if task == "pathmmu"
        else prompt_for_record(task, record)
    )
    return {
        "case_id": f"{task}_{index}",
        "task": task,
        "index": index,
        "record": record,
        "source_record_sha256": record_sha256(record),
        "image": image,
        "image_sha256": sha256_file(image),
        "prompt": prompt,
        "max_tokens": 1024 if task == "pathmmu" else (128 if task == "pathvqa" else 64),
    }


def task_fields(case: dict[str, Any]) -> dict[str, Any]:
    record = case["record"]
    if case["task"] == "pathmmu":
        return {"problem": record["problem"], "solution": record["solution"]}
    if case["task"] == "pathvqa":
        return {
            "question": record["question"],
            "answer": record["answer"],
            "answer_type": record["answer_type"],
        }
    return {
        key: record.get(key)
        for key in (
            "question_id", "question", "gt_answer", "target_choice",
            "option_A", "option_B", "option_C", "option_D", "dataset", "modality",
        )
    }


def evaluate_one(api_key: str, model: str, case: dict[str, Any]) -> dict[str, Any]:
    base = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "formal_result": True,
        "result_class": "authorized_full_contemporary_baseline_evaluation",
        "requested_model": model,
        "case_id": case["case_id"],
        "task": case["task"],
        "index": case["index"],
        "source_record_sha256": case["source_record_sha256"],
        "image_path": str(case["image"]),
        "image_sha256": case["image_sha256"],
        "prompt": case["prompt"],
        "prompt_sha256": hashlib.sha256(case["prompt"].encode()).hexdigest(),
        "max_tokens": case["max_tokens"],
        **task_fields(case),
    }
    try:
        response = request_case(api_key, model, case)
    except SmokeFailure as exc:
        return {**base, "status": "failed", "error": str(exc), **exc.evidence}
    row = {
        **base,
        **response,
        "status": "passed" if not response["validation_errors"] else "failed",
    }
    if row["status"] == "passed":
        row["score"] = score_case(case, response["completion"])
    return row


def load_existing(path: Path, task: str, expected_count: int) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return existing
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        case_id = str(row.get("case_id"))
        if row.get("task") != task or case_id in existing:
            raise ValueError(f"invalid/duplicate prediction at line {line_number}: {case_id}")
        index = int(row["index"])
        if not 0 <= index < expected_count or case_id != f"{task}_{index}":
            raise ValueError(f"out-of-contract prediction at line {line_number}: {case_id}")
        existing[case_id] = row
    return existing


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with WRITE_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def aggregate(task: str, rows: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    passed = [row for row in rows if row.get("status") == "passed"]
    failed = [row for row in rows if row.get("status") != "passed"]
    result: dict[str, Any] = {
        "expected_count": expected,
        "recorded_count": len(rows),
        "passed_count": len(passed),
        "failed_count": len(failed),
        "complete": len(rows) == expected,
    }
    if not passed:
        return result
    if task == "pathmmu":
        result.update({
            "correct": sum(row["score"]["accuracy_reward"] == 1.0 for row in passed),
            "accuracy": sum(row["score"]["accuracy_reward"] for row in passed) / len(passed),
            "format_pass_rate": sum(row["score"]["format_reward"] for row in passed) / len(passed),
        })
    elif task == "pathvqa":
        yes_no = [row for row in passed if row["answer_type"] == "yes_no"]
        free_form = [row for row in passed if row["answer_type"] == "free_form"]
        result.update({
            "yes_no_count": len(yes_no),
            "yes_no_contract_aligned_correct": sum(
                row["score"]["contract_aligned_exact_match"] for row in yes_no
            ),
            "free_form_count": len(free_form),
            "free_form_semantic_judge_status": "pending_separate_gpt5mini_pass",
            "free_form_strict_exact_correct": sum(
                row["score"]["strict_exact_match"] for row in free_form
            ),
        })
    else:
        result.update({
            "legacy_official_correct": sum(
                row["score"]["official_most_similar_correct"] for row in passed
            ),
            "legacy_official_accuracy": sum(
                row["score"]["official_most_similar_correct"] for row in passed
            ) / len(passed),
            "contract_aligned_correct": sum(
                row["score"]["contract_aligned_correct"] for row in passed
            ),
            "contract_aligned_accuracy": sum(
                row["score"]["contract_aligned_correct"] for row in passed
            ) / len(passed),
        })
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=sorted(DATASETS), required=True)
    parser.add_argument("--smoke-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("AIGCBEST_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("AIGCBEST_API_KEY is not set")
    smoke = json.loads(args.smoke_summary.read_text(encoding="utf-8"))
    if smoke.get("status") != "passed" or smoke.get("requested_model") != args.model:
        raise RuntimeError("full evaluation requires a passed exact-identity smoke")
    if smoke.get("served_models") != [args.model]:
        raise RuntimeError("smoke served-model identity is not exact")

    dataset = DATASETS[args.task]
    records = json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("evaluation dataset is empty or malformed")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = output_dir / "predictions.jsonl"
    config_path = output_dir / "run_config.json"
    summary_path = output_dir / "metrics.json"
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(
        ["/home/dataset-assist-0/czy/wjy/.local-git/usr/bin/git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    config = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "formal_result": True,
        "result_class": "authorized_full_contemporary_baseline_evaluation",
        "model": args.model,
        "task": args.task,
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "expected_count": len(records),
        "smoke_summary": str(args.smoke_summary.resolve()),
        "smoke_summary_sha256": sha256_file(args.smoke_summary),
        "workers": args.workers,
        "temperature": 0,
        "max_tokens": 1024 if args.task == "pathmmu" else (128 if args.task == "pathvqa" else 64),
        "maximum_physical_attempts_per_record": 3,
        "ambiguous_transport_retry": False,
        "repository_commit": commit,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "test_accessed": args.task == "pathmmu",
    }
    if config_path.exists():
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        stable = {key: value for key, value in config.items() if key != "created_at"}
        prior_stable = {key: value for key, value in prior.items() if key != "created_at"}
        if stable != prior_stable:
            raise RuntimeError("resume configuration differs from the frozen run config")
    else:
        atomic_json(config_path, config)

    existing = load_existing(predictions, args.task, len(records))
    pending = [
        build_case(args.task, index, record)
        for index, record in enumerate(records)
        if f"{args.task}_{index}" not in existing
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for row in executor.map(
            lambda case: evaluate_one(api_key, args.model, case), pending
        ):
            append_jsonl(predictions, row)
            existing[row["case_id"]] = row
            if len(existing) % 100 == 0:
                print(canonical_json({
                    "model": args.model, "task": args.task,
                    "recorded": len(existing), "expected": len(records),
                }), flush=True)

    ordered = [existing[f"{args.task}_{index}"] for index in range(len(records))]
    metrics = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "formal_result": True,
        "result_class": "authorized_full_contemporary_baseline_evaluation",
        "model": args.model,
        "task": args.task,
        "status": "completed" if all(row.get("status") == "passed" for row in ordered) else "completed_with_failures",
        "predictions": str(predictions),
        **aggregate(args.task, ordered, len(records)),
    }
    atomic_json(summary_path, metrics)
    print(canonical_json(metrics))


if __name__ == "__main__":
    main()
