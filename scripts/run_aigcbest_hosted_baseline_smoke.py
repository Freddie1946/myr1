#!/usr/bin/env python3
"""Fixed four-case behavioral smoke for hosted vision baselines.

The smoke validates transport, model identity, image acceptance, raw-output retention,
answer extraction, and deterministic scoring.  Its accuracy is never a model-selection gate.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from external_vqa_contract import prompt_for_record, record_sha256, score_record
from pathmmu_rewards import accuracy_reward, choice_letter, format_reward


BASE_URL = "https://api2.aigcbest.top"
PATHMMU_DATA = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/"
    "pathmmu_image_disjoint_v2/rewritten_records/validation_0385.json"
)
PATHVQA_DATA = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/"
    "external_vqa_contract_v1_20260729/pathvqa_test_6719.json"
)
OMNIMEDVQA_DATA = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/"
    "external_vqa_contract_v1_20260729/omnimedvqa_four_sources_8518.json"
)
PATHMMU_PROMPT = (
    "{problem} First output the thinking process in <think> </think> tags and then "
    "output the final answer in <answer> </answer> tags."
)
TRANSIENT_HTTP_STATUSES = {408, 429, 500, 502, 503, 529}
RETRY_DELAYS_SECONDS = (5, 15)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fixed_cases() -> list[dict[str, Any]]:
    pathmmu = json.loads(PATHMMU_DATA.read_text(encoding="utf-8"))
    pathvqa = json.loads(PATHVQA_DATA.read_text(encoding="utf-8"))
    omnimed = json.loads(OMNIMEDVQA_DATA.read_text(encoding="utf-8"))
    selections = [
        ("pathmmu", 0, pathmmu[0]),
        # These two PathVQA records were already used in the semantic-judge audit.
        ("pathvqa", 1, pathvqa[1]),
        ("pathvqa", 3, pathvqa[3]),
        ("omnimedvqa", 0, omnimed[0]),
    ]
    cases = []
    for task, index, record in selections:
        if int(record.get("index", index)) != index and task != "pathmmu":
            raise ValueError(f"fixed {task} index drifted: {index}")
        image = Path(record["image"]).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        if task == "pathmmu":
            prompt = PATHMMU_PROMPT.format(problem=record["problem"])
        else:
            prompt = prompt_for_record(task, record)
        cases.append(
            {
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
        )
    return cases


def data_url(image: Path) -> str:
    mime = mimetypes.guess_type(image.name)[0] or "image/jpeg"
    if not mime.startswith("image/"):
        raise ValueError(f"unsupported image MIME: {mime}")
    return f"data:{mime};base64,{base64.b64encode(image.read_bytes()).decode('ascii')}"


def make_payload(model: str, case: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url(case["image"])}},
                    {"type": "text", "text": case["prompt"]},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": case["max_tokens"],
        "stream": False,
    }


def score_case(case: dict[str, Any], completion: str) -> dict[str, Any]:
    if case["task"] != "pathmmu":
        scored = score_record(case["task"], completion, case["record"])
        if case["task"] == "pathvqa" and case["record"]["answer_type"] == "free_form":
            scored["semantic_judge_input"] = {
                "question": case["record"]["question"],
                "reference": case["record"]["answer"],
                "candidate": scored["contract_aligned_answer"],
            }
            scored["semantic_judge_required_for_full_score"] = True
        return scored
    wrapped = [[{"role": "assistant", "content": completion}]]
    return {
        "target_choice": choice_letter(case["record"]["solution"]),
        "predicted_choice": choice_letter(completion),
        "accuracy_reward": accuracy_reward(wrapped, [case["record"]["solution"]])[0],
        "format_reward": format_reward(wrapped)[0],
    }


def request_case(api_key: str, model: str, case: dict[str, Any]) -> dict[str, Any]:
    payload = make_payload(model, case)
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, len(RETRY_DELAYS_SECONDS) + 2):
        request = urllib.request.Request(
            BASE_URL + "/v1/chat/completions",
            data=canonical_json(payload).encode(),
            method="POST",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                status = int(response.status)
                raw = response.read().decode("utf-8", errors="replace")
                safe_headers = {
                    name.lower(): value
                    for name, value in response.headers.items()
                    if name.lower() in {"x-request-id", "x-oneapi-request-id", "content-type"}
                }
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            raw = exc.read().decode("utf-8", errors="replace")
            safe_headers = {}
            attempts.append(
                {"attempt": attempt, "http_status": status, "latency_seconds": time.monotonic() - started}
            )
            if status in TRANSIENT_HTTP_STATUSES and attempt <= len(RETRY_DELAYS_SECONDS):
                time.sleep(RETRY_DELAYS_SECONDS[attempt - 1])
                continue
            raise RuntimeError(f"terminal HTTP {status}: {raw[:500]}") from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            # Delivery is ambiguous; do not risk a duplicate paid request.
            raise RuntimeError(f"ambiguous transport failure, not retried: {exc}") from exc
        latency = time.monotonic() - started
        attempts.append({"attempt": attempt, "http_status": status, "latency_seconds": latency})
        body = json.loads(raw)
        choices = body.get("choices")
        if status != 200 or not isinstance(choices, list) or len(choices) != 1:
            raise RuntimeError("successful HTTP response lacks exactly one choice")
        served = body.get("model")
        if served != model:
            raise RuntimeError(f"served-model mismatch: requested={model!r}, served={served!r}")
        choice = choices[0]
        content = choice.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("empty hosted-model response")
        if not choice.get("finish_reason"):
            raise RuntimeError("missing finish_reason")
        return {
            "http_status": status,
            "safe_response_headers": safe_headers,
            "attempts": attempts,
            "response_id": body.get("id"),
            "served_model": served,
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage"),
            "completion": content,
            "raw_response": body,
        }
    raise AssertionError("unreachable")


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("AIGCBEST_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("AIGCBEST_API_KEY is not set")
    output_dir = args.output_dir.resolve()
    predictions = output_dir / "predictions.jsonl"
    summary_path = output_dir / "smoke_summary.json"
    if predictions.exists() or summary_path.exists():
        raise RuntimeError(f"refusing to repeat a paid smoke in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    catalog_request = urllib.request.Request(
        BASE_URL + "/v1/models",
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": "PathVLM-R1 baseline smoke"},
    )
    with urllib.request.urlopen(catalog_request, timeout=30) as response:
        catalog = json.load(response).get("data", [])
    if sum(isinstance(row, dict) and row.get("id") == args.model for row in catalog) != 1:
        raise RuntimeError(f"exact model absent from authenticated catalog: {args.model}")

    cases = load_fixed_cases()
    completed = []
    for case in cases:
        response = request_case(api_key, args.model, case)
        row = {
            "schema_version": 1,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "formal_result": False,
            "purpose": "adapter_behavior_only_not_model_selection",
            "requested_model": args.model,
            "case_id": case["case_id"],
            "task": case["task"],
            "index": case["index"],
            "source_record_sha256": case["source_record_sha256"],
            "image_path": str(case["image"]),
            "image_sha256": case["image_sha256"],
            "prompt": case["prompt"],
            "prompt_sha256": hashlib.sha256(case["prompt"].encode()).hexdigest(),
            "max_tokens": case["max_tokens"],
            **response,
            "score": score_case(case, response["completion"]),
        }
        append_jsonl(predictions, row)
        completed.append(row)

    summary = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "formal_result": False,
        "accuracy_used_for_selection": False,
        "requested_model": args.model,
        "served_models": sorted({row["served_model"] for row in completed}),
        "case_ids": [row["case_id"] for row in completed],
        "all_images_accepted": len(completed) == len(cases),
        "all_responses_nonempty": all(bool(row["completion"].strip()) for row in completed),
        "all_finish_reasons_present": all(bool(row["finish_reason"]) for row in completed),
        "all_scoring_paths_executed": all(isinstance(row["score"], dict) for row in completed),
        "predictions": str(predictions),
        "dataset_sha256": {
            "pathmmu_validation": sha256_file(PATHMMU_DATA),
            "pathvqa": sha256_file(PATHVQA_DATA),
            "omnimedvqa": sha256_file(OMNIMEDVQA_DATA),
        },
    }
    atomic_json(summary_path, summary)
    print(canonical_json(summary))


if __name__ == "__main__":
    main()
