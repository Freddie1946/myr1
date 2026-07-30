#!/usr/bin/env python3
"""Run resumable PathMMU diagnostics through HuatuoGPT-Vision's native interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import torch
from cli import HuatuoChatbot

from pathmmu_rewards import accuracy_reward, choice_letter, format_reward


QUESTION_TEMPLATE = (
    "{question}\n"
    "Give concise image-grounded reasoning inside <think>...</think>. Then output exactly one "
    "option letter (A, B, C, or D) inside <answer>...</answer>."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_sha256(record: dict[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_existing(path: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for index, row in enumerate(rows):
        if row.get("index") != index:
            raise ValueError(f"non-contiguous existing prediction at row {index}")
        if row.get("source_record_sha256") != record_sha256(records[index]):
            raise ValueError(f"source record changed at row {index}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--split-role",
        required=True,
        choices=("validation_smoke", "test999_development"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def generate_one(bot: HuatuoChatbot, question: str, image: str) -> tuple[str, int, bool]:
    text = bot.input_moderation(question)
    text = bot.insert_image_placeholder(text, 1)
    conversation = bot.get_conv_without_history(text)
    input_ids = bot.preprocess(conversation, return_tensors="pt").unsqueeze(0).to(bot.device)
    image_tensors = torch.stack(bot.get_image_tensors([image])).to(
        dtype=torch.bfloat16, device=bot.device
    )
    with torch.inference_mode():
        output_ids = bot.model.generate(
            input_ids,
            images=image_tensors,
            use_cache=True,
            **bot.gen_kwargs,
        )
    sequence = output_ids[0]
    if sequence.shape[0] >= input_ids.shape[1] and torch.equal(
        sequence[: input_ids.shape[1]], input_ids[0]
    ):
        sequence = sequence[input_ids.shape[1] :]
    generated = sequence.detach().cpu().tolist()
    completion = bot.tokenizer.decode(generated, skip_special_tokens=True).strip()
    ended_with_eos = bool(generated and generated[-1] == bot.tokenizer.eos_token_id)
    return completion, len(generated), ended_with_eos


def main() -> None:
    args = parse_args()
    if args.max_new_tokens != 1024:
        raise ValueError("the approved diagnostic contract requires max_new_tokens=1024")
    records = json.loads(args.data.read_text(encoding="utf-8"))
    expected = 999 if args.split_role == "test999_development" else 385
    if len(records) != expected:
        raise ValueError(f"expected {expected} records, got {len(records)}")
    if args.limit is not None:
        if args.split_role == "test999_development":
            raise ValueError("test999 development run must cover all 999 records")
        records = records[: args.limit]

    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    metrics_path = args.output_dir / "metrics.json"
    config_path = args.output_dir / "run_config.json"
    if metrics_path.exists():
        raise FileExistsError(metrics_path)

    config = {
        "schema_version": 1,
        "status": "running",
        "backend": "huatuogpt_vision_native",
        "split_role": args.split_role,
        "test_accessed": args.split_role == "test999_development",
        "diagnostic_only": args.split_role == "test999_development",
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "data_path": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "selected_count": len(records),
        "prompt_template": "<image>\n" + QUESTION_TEMPLATE,
        "dtype": "bfloat16",
        "quantization": "none",
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": 1,
        "repetition_penalty": 1.0,
    }
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if {**old, "status": "running"} != config:
            raise ValueError("resume configuration differs")
    else:
        write_json(config_path, config)
    rows = load_existing(predictions_path, records)

    bot = HuatuoChatbot(str(args.model))
    bot.debug = False
    bot.gen_kwargs = {
        "do_sample": False,
        "max_new_tokens": 1024,
        "min_new_tokens": 1,
        "repetition_penalty": 1.0,
        "eos_token_id": bot.tokenizer.eos_token_id,
        "pad_token_id": bot.tokenizer.pad_token_id or bot.tokenizer.eos_token_id,
    }

    for index in range(len(rows), len(records)):
        record = records[index]
        completion, token_count, ended_with_eos = generate_one(
            bot,
            QUESTION_TEMPLATE.format(question=record["problem"]),
            record["image"],
        )
        wrapped = [[{"role": "assistant", "content": completion}]]
        row = {
            "index": index,
            "source_record_sha256": record_sha256(record),
            "image": record["image"],
            "problem": record["problem"],
            "solution": record["solution"],
            "completion": completion,
            "generated_token_count": token_count,
            "ended_with_eos": ended_with_eos,
            "reached_generation_cap": token_count >= 1024,
            "predicted_choice": choice_letter(completion),
            "target_choice": choice_letter(record["solution"]),
            "accuracy_reward": accuracy_reward(wrapped, [record["solution"]])[0],
            "format_reward": format_reward(wrapped)[0],
            "test999_development_diagnostic": args.split_role
            == "test999_development",
        }
        with predictions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        rows.append(row)
        print(
            json.dumps(
                {
                    "completed": index + 1,
                    "total": len(records),
                    "correct": sum(item["accuracy_reward"] for item in rows),
                }
            ),
            flush=True,
        )

    token_counts = [row["generated_token_count"] for row in rows]
    correct = int(sum(row["accuracy_reward"] for row in rows))
    metrics = {
        **config,
        "status": "completed",
        "eligible_for_untouched_final_stage3_claim": (
            False if args.split_role == "test999_development" else None
        ),
        "count": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "format_correct": int(sum(row["format_reward"] for row in rows)),
        "choice_extracted": sum(row["predicted_choice"] is not None for row in rows),
        "empty_completion_count": sum(not row["completion"] for row in rows),
        "mean_generated_tokens": statistics.mean(token_counts),
        "median_generated_tokens": statistics.median(token_counts),
        "maximum_generated_tokens": max(token_counts),
        "generation_cap_hit_count": sum(row["reached_generation_cap"] for row in rows),
        "predictions_sha256": sha256_file(predictions_path),
    }
    write_json(metrics_path, metrics)
    write_json(config_path, {**config, "status": "completed"})
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
