#!/usr/bin/env python3
"""Resumable external VQA through HuatuoGPT-Vision's native interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from cli import HuatuoChatbot

from external_vqa_contract import (
    PATHVQA_PROMPT,
    OMNIMEDVQA_PROMPT,
    prompt_for_record,
    score_record,
    sha256_file,
)
from external_vqa_native_utils import (
    append_row,
    load_existing,
    result_row,
    summarize,
    write_json,
)


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
            input_ids, images=image_tensors, use_cache=True, **bot.gen_kwargs
        )
    sequence = output_ids[0]
    if sequence.shape[0] >= input_ids.shape[1] and torch.equal(
        sequence[: input_ids.shape[1]], input_ids[0]
    ):
        sequence = sequence[input_ids.shape[1] :]
    generated = sequence.detach().cpu().tolist()
    completion = bot.tokenizer.decode(generated, skip_special_tokens=True).strip()
    ended = bool(generated and generated[-1] == bot.tokenizer.eos_token_id)
    return completion, len(generated), ended


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("pathvqa", "omnimedvqa"))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--split-role", required=True, choices=("adapter_smoke", "external_test")
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens != 64:
        raise ValueError("frozen contract requires max_new_tokens=64")
    records = json.loads(args.data.read_text(encoding="utf-8"))
    expected = 6719 if args.task == "pathvqa" else 8518
    if len(records) != expected:
        raise ValueError(f"expected {expected}, got {len(records)}")
    if args.limit is not None:
        if args.split_role != "adapter_smoke" or not 1 <= args.limit <= len(records):
            raise ValueError("invalid smoke limit")
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
        "task": args.task,
        "split_role": args.split_role,
        "backend": "huatuogpt_vision_native",
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "data_path": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "selected_count": len(records),
        "prompt_template": PATHVQA_PROMPT if args.task == "pathvqa" else OMNIMEDVQA_PROMPT,
        "dtype": "bfloat16",
        "quantization": "none",
        "do_sample": False,
        "max_new_tokens": 64,
        "min_new_tokens": 1,
        "repetition_penalty": 1.0,
    }
    if config_path.exists():
        if {**json.loads(config_path.read_text()), "status": "running"} != config:
            raise ValueError("resume configuration differs")
    else:
        write_json(config_path, config)
    rows = load_existing(predictions_path, records, args.task)

    bot = HuatuoChatbot(str(args.model))
    bot.debug = False
    bot.gen_kwargs = {
        "do_sample": False,
        "max_new_tokens": 64,
        "min_new_tokens": 1,
        "repetition_penalty": 1.0,
        "eos_token_id": bot.tokenizer.eos_token_id,
        "pad_token_id": bot.tokenizer.pad_token_id or bot.tokenizer.eos_token_id,
    }
    for index in range(len(rows), len(records)):
        record = records[index]
        completion, token_count, ended = generate_one(
            bot, prompt_for_record(args.task, record), record["image"]
        )
        row = result_row(
            index=index,
            task=args.task,
            record=record,
            completion=completion,
            token_count=token_count,
            ended_with_eos=ended,
            score=score_record(args.task, completion, record),
            max_new_tokens=64,
        )
        append_row(predictions_path, row)
        rows.append(row)
        if (index + 1) % 100 == 0 or index + 1 == len(records):
            key = (
                "contract_aligned_exact_match"
                if args.task == "pathvqa"
                else "contract_aligned_correct"
            )
            print(
                json.dumps(
                    {
                        "completed": index + 1,
                        "total": len(records),
                        "correct": sum(item[key] for item in rows),
                    }
                ),
                flush=True,
            )
    metrics = summarize(
        rows, task=args.task, common=config, predictions_path=predictions_path
    )
    write_json(metrics_path, metrics)
    write_json(config_path, {**config, "status": "completed"})
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
