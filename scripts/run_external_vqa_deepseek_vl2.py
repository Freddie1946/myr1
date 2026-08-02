#!/usr/bin/env python3
"""Resumable PathVQA/OmniMedVQA inference through DeepSeek-VL2's official API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor
from deepseek_vl2.utils.io import load_pil_images
from external_vqa_contract import (
    OMNIMEDVQA_PROMPT,
    PATHVQA_PROMPT,
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


def generate_one(
    model: torch.nn.Module,
    processor: DeepseekVLV2Processor,
    *,
    prompt: str,
    image: str,
    max_new_tokens: int,
) -> tuple[str, int, bool]:
    tokenizer = processor.tokenizer
    conversation = [
        {
            "role": "<|User|>",
            "content": "<image>\n" + prompt,
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    pil_images = load_pil_images(conversation)
    prepared = processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True,
        system_prompt="",
    ).to(model.device)
    with torch.inference_mode():
        inputs_embeds = model.prepare_inputs_embeds(**prepared)
        output_ids = model.language.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepared.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    generated = output_ids[0].detach().cpu().tolist()
    completion = tokenizer.decode(generated, skip_special_tokens=True).strip()
    ended = bool(generated and generated[-1] == tokenizer.eos_token_id)
    return completion, len(generated), ended


def main() -> None:
    args = parse_args()
    if args.max_new_tokens != 64:
        raise ValueError("the frozen external VQA contract requires max_new_tokens=64")
    records = json.loads(args.data.read_text(encoding="utf-8"))
    expected = 6719 if args.task == "pathvqa" else 8518
    if not isinstance(records, list) or len(records) != expected:
        raise ValueError(f"expected {expected} {args.task} records, got {len(records)}")
    if args.limit is not None:
        if args.split_role != "adapter_smoke" or not 1 <= args.limit <= len(records):
            raise ValueError("only adapter_smoke permits a positive --limit")
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
        "backend": "deepseek_vl2_official",
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
    }
    if config_path.exists():
        if {**json.loads(config_path.read_text()), "status": "running"} != config:
            raise ValueError("resume configuration differs")
    else:
        write_json(config_path, config)
    rows = load_existing(predictions_path, records, args.task)

    processor = DeepseekVLV2Processor.from_pretrained(args.model, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        .to(torch.bfloat16)
        .cuda()
        .eval()
    )

    for index in range(len(rows), len(records)):
        record = records[index]
        completion, token_count, ended = generate_one(
            model,
            processor,
            prompt=prompt_for_record(args.task, record),
            image=record["image"],
            max_new_tokens=64,
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
            score_key = (
                "contract_aligned_exact_match"
                if args.task == "pathvqa"
                else "contract_aligned_correct"
            )
            print(
                json.dumps(
                    {
                        "completed": index + 1,
                        "total": len(records),
                        "correct": sum(item[score_key] for item in rows),
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

