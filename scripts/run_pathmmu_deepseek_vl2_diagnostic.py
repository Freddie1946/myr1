#!/usr/bin/env python3
"""Run resumable PathMMU diagnostics through the official DeepSeek-VL2 interface."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor
from deepseek_vl2.utils.io import load_pil_images
from pathmmu_rewards import accuracy_reward, choice_letter, format_reward


QUESTION_TEMPLATE = (
    "<image>\n{question}\n"
    "Give concise image-grounded reasoning inside <think>...</think>. Then output exactly one "
    "option letter (A, B, C, or D) inside <answer>...</answer>."
)
LETTER_ONLY_TEMPLATE = (
    "<image>\n{question}\n"
    "Answer with exactly one uppercase option letter: A, B, C, or D. Do not output any other text."
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
    parser.add_argument(
        "--prompt-contract", choices=("reasoning_v1", "letter_only_v2"),
        default="reasoning_v1",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--device-map-auto", action="store_true",
        help="shard the unquantized model across all visible GPUs with a 48 GiB per-GPU cap",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_tokens = 1024 if args.prompt_contract == "reasoning_v1" else 32
    if args.max_new_tokens != expected_tokens:
        raise ValueError(
            f"{args.prompt_contract} requires max_new_tokens={expected_tokens}"
        )
    prompt_template = (
        QUESTION_TEMPLATE if args.prompt_contract == "reasoning_v1" else LETTER_ONLY_TEMPLATE
    )
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
        "backend": "deepseek_vl2_official",
        "split_role": args.split_role,
        "test_accessed": args.split_role == "test999_development",
        "diagnostic_only": args.split_role == "test999_development",
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "data_path": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "source_count": expected,
        "selected_count": len(records),
        "prompt_contract": args.prompt_contract,
        "prompt_template": prompt_template,
        "dtype": "bfloat16",
        "quantization": "none",
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "device_map": "auto" if args.device_map_auto else "single_gpu",
        "per_visible_gpu_max_memory": "48GiB" if args.device_map_auto else None,
    }
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if {**old, "status": "running"} != config:
            raise ValueError("resume configuration differs")
    else:
        write_json(config_path, config)
    rows = load_existing(predictions_path, records)

    processor = DeepseekVLV2Processor.from_pretrained(
        args.model, local_files_only=True
    )
    tokenizer = processor.tokenizer
    load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if args.device_map_auto:
        load_kwargs.update({
            "device_map": "auto",
            "max_memory": {
                index: "48GiB" for index in range(torch.cuda.device_count())
            },
        })
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).eval()
    else:
        model = (
            AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
            .to(torch.bfloat16)
            .cuda()
            .eval()
        )

    for index in range(len(rows), len(records)):
        record = records[index]
        conversation = [
            {
                "role": "<|User|>",
                "content": prompt_template.format(question=record["problem"]),
                "images": [record["image"]],
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
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        generated_ids = output_ids[0].detach().cpu().tolist()
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        token_count = len(generated_ids)
        wrapped = [[{"role": "assistant", "content": completion}]]
        row = {
            "index": index,
            "source_record_sha256": record_sha256(record),
            "image": record["image"],
            "problem": record["problem"],
            "solution": record["solution"],
            "completion": completion,
            "generated_token_count": token_count,
            "ended_with_eos": bool(
                generated_ids and generated_ids[-1] == tokenizer.eos_token_id
            ),
            "reached_generation_cap": token_count >= args.max_new_tokens,
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
