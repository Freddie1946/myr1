#!/usr/bin/env python3
"""Run resumable Transformers-native VLM PathMMU development diagnostics.

This runner is intentionally separate from the historical final/validation runners. It records
PathMMU test999 access truthfully and labels every test result as Stage-3 development diagnostic
evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

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


def eos_ids(value: int | list[int] | tuple[int, ...] | None) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(item) for item in value}


def load_existing(path: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for expected_index, line in enumerate(handle):
            row = json.loads(line)
            if row.get("index") != expected_index:
                raise ValueError(f"non-contiguous existing prediction at row {expected_index}")
            if expected_index >= len(records):
                raise ValueError("existing prediction count exceeds selected input count")
            if row.get("source_record_sha256") != record_sha256(records[expected_index]):
                raise ValueError(f"source record changed at row {expected_index}")
            rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    lengths = [int(row["generated_token_count"]) for row in rows]
    ordered = sorted(lengths)
    return {
        "schema_version": 1,
        "status": "completed",
        "diagnostic_only": args.split_role == "test999_development",
        "eligible_for_untouched_final_stage3_claim": False
        if args.split_role == "test999_development"
        else None,
        "split_role": args.split_role,
        "test_accessed": args.split_role == "test999_development",
        "count": len(rows),
        "correct": sum(int(row["accuracy_reward"]) for row in rows),
        "accuracy": sum(float(row["accuracy_reward"]) for row in rows) / len(rows),
        "format_correct": sum(int(row["format_reward"]) for row in rows),
        "choice_extracted": sum(row["predicted_choice"] is not None for row in rows),
        "empty_completion_count": sum(not str(row["completion"]) for row in rows),
        "mean_generated_tokens": statistics.fmean(lengths),
        "median_generated_tokens": statistics.median(lengths),
        "maximum_generated_tokens": max(lengths),
        "eos_terminated_count": sum(bool(row["ended_with_eos"]) for row in rows),
        "generation_cap_hit_count": sum(bool(row["reached_generation_cap"]) for row in rows),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "dtype": "bfloat16",
        "quantization": "none",
        "backend": args.backend,
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "data_path": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "predictions_sha256": sha256_file(args.output_dir / "predictions.jsonl"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--backend",
        required=True,
        choices=("gemma3", "mllama", "qwen2_vl", "qwen2_5_vl", "qwen3_5"),
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--split-role",
        required=True,
        choices=("validation_smoke", "test999_development"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens != 1024:
        raise ValueError("the approved diagnostic contract requires max_new_tokens=1024")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    records = json.loads(args.data.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("input data must be a JSON list")
    expected_full_count = 999 if args.split_role == "test999_development" else 385
    if len(records) != expected_full_count:
        raise ValueError(
            f"{args.split_role} requires source count {expected_full_count}, got {len(records)}"
        )
    if args.limit is not None:
        if args.limit < 1 or args.limit > len(records):
            raise ValueError("invalid --limit")
        records = records[: args.limit]
    if args.split_role == "test999_development" and args.limit is not None:
        raise ValueError("test999 development run must cover all 999 records")

    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    metrics_path = args.output_dir / "metrics.json"
    run_config_path = args.output_dir / "run_config.json"
    if metrics_path.exists():
        raise FileExistsError(f"completed metrics already exist: {metrics_path}")

    run_config = {
        "schema_version": 1,
        "status": "running",
        "diagnostic_only": args.split_role == "test999_development",
        "split_role": args.split_role,
        "test_accessed": args.split_role == "test999_development",
        "source_count": expected_full_count,
        "selected_count": len(records),
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "backend": args.backend,
        "data_path": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "prompt_template": QUESTION_TEMPLATE,
        "dtype": "bfloat16",
        "quantization": "none",
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "resume": args.resume,
    }
    if run_config_path.exists():
        existing_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        comparable = dict(existing_config)
        comparable["resume"] = args.resume
        if comparable != run_config:
            raise ValueError("resume configuration differs from the existing run")
    else:
        write_json(run_config_path, run_config)

    rows = load_existing(predictions_path, records)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = 65536
        processor.image_processor.min_pixels = 3136
    if args.backend == "gemma3":
        # Keep the Qwen-only environment usable even when its pinned Transformers build predates
        # Gemma 3. The MedGemma environment imports this class only when that backend is selected.
        from transformers import Gemma3ForConditionalGeneration

        model_class = Gemma3ForConditionalGeneration
    elif args.backend == "mllama":
        from transformers import MllamaForConditionalGeneration

        model_class = MllamaForConditionalGeneration
    elif args.backend == "qwen3_5":
        from transformers import Qwen3_5ForConditionalGeneration

        model_class = Qwen3_5ForConditionalGeneration
    else:
        model_class = {
            "qwen2_vl": Qwen2VLForConditionalGeneration,
            "qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
        }[args.backend]
    load_kwargs = {
        "local_files_only": True,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }
    if args.backend == "qwen2_5_vl":
        # Some checkpoints saved by Transformers >=4.56 persist a nested
        # ``text_config``.  Transformers 4.49 treats that unknown field as a
        # raw dict, while this generation class expects either a
        # PretrainedConfig or the legacy flat top-level configuration.  The
        # top-level fields in these checkpoints are complete, so discard only
        # the incompatible duplicate in memory; never rewrite config.json.
        compatible_config = AutoConfig.from_pretrained(
            args.model, local_files_only=True
        )
        if isinstance(getattr(compatible_config, "text_config", None), dict):
            delattr(compatible_config, "text_config")
        load_kwargs["config"] = compatible_config
    if args.backend == "mllama":
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = {
            index: "76GiB" for index in range(torch.cuda.device_count())
        }
        model = model_class.from_pretrained(args.model, **load_kwargs)
    else:
        model = model_class.from_pretrained(args.model, **load_kwargs).to("cuda")
    model.eval()
    # Some medical checkpoints persist sampling-only defaults. They are irrelevant under greedy
    # decoding, but clear them explicitly so the frozen execution log is unambiguous.
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    stop_ids = eos_ids(model.generation_config.eos_token_id)

    for batch_start in range(len(rows), len(records), args.batch_size):
        batch_records = records[batch_start : batch_start + args.batch_size]
        images = []
        prompts = []
        for record in batch_records:
            with Image.open(record["image"]) as source_image:
                images.append(source_image.convert("RGB"))
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(question=record["problem"]),
                        },
                    ],
                }
            ]
            prompts.append(
                processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        processor_images = [[image] for image in images] if args.backend == "gemma3" else images
        inputs = processor(
            text=prompts, images=processor_images, return_tensors="pt", padding=True
        )
        input_device = next(model.parameters()).device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        batch_completion_ids = generated[:, inputs["input_ids"].shape[1] :]
        for offset, record in enumerate(batch_records):
            token_ids = [int(value) for value in batch_completion_ids[offset].tolist()]
            generated_token_count = len(token_ids)
            for token_index, token_id in enumerate(token_ids):
                if token_id in stop_ids:
                    generated_token_count = token_index + 1
                    break
            token_ids = token_ids[:generated_token_count]
            final_token_id = token_ids[-1] if token_ids else None
            completion = processor.decode(token_ids, skip_special_tokens=True)
            wrapped = [[{"role": "assistant", "content": completion}]]
            index = batch_start + offset
            row = {
                "index": index,
                "source_record_sha256": record_sha256(record),
                "image": record["image"],
                "problem": record["problem"],
                "solution": record["solution"],
                "completion": completion,
                "generated_token_count": generated_token_count,
                "ended_with_eos": final_token_id in stop_ids,
                "reached_generation_cap": generated_token_count >= args.max_new_tokens,
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
        completed = batch_start + len(batch_records)
        if completed % 25 < args.batch_size or completed == len(records):
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": len(records),
                        "correct": sum(int(item["accuracy_reward"]) for item in rows),
                    }
                ),
                flush=True,
            )

    metrics = summarize(rows, args)
    write_json(metrics_path, metrics)
    run_config["status"] = "completed"
    write_json(run_config_path, run_config)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
