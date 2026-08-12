#!/usr/bin/env python3
"""Format-neutral Yes/No next-token diagnostic for Qwen2.5-VL checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageStat
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

from external_vqa_contract import PATHVQA_PROMPT, normalize_short_answer, record_sha256, sha256_file


VERBALIZERS = {"yes": "Yes", "no": "No"}


def load_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("records")
    if not isinstance(value, list):
        raise ValueError("input must be a record list or an object containing records")
    rows = [row for row in value if normalize_short_answer(str(row.get("answer") or "")) in {"yes", "no"}]
    if not rows:
        raise ValueError("input contains no binary records")
    return rows


def binary_result(yes_logit: float, no_logit: float, target: str) -> dict[str, Any]:
    margin = yes_logit - no_logit
    probability_yes = 1.0 / (1.0 + math.exp(-max(-80.0, min(80.0, margin))))
    predicted = "yes" if margin >= 0 else "no"
    normalized_target = normalize_short_answer(target)
    return {
        "yes_logit": yes_logit,
        "no_logit": no_logit,
        "yes_minus_no_margin": margin,
        "binary_probability_yes": probability_yes,
        "forced_binary_answer": predicted,
        "forced_binary_correct": predicted == normalized_target,
        "target_margin": margin if normalized_target == "yes" else -margin,
    }


def cyclic_image_map(records: list[dict[str, Any]]) -> dict[str, str]:
    paths = sorted({str(row["image"]) for row in records})
    if len(paths) < 2:
        raise ValueError("cyclic mismatch requires at least two unique images")
    return {path: paths[(index + 1) % len(paths)] for index, path in enumerate(paths)}


def intervention_image(source_path: str, mode: str, mismatch: dict[str, str]) -> tuple[Image.Image, str]:
    input_path = mismatch[source_path] if mode == "cyclic_mismatch" else source_path
    with Image.open(input_path) as source:
        image = source.convert("RGB")
    if mode == "global_mean_blank":
        mean = tuple(round(value) for value in ImageStat.Stat(image).mean)
        image = Image.new("RGB", image.size, mean)
    return image, input_path


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--split-role", required=True, choices=("validation_diagnostic", "post_hoc_test_sensitivity"))
    parser.add_argument("--image-mode", choices=("original", "global_mean_blank", "cyclic_mismatch"), default="original")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    all_records = load_records(args.data)
    indexed_records = [
        (index, record)
        for index, record in enumerate(all_records)
        if index % args.shard_count == args.shard_index
    ]
    if not indexed_records:
        raise ValueError("selected shard contains no records")
    global_indices = [index for index, _ in indexed_records]
    records = [record for _, record in indexed_records]
    mismatch = cyclic_image_map(records) if args.image_mode == "cyclic_mismatch" else {}
    args.output_dir.mkdir(parents=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = 65536
        processor.image_processor.min_pixels = 3136
    token_ids = {
        answer: processor.tokenizer.encode(text, add_special_tokens=False)
        for answer, text in VERBALIZERS.items()
    }
    if any(len(ids) != 1 for ids in token_ids.values()):
        raise ValueError(f"verbalizers must each be one token: {token_ids}")
    compatible_config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if isinstance(getattr(compatible_config, "text_config", None), dict):
        delattr(compatible_config, "text_config")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, config=compatible_config, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to("cuda")
    if args.adapter is not None:
        if not (args.adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(args.adapter / "adapter_config.json")
        model = PeftModel.from_pretrained(
            model, args.adapter, local_files_only=True, is_trainable=False
        )
    model.eval()
    rows = []
    for batch_start in range(0, len(records), args.batch_size):
        batch = records[batch_start:batch_start + args.batch_size]
        images, prompts = [], []
        input_image_paths = []
        for record in batch:
            image, input_path = intervention_image(str(record["image"]), args.image_mode, mismatch)
            images.append(image)
            input_image_paths.append(input_path)
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": PATHVQA_PROMPT.format(question=record["question"])},
            ]}]
            prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs, use_cache=False).logits[:, -1, :].float()
        for offset, record in enumerate(batch):
            result = binary_result(
                float(logits[offset, token_ids["yes"][0]].item()),
                float(logits[offset, token_ids["no"][0]].item()),
                str(record["answer"]),
            )
            row = {
                "index": global_indices[batch_start + offset],
                "source_index": record.get("source_index", record.get("index")),
                "source_record_sha256": record_sha256(record),
                "image": record["image"],
                "input_image": input_image_paths[offset],
                "image_mode": args.image_mode,
                "question": record["question"],
                "target": normalize_short_answer(str(record["answer"])),
                **result,
            }
            with predictions_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            rows.append(row)
        if len(rows) % 100 < args.batch_size or len(rows) == len(records):
            print(json.dumps({"completed": len(rows), "total": len(records)}), flush=True)
    correct = sum(row["forced_binary_correct"] for row in rows)
    write_json(args.output_dir / "metrics.json", {
        "schema_version": 1,
        "status": "completed",
        "formal_result": False,
        "result_role": args.split_role,
        "selection_or_tuning_use_forbidden": args.split_role == "post_hoc_test_sensitivity",
        "model": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "adapter": str(args.adapter.resolve()) if args.adapter is not None else None,
        "adapter_config_sha256": (
            sha256_file(args.adapter / "adapter_config.json") if args.adapter is not None else None
        ),
        "adapter_model_sha256": (
            sha256_file(args.adapter / "adapter_model.safetensors") if args.adapter is not None else None
        ),
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "source_count": len(all_records),
        "selected_count": len(records),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "prompt_template": PATHVQA_PROMPT,
        "verbalizers": VERBALIZERS,
        "decision_rule": "compare next-token logits for the single-token strings Yes and No",
        "image_mode": args.image_mode,
        "count": len(rows),
        "coverage": 1.0,
        "correct": correct,
        "accuracy": correct / len(rows),
        "mean_absolute_margin": sum(abs(row["yes_minus_no_margin"]) for row in rows) / len(rows),
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": sha256_file(predictions_path),
    })


if __name__ == "__main__":
    main()
