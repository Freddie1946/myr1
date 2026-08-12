#!/usr/bin/env python3
"""Evaluate Qwen2.5-VL base/checkpoint adapters on a frozen MMMU dev panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration


IMAGE_MARKER = re.compile(r"<image\s+(\d+)>", re.I)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_sha256(record: dict) -> str:
    value = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode()).hexdigest()


def extract_choice(completion: str, valid_letters: str) -> str | None:
    tagged = re.findall(
        r"<answer\b[^>]*>\s*(.*?)(?:</answer(?:\s*>|\s*$)|$)", completion, re.I | re.S
    )
    candidate = tagged[-1] if tagged else completion
    patterns = (
        rf"\\boxed\s*\{{\s*([{valid_letters}])\s*\}}",
        rf"^\s*(?:\*\*|__)?\(?\s*([{valid_letters}])(?:\s*[\)).,:;\-]|\s*$)",
        rf"\b(?:final\s+answer|answer|option|choice)\s*(?:is|:|=)\s*([{valid_letters}])\b",
        rf"(?:^|\n)\s*([{valid_letters}])\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, candidate, re.I)
        if match:
            return match.group(1).upper()
    return None


def content_for_record(record: dict) -> tuple[list[dict], list[Image.Image]]:
    question = record["question"]
    images = [Image.open(path).convert("RGB") for path in record["images"]]
    content: list[dict] = []
    cursor = 0
    seen: set[int] = set()
    for match in IMAGE_MARKER.finditer(question):
        if match.start() > cursor:
            content.append({"type": "text", "text": question[cursor : match.start()]})
        image_index = int(match.group(1)) - 1
        if image_index < 0 or image_index >= len(images):
            raise ValueError(f"invalid image marker in {record['id']}: {match.group(0)}")
        content.append({"type": "image"})
        seen.add(image_index)
        cursor = match.end()
    if cursor < len(question):
        content.append({"type": "text", "text": question[cursor:]})
    for image_index in range(len(images)):
        if image_index not in seen:
            content.insert(0, {"type": "image"})
    option_text = "\nOptions:\n" + "\n".join(
        f"{chr(65 + index)}) {value}" for index, value in enumerate(record["options"])
    )
    instruction = (
        option_text
        + "\nGive concise image-grounded reasoning inside <think>...</think>. Then output exactly "
        + "one option letter inside <answer>...</answer>."
    )
    content.append({"type": "text", "text": instruction})
    return content, images


def load_existing(path: Path, records: list[dict]) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    for index, row in enumerate(rows):
        if row["index"] != index or row["source_record_sha256"] != record_sha256(records[index]):
            raise ValueError(f"resume mismatch at record {index}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    records = json.loads(args.panel.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 116:
        raise ValueError("expected frozen 116-record MMMU retention panel")
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    metrics_path = args.output_dir / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError(metrics_path)
    rows = load_existing(predictions_path, records)

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    processor.image_processor.max_pixels = 65536
    processor.image_processor.min_pixels = 3136
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if isinstance(getattr(config, "text_config", None), dict):
        delattr(config, "text_config")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        config=config,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda")
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True, is_trainable=False)
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    eos = model.generation_config.eos_token_id
    stop_ids = {eos} if isinstance(eos, int) else set(eos or [])

    for start in range(len(rows), len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        prompts: list[str] = []
        all_images: list[Image.Image] = []
        for record in batch:
            content, images = content_for_record(record)
            prompts.append(processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
            ))
            all_images.extend(images)
        inputs = processor(text=prompts, images=all_images, return_tensors="pt", padding=True)
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True
            )
        completion_ids = generated[:, inputs["input_ids"].shape[1] :]
        for offset, record in enumerate(batch):
            token_ids = [int(value) for value in completion_ids[offset].tolist()]
            token_count = len(token_ids)
            for token_index, token_id in enumerate(token_ids):
                if token_id in stop_ids:
                    token_count = token_index + 1
                    break
            token_ids = token_ids[:token_count]
            completion = processor.decode(token_ids, skip_special_tokens=True).strip()
            valid = "ABCDEFGHI"[: len(record["options"])]
            predicted = extract_choice(completion, valid)
            row = {
                "index": start + offset,
                "source_record_sha256": record_sha256(record),
                "id": record["id"],
                "subject": record["subject"],
                "target_choice": record["answer"],
                "predicted_choice": predicted,
                "correct": predicted == record["answer"],
                "completion": completion,
                "generated_token_count": token_count,
                "ended_with_eos": bool(token_ids and token_ids[-1] in stop_ids),
                "reached_generation_cap": token_count >= args.max_new_tokens,
            }
            with predictions_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            rows.append(row)

    lengths = [row["generated_token_count"] for row in rows]
    metrics = {
        "schema_version": 1,
        "status": "completed",
        "formal_result": False,
        "split_role": "official_mmmu_nonmedical_dev_retention_sanity_check",
        "test_accessed": False,
        "count": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "choice_extracted": sum(row["predicted_choice"] is not None for row in rows),
        "generation_cap_hit_count": sum(row["reached_generation_cap"] for row in rows),
        "mean_generated_tokens": statistics.fmean(lengths),
        "median_generated_tokens": statistics.median(lengths),
        "maximum_generated_tokens": max(lengths),
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "adapter_path": str(args.adapter.resolve()) if args.adapter else None,
        "adapter_config_sha256": sha256_file(args.adapter / "adapter_config.json") if args.adapter else None,
        "adapter_model_sha256": sha256_file(args.adapter / "adapter_model.safetensors") if args.adapter else None,
        "panel": str(args.panel.resolve()),
        "panel_sha256": sha256_file(args.panel),
        "predictions_sha256": sha256_file(predictions_path),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
