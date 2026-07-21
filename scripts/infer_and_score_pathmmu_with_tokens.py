#!/usr/bin/env python3
"""Run deterministic PathMMU inference with exact generation-length provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from pathmmu_rewards import accuracy_reward, choice_letter, format_reward


QUESTION_TEMPLATE = (
    "{question} First output the thinking process in <think> </think> tags and then "
    "output the final answer in <answer> </answer> tags."
)


def eos_ids(value: int | list[int] | tuple[int, ...] | None) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(item) for item in value}


def summarize_rows(rows: list[dict], max_new_tokens: int) -> dict:
    lengths = [int(row["generated_token_count"]) for row in rows]
    if not rows or any(length < 1 or length > max_new_tokens for length in lengths):
        raise ValueError("invalid generated token count")
    ordered = sorted(lengths)
    return {
        "count": len(rows),
        "mean_accuracy_reward": sum(row["accuracy_reward"] for row in rows) / len(rows),
        "mean_format_reward": sum(row["format_reward"] for row in rows) / len(rows),
        "choice_extracted_count": sum(row["predicted_choice"] is not None for row in rows),
        "empty_completion_count": sum(not str(row["completion"]) for row in rows),
        "mean_generated_tokens": sum(lengths) / len(lengths),
        "median_generated_tokens": ordered[len(ordered) // 2],
        "maximum_generated_tokens": max(lengths),
        "eos_terminated_count": sum(bool(row["ended_with_eos"]) for row in rows),
        "generation_cap_hit_count": sum(bool(row["reached_generation_cap"]) for row in rows),
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "test_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-new-tokens", required=True, type=int)
    parser.add_argument("--chat-template-file", required=True, type=Path)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")

    records = json.loads(args.data.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = args.output_dir / "predictions.jsonl"
    summary_path = args.output_dir / "metrics.json"

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    chat_template_payload = json.loads(args.chat_template_file.read_text(encoding="utf-8"))
    chat_template = chat_template_payload.get("chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError(f"chat_template is missing from {args.chat_template_file}")
    processor.chat_template = chat_template
    chat_template_sha256 = hashlib.sha256(args.chat_template_file.read_bytes()).hexdigest()
    processor.image_processor.max_pixels = 65536
    processor.image_processor.min_pixels = 3136

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    stop_ids = eos_ids(model.generation_config.eos_token_id)

    rows = []
    for index, record in enumerate(records):
        with Image.open(record["image"]) as source_image:
            image = source_image.convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": QUESTION_TEMPLATE.format(question=record["problem"])},
        ]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        completion_ids = generated[:, inputs["input_ids"].shape[1]:]
        generated_token_count = int(completion_ids.shape[1])
        final_token_id = int(completion_ids[0, -1].item()) if generated_token_count else None
        completion = processor.batch_decode(completion_ids, skip_special_tokens=True)[0]
        wrapped = [[{"role": "assistant", "content": completion}]]
        row = {
            "index": index,
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
        }
        rows.append(row)
        with predictions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize_rows(rows, args.max_new_tokens)
    summary.update({
        "predictions_file": str(predictions_path),
        "chat_template_file": str(args.chat_template_file.resolve()),
        "chat_template_sha256": chat_template_sha256,
    })
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
