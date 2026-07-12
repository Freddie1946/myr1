#!/usr/bin/env python3
"""Run deterministic PathMMU validation inference and offline reward scoring."""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    records = json.loads(args.data.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    summary_path = args.output_dir / "metrics.json"

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
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

    rows = []
    for index, record in enumerate(records):
        image = Image.open(record["image"]).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": QUESTION_TEMPLATE.format(question=record["problem"])},
                ],
            }
        ]
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
        completion_ids = generated[:, inputs["input_ids"].shape[1] :]
        completion = processor.batch_decode(completion_ids, skip_special_tokens=True)[0]
        wrapped = [[{"role": "assistant", "content": completion}]]
        accuracy = accuracy_reward(wrapped, [record["solution"]])[0]
        format_score = format_reward(wrapped)[0]
        rows.append(
            {
                "index": index,
                "image": record["image"],
                "problem": record["problem"],
                "solution": record["solution"],
                "completion": completion,
                "predicted_choice": choice_letter(completion),
                "target_choice": choice_letter(record["solution"]),
                "accuracy_reward": accuracy,
                "format_reward": format_score,
            }
        )
        with predictions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(rows[-1], ensure_ascii=False) + "\n")

    summary = {
        "count": len(rows),
        "mean_accuracy_reward": sum(row["accuracy_reward"] for row in rows) / len(rows),
        "mean_format_reward": sum(row["format_reward"] for row in rows) / len(rows),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "test_accessed": False,
        "predictions_file": str(predictions_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

