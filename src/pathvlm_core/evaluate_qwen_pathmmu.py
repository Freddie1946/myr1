"""Deterministic PathMMU evaluation for Qwen2.5-VL style models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from pathvlm_core.answer_parser import exact_answer_match, extract_answer_block

PROMPT_TEMPLATE = (
    "{Question}\n"
    "First output the thinking process in <think></think> tags. "
    "Then output the final answer in <answer></answer> tags."
)


def load_rows(path: str) -> list[dict]:
    data = json.load(open(path, encoding="utf-8"))
    return data if isinstance(data, list) else data.get("results", [])


def resolve_image(image_value: str, image_root: str | None) -> str:
    if os.path.isabs(image_value):
        return image_value
    return os.path.join(image_root or "", image_value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    rows = load_rows(args.data)
    if args.limit:
        rows = rows[: args.limit]

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    results = []
    correct = 0
    for index, row in enumerate(rows, 1):
        image_path = resolve_image(row["image"], args.image_root)
        messages = [{"role": "user", "content": [{"type": "image", "image": Image.open(image_path).convert("RGB")}, {"type": "text", "text": PROMPT_TEMPLATE.format(Question=row["problem"])}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=args.max_new_tokens, do_sample=False)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        is_correct = exact_answer_match(output_text, row["solution"])
        correct += int(is_correct)
        results.append({
            "index": index,
            "image": row["image"],
            "question": row["problem"],
            "ground_truth": row["solution"],
            "model_output": output_text,
            "predicted_answer": extract_answer_block(output_text),
            "correct": bool(is_correct),
        })
        print(f"[{index}/{len(rows)}] correct={is_correct}")

    output = {"accuracy": correct / len(rows) * 100 if rows else 0.0, "total": len(rows), "correct": correct, "results": results}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(output, open(args.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Accuracy: {output['accuracy']:.2f}% ({correct}/{len(rows)})")


if __name__ == "__main__":
    main()
