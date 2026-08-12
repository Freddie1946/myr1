#!/usr/bin/env python3
"""Sample eight rule-reward rollouts per fixed RL calibration prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

from pathmmu_rewards import accuracy_reward, choice_letter, format_reward
from run_pathmmu_qwen_diagnostic import QUESTION_TEMPLATE, eos_ids, record_sha256, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=424242)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index")
    if args.rollouts != 8 or args.max_new_tokens != 192 or args.temperature != 1.0:
        raise ValueError("frozen readiness contract requires 8 rollouts, 192 tokens, temperature 1.0")
    records = json.loads(args.panel.read_text(encoding="utf-8"))
    if len(records) != 300:
        raise ValueError("expected fixed 300-prompt panel")
    args.output_dir.mkdir(parents=True)

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    processor.image_processor.max_pixels = 65536
    processor.image_processor.min_pixels = 3136
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if isinstance(getattr(config, "text_config", None), dict):
        delattr(config, "text_config")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, config=config, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to("cuda")
    model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True, is_trainable=False)
    model.eval()
    stop_ids = eos_ids(model.generation_config.eos_token_id)
    rows = []
    output = args.output_dir / "rollouts.jsonl"
    for index in range(args.shard_index, len(records), args.shard_count):
        record = records[index]
        with Image.open(record["image"]) as source:
            image = source.convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": QUESTION_TEMPLATE.format(question=record["problem"])},
        ]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        torch.manual_seed(args.seed + index)
        torch.cuda.manual_seed_all(args.seed + index)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, do_sample=True, temperature=args.temperature, top_p=1.0,
                num_return_sequences=args.rollouts, max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )
        generated = generated[:, inputs["input_ids"].shape[1]:]
        completions, lengths, cap_hits = [], [], []
        for sequence in generated.tolist():
            length = len(sequence)
            for position, token_id in enumerate(sequence):
                if int(token_id) in stop_ids:
                    length = position + 1
                    break
            token_ids = [int(value) for value in sequence[:length]]
            completions.append(processor.decode(token_ids, skip_special_tokens=True))
            lengths.append(length)
            cap_hits.append(length >= args.max_new_tokens and not (token_ids and token_ids[-1] in stop_ids))
        wrapped = [[{"role": "assistant", "content": completion}] for completion in completions]
        accuracy = accuracy_reward(wrapped, [record["solution"]] * args.rollouts)
        formatting = format_reward(wrapped)
        row = {
            "index": index,
            "source_record_sha256": record_sha256(record),
            "target_choice": choice_letter(record["solution"]),
            "completions": completions,
            "predicted_choices": [choice_letter(value) for value in completions],
            "accuracy_rewards": accuracy,
            "format_rewards": formatting,
            "generated_token_counts": lengths,
            "generation_cap_hits": cap_hits,
            "sampling_seed": args.seed + index,
        }
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        rows.append(row)
        print(json.dumps({"shard": args.shard_index, "index": index, "correct": sum(accuracy)}), flush=True)
    metrics = {
        "schema_version": 1,
        "status": "shard_completed",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "prompt_count": len(rows),
        "rollouts_per_prompt": args.rollouts,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": 1.0,
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "adapter_config_sha256": sha256_file(args.adapter / "adapter_config.json"),
        "adapter_model_sha256": sha256_file(args.adapter / "adapter_model.safetensors"),
        "panel_sha256": sha256_file(args.panel),
        "rollouts_sha256": sha256_file(output),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
