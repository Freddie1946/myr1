#!/usr/bin/env python3
"""Run one GPU shard of the PathMMU SFT/RL split pass@k audit."""

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
from run_pathmmu_qwen_diagnostic import QUESTION_TEMPLATE, eos_ids, sha256_file


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def selected_records(sft: list[dict], rl: list[dict], limit: int | None) -> list[dict]:
    rows = []
    for split, source in (("sft3000", sft), ("rl1000", rl)):
        if limit is not None:
            source = source[:limit]
        for record in source:
            rows.append({"split": split, **record})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--sft-data", required=True, type=Path)
    parser.add_argument("--rl-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--prompt-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index")
    if args.shard_count != 8 or args.rollouts != 8:
        raise ValueError("formal audit requires eight shards and eight rollouts")
    if args.max_new_tokens != 384 or args.temperature != 0.9:
        raise ValueError("formal audit sampling contract mismatch")
    if args.prompt_batch_size < 1:
        raise ValueError("prompt batch size must be positive")

    sft = json.loads(args.sft_data.read_text(encoding="utf-8"))
    rl = json.loads(args.rl_data.read_text(encoding="utf-8"))
    if len(sft) != 3000 or len(rl) != 1000:
        raise ValueError("expected the frozen SFT3000 and RL1000 splits")
    records = selected_records(sft, rl, args.limit_per_split)
    assigned = list(range(args.shard_index, len(records), args.shard_count))
    output = args.output_dir / "rollouts.jsonl"
    completed: set[int] = set()
    if args.output_dir.exists():
        if not args.resume:
            raise FileExistsError(args.output_dir)
        if output.is_file():
            for line in output.open(encoding="utf-8"):
                row = json.loads(line)
                completed.add(int(row["index"]))
        if not completed.issubset(set(assigned)):
            raise RuntimeError("resume output contains indices assigned to another shard")
    else:
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
    if args.adapter is not None:
        model = PeftModel.from_pretrained(
            model, args.adapter, local_files_only=True, is_trainable=False
        )
    model.eval()
    stop_ids = eos_ids(model.generation_config.eos_token_id)
    remaining = [index for index in assigned if index not in completed]
    with output.open("a", encoding="utf-8") as handle:
        for start in range(0, len(remaining), args.prompt_batch_size):
            indices = remaining[start:start + args.prompt_batch_size]
            texts, images = [], []
            try:
                for index in indices:
                    record = records[index]
                    source = Image.open(record["image"])
                    images.append(source.convert("RGB"))
                    source.close()
                    messages = [{"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(
                            question=record["problem"]
                        )},
                    ]}]
                    texts.append(processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    ))
                inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
                inputs = {key: value.to("cuda") for key, value in inputs.items()}
                batch_seed = args.seed + indices[0]
                torch.manual_seed(batch_seed)
                torch.cuda.manual_seed_all(batch_seed)
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs, do_sample=True, temperature=args.temperature,
                        top_p=1.0, top_k=0, typical_p=1.0, repetition_penalty=1.0,
                        num_return_sequences=args.rollouts,
                        max_new_tokens=args.max_new_tokens, use_cache=True,
                    )
                generated = generated[:, inputs["input_ids"].shape[1]:]
                if generated.shape[0] != len(indices) * args.rollouts:
                    raise RuntimeError("unexpected generation batch shape")
                for offset, index in enumerate(indices):
                    record = records[index]
                    sequences = generated[
                        offset * args.rollouts:(offset + 1) * args.rollouts
                    ].tolist()
                    completions, lengths, cap_hits = [], [], []
                    for sequence in sequences:
                        length = len(sequence)
                        terminated = False
                        for position, token_id in enumerate(sequence):
                            if int(token_id) in stop_ids:
                                length = position + 1
                                terminated = True
                                break
                        token_ids = [int(value) for value in sequence[:length]]
                        completions.append(processor.decode(token_ids, skip_special_tokens=True))
                        lengths.append(length)
                        cap_hits.append(length >= args.max_new_tokens and not terminated)
                    wrapped = [[{"role": "assistant", "content": value}]
                               for value in completions]
                    accuracy = accuracy_reward(wrapped, [record["solution"]] * args.rollouts)
                    formatting = format_reward(wrapped)
                    row = {
                        "index": index,
                        "split": record["split"],
                        "source_record_sha256": digest({
                            key: record[key] for key in ("image", "problem", "solution")
                        }),
                        "target_choice": choice_letter(record["solution"]),
                        "completions": completions,
                        "predicted_choices": [choice_letter(value) for value in completions],
                        "accuracy_rewards": accuracy,
                        "format_rewards": formatting,
                        "generated_token_counts": lengths,
                        "generation_cap_hits": cap_hits,
                        "sampling_seed": batch_seed,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed.add(index)
                    print(json.dumps({
                        "shard": args.shard_index, "index": index,
                        "split": record["split"], "correct": sum(accuracy),
                    }), flush=True)
            finally:
                for image in images:
                    image.close()

    if completed != set(assigned):
        raise RuntimeError("shard did not complete all assigned records")
    metrics = {
        "schema_version": 1, "status": "shard_completed",
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "prompt_count": len(assigned), "rollouts_per_prompt": args.rollouts,
        "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
        "top_p": 1.0, "top_k": 0, "prompt_batch_size": args.prompt_batch_size,
        "seed": args.seed, "limit_per_split": args.limit_per_split,
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "adapter_config_sha256": (
            sha256_file(args.adapter / "adapter_config.json") if args.adapter else None
        ),
        "adapter_model_sha256": (
            sha256_file(args.adapter / "adapter_model.safetensors") if args.adapter else None
        ),
        "sft_data_sha256": sha256_file(args.sft_data),
        "rl_data_sha256": sha256_file(args.rl_data),
        "rollouts_sha256": sha256_file(output),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
