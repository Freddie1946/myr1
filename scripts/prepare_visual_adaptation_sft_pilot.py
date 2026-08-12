#!/usr/bin/env python3
"""Create frozen LLaMA-Factory configs for matched C0/L/A/B2/B4 SFT pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


ARMS = ("c0", "l", "a", "b2", "b4")
LANGUAGE_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
VISION_SUFFIXES = (
    "attn.qkv",
    "attn.proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_modules(arm: str) -> list[str]:
    modules = [
        f"model.layers.{layer}.{suffix}"
        for layer in range(28)
        for suffix in LANGUAGE_SUFFIXES
    ]
    if arm in {"b2", "b4"}:
        first = 30 if arm == "b2" else 28
        modules.extend(
            f"visual.blocks.{block}.{suffix}"
            for block in range(first, 32)
            for suffix in VISION_SUFFIXES
        )
    return modules


def build_config(args: argparse.Namespace) -> dict:
    smoke = args.mode == "smoke"
    throughput_smoke = args.mode == "throughput_smoke"
    full = args.arm == "c0"
    config = {
        "model_name_or_path": str(args.model.resolve()),
        "trust_remote_code": True,
        "image_max_pixels": 65536,
        "video_max_pixels": 8192,
        "flash_attn": "sdpa",
        "use_cache": False,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full" if full else "lora",
        "freeze_vision_tower": True if args.arm in {"c0", "l", "a"} else False,
        "freeze_multi_modal_projector": True if args.arm in {"c0", "l"} else False,
        "freeze_language_model": False,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "dataset": "pathvlm_sft_smoke_n0008" if smoke else "pathvlm_sft_n0500",
        "template": "qwen2_vl",
        "cutoff_len": 512,
        "max_samples": 8 if smoke else 500,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        "output_dir": str(args.output_dir.resolve()),
        "logging_steps": 1,
        "save_strategy": "no" if throughput_smoke else args.save_strategy,
        "save_total_limit": 2,
        "save_only_model": True,
        "plot_loss": True,
        "overwrite_output_dir": False,
        "report_to": "none",
        "per_device_train_batch_size": args.full_micro_batch if full else args.lora_micro_batch,
        "gradient_accumulation_steps": 1,
        "learning_rate": args.full_learning_rate if full else args.lora_learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "bf16": True,
        "gradient_checkpointing": False,
        "disable_gradient_checkpointing": True,
        "optim": "adamw_torch_fused",
        "seed": 42,
        "data_seed": 42,
        "disable_shuffling": True,
        "ddp_timeout": 180000000,
        "ddp_find_unused_parameters": False,
    }
    if smoke:
        if config["save_strategy"] == "steps":
            config["save_steps"] = 2
        config["max_steps"] = 2
    elif throughput_smoke:
        config["max_steps"] = args.smoke_steps
    else:
        if config["save_strategy"] == "steps":
            config["save_steps"] = args.save_steps
        config["num_train_epochs"] = args.num_train_epochs
    if full:
        config["deepspeed"] = str(args.deepspeed.resolve())
    else:
        config.update({
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "lora_target": target_modules(args.arm),
        })
        if args.arm != "l":
            config["additional_target"] = ["visual.merger"]
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument(
        "--mode", choices=("smoke", "throughput_smoke", "formal"), required=True
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--deepspeed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lora-micro-batch", type=int, default=4)
    parser.add_argument("--full-micro-batch", type=int, default=1)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--smoke-steps", type=int, default=12)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--save-strategy", choices=("no", "steps"), default="steps")
    parser.add_argument("--save-steps", type=int, default=63)
    parser.add_argument("--full-learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--lora-learning-rate", type=float, default=5.0e-5)
    args = parser.parse_args()
    if args.world_size is None:
        args.world_size = 8 if args.arm == "c0" else 2
    if args.config.exists() or args.manifest.exists() or args.output_dir.exists():
        raise FileExistsError("refusing to overwrite config, manifest, or output directory")
    if args.lora_micro_batch < 1 or args.full_micro_batch < 1:
        raise ValueError("micro batches must be positive")
    if not 1 <= args.world_size <= 8:
        raise ValueError("world size must be in [1, 8]")
    if args.smoke_steps < 2:
        raise ValueError("smoke steps must be at least two")
    if args.num_train_epochs <= 0 or args.save_steps < 1:
        raise ValueError("epochs and save steps must be positive")
    if args.full_learning_rate <= 0 or args.lora_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    for required in (args.model / "config.json", args.dataset_dir / "dataset_info.json", args.deepspeed):
        if not required.exists():
            raise FileNotFoundError(required)
    config = build_config(args)
    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "frozen_before_training",
        "arm": args.arm,
        "mode": args.mode,
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "model_config_sha256": sha256(args.model / "config.json"),
        "dataset_info_sha256": sha256(args.dataset_dir / "dataset_info.json"),
        "dataset_file_sha256": sha256(
            args.dataset_dir / ("pathvlm_sft_smoke_n0008.json" if args.mode == "smoke" else "pathvlm_sft_n0500.json")
        ),
        "expected_world_size": args.world_size,
        "expected_global_batch_size": (
            args.world_size
            * (args.full_micro_batch if args.arm == "c0" else args.lora_micro_batch)
        ),
        "vision_encoder_policy": (
            "frozen" if args.arm in {"c0", "l", "a"} else f"LoRA_only_blocks_{30 if args.arm == 'b2' else 28}_through_31"
        ),
        "projector_policy": "frozen" if args.arm in {"c0", "l"} else "full_train_visual.merger",
        "language_policy": "full_train" if args.arm == "c0" else "LoRA_r16_exact_28_layers",
        "lora_target_modules": [] if args.arm == "c0" else target_modules(args.arm),
        "selection_data_forbidden": ["PathVQA test", "OmniMedVQA test"],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "prepared", "arm": args.arm, "mode": args.mode, "config": str(args.config)}))


if __name__ == "__main__":
    main()
