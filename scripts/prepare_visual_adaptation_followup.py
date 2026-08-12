#!/usr/bin/env python3
"""Prepare matched high-throughput L/A SFT follow-up configurations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from prepare_visual_adaptation_sft_pilot import target_modules


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("l", "a"), required=True)
    parser.add_argument("--seed", choices=(42, 43), type=int, required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--deepspeed", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=1500)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--micro-batch", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    data_name = f"pathvlm_sft_stratified_n{args.count:04d}"
    data_path = args.dataset_dir / f"{data_name}.json"
    for path in (args.model / "config.json", args.dataset_dir / "dataset_info.json", data_path, args.deepspeed):
        if not path.is_file():
            raise FileNotFoundError(path)

    # ceil(1500 / (8 * 12)) = 16 steps/epoch; save every half epoch.
    steps_per_epoch = (args.count + 8 * args.micro_batch - 1) // (8 * args.micro_batch)
    save_steps = max(1, steps_per_epoch // 2)
    output = args.run_dir / "output"
    config = {
        "model_name_or_path": str(args.model.resolve()),
        "trust_remote_code": True,
        "image_max_pixels": 65536,
        "video_max_pixels": 8192,
        "flash_attn": "sdpa",
        "use_cache": False,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": args.arm == "l",
        "freeze_language_model": False,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "dataset": data_name,
        "template": "qwen2_vl",
        "cutoff_len": 1024,
        "max_samples": args.count,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        "output_dir": str(output.resolve()),
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": save_steps,
        "save_total_limit": 8,
        "save_only_model": True,
        "plot_loss": True,
        "overwrite_output_dir": False,
        "report_to": "none",
        "per_device_train_batch_size": args.micro_batch,
        "gradient_accumulation_steps": 1,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "bf16": True,
        "gradient_checkpointing": False,
        "disable_gradient_checkpointing": True,
        "optim": "adamw_torch_fused",
        "seed": args.seed,
        "data_seed": args.seed,
        "disable_shuffling": False,
        "ddp_timeout": 180000000,
        "ddp_find_unused_parameters": False,
        "num_train_epochs": args.epochs,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target": target_modules(args.arm),
    }
    if args.arm == "a":
        config["additional_target"] = ["visual.merger"]
    if args.max_steps is not None:
        if args.max_steps < 2:
            raise ValueError("max-steps must be at least two")
        config["max_steps"] = args.max_steps
        config["save_strategy"] = "no"
    args.run_dir.mkdir(parents=True)
    config_path = args.run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "frozen_before_training",
        "role": "L_vs_A_representative_followup",
        "arm": args.arm,
        "seed": args.seed,
        "count": args.count,
        "epochs": args.epochs,
        "world_size": 8,
        "micro_batch_per_gpu": args.micro_batch,
        "global_batch": 8 * args.micro_batch,
        "steps_per_epoch": steps_per_epoch,
        "save_steps": save_steps,
        "checkpoint_spacing_epochs_approx": save_steps / steps_per_epoch,
        "max_steps_override": args.max_steps,
        "model_config_sha256": sha256(args.model / "config.json"),
        "dataset_sha256": sha256(data_path),
        "dataset_manifest_sha256": sha256(args.dataset_dir / "manifest.json"),
        "config_sha256": sha256(config_path),
        "vision_policy": "frozen",
        "projector_policy": "frozen" if args.arm == "l" else "full_train_visual.merger",
        "language_policy": "LoRA_r16_exact_28_layers",
        "selection_splits": ["PathMMU validation385", "PathVQA validation512", "MMMU nonmedical dev116"],
        "forbidden_for_selection": ["PathMMU test999", "PathVQA test", "OmniMedVQA test"],
    }
    (args.run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
