#!/usr/bin/env python3
"""Prepare the formal SFT3000 run from the frozen L-vs-A architecture decision."""

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
    parser.add_argument("--decision", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--micro-batch", type=int, default=12)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--projector-mechanism-isolation",
        action="store_true",
        help=(
            "prepare a matched A arm (train visual.merger) without changing the "
            "architecture decision, which must remain L"
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="bounded smoke override; omitting it prepares the full three-epoch run",
    )
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    decision = json.loads(args.decision.read_text(encoding="utf-8"))
    if decision.get("status") != "architecture_decision_completed":
        raise ValueError("architecture decision is not complete")
    selected_arm = decision.get("selected_architecture")
    if selected_arm not in {"l", "a"}:
        raise ValueError(f"invalid selected architecture: {selected_arm!r}")
    if args.projector_mechanism_isolation:
        if selected_arm != "l":
            raise ValueError("projector mechanism isolation requires the frozen main decision L")
        arm = "a"
        role = "formal_projector_mechanism_isolation_sft3000"
    else:
        arm = selected_arm
        role = "formal_selected_architecture_sft3000"
    if args.max_steps is not None and args.max_steps < 2:
        raise ValueError("max-steps must be at least two")
    if args.lora_rank < 1 or args.lora_alpha < 1:
        raise ValueError("LoRA rank and alpha must be positive")
    data_path = args.dataset_dir / "pathvlm_sft_n3000.json"
    for path in (args.model / "config.json", args.dataset_dir / "dataset_info.json", data_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    steps_per_epoch = (3000 + 8 * args.micro_batch - 1) // (8 * args.micro_batch)
    save_steps = steps_per_epoch // 2
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
        "freeze_multi_modal_projector": arm == "l",
        "freeze_language_model": False,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "dataset": "pathvlm_sft_n3000",
        "template": "qwen2_vl",
        "cutoff_len": 1024,
        "max_samples": 3000,
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
        "learning_rate": 5.0e-5,
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
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.05,
        "lora_target": target_modules(arm),
    }
    if arm == "a":
        config["additional_target"] = ["visual.merger"]
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
        config["save_strategy"] = "no"
    args.run_dir.mkdir(parents=True)
    config_path = args.run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "frozen_before_training",
        "role": role,
        "selected_architecture": selected_arm,
        "arm": arm,
        "main_architecture_unchanged": selected_arm,
        "projector_mechanism_isolation": args.projector_mechanism_isolation,
        "decision": str(args.decision.resolve()),
        "decision_sha256": sha256(args.decision),
        "seed": args.seed,
        "sample_count": 3000,
        "epochs": args.epochs,
        "world_size": 8,
        "micro_batch_per_gpu": args.micro_batch,
        "global_batch": 8 * args.micro_batch,
        "steps_per_epoch": steps_per_epoch,
        "save_steps": save_steps,
        "checkpoint_spacing_epochs": save_steps / steps_per_epoch,
        "max_steps_override": args.max_steps,
        "cutoff_len": 1024,
        "dataset_sha256": sha256(data_path),
        "model_config_sha256": sha256(args.model / "config.json"),
        "config_sha256": sha256(config_path),
        "vision_policy": "frozen",
        "projector_policy": "frozen" if arm == "l" else "full_train_visual.merger",
        "language_policy": f"LoRA_r{args.lora_rank}_alpha{args.lora_alpha}_exact_28_layers",
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_scaling_alpha_over_r": args.lora_alpha / args.lora_rank,
        "lora_target_modules": target_modules(arm),
        "rl_not_started": True,
        "test_accessed_for_selection": False,
    }
    (args.run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
