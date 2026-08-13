#!/usr/bin/env python3
"""Freeze the L-r16 SFT3000 -> RL1000-as-SFT causal control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml


EXPECTED_RL1000_SHA256 = "5a818b3b4f3c9e5e1a17579111ff3e21008cf8124c4a05cb74eb2dcd326bbd73"
EXPECTED_PARENT_ADAPTER_SHA256 = "60148876d9dd35339c15bcce845259ee0ed0bdedbd51c681261f1a029a6e4abb"
EXPECTED_PARENT_CONFIG_SHA256 = "8d14bbeaef974e8294b59e673e164647e8db10cdab8583c5bf237b2dd6322033"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_parent(parent: Path) -> dict:
    weights = parent / "adapter_model.safetensors"
    config_path = parent / "adapter_config.json"
    state_path = parent / "trainer_state.json"
    for path in (weights, config_path, state_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(weights) != EXPECTED_PARENT_ADAPTER_SHA256:
        raise ValueError("selected SFT3000 adapter weight hash mismatch")
    if sha256(config_path) != EXPECTED_PARENT_CONFIG_SHA256:
        raise ValueError("selected SFT3000 adapter config hash mismatch")
    config = load_json(config_path)
    state = load_json(state_path)
    if config.get("peft_type") != "LORA" or config.get("r") != 16 or config.get("lora_alpha") != 32:
        raise ValueError("selected parent is not the frozen L-r16/alpha32 adapter")
    if state.get("global_step") != 80:
        raise ValueError("selected parent is not SFT3000 checkpoint-80")
    return {
        "path": str(parent.resolve()),
        "adapter_model_sha256": sha256(weights),
        "adapter_config_sha256": sha256(config_path),
        "source_global_step": state["global_step"],
        "source_epoch": state.get("epoch"),
        "lora_rank": config["r"],
        "lora_alpha": config["lora_alpha"],
        "target_module_count": len(config.get("target_modules", [])),
    }


def render_dataset(source: Path, data_dir: Path) -> dict:
    if sha256(source) != EXPECTED_RL1000_SHA256:
        raise ValueError("RL1000 frozen data hash mismatch")
    rows = load_json(source)
    if len(rows) != 1000:
        raise ValueError(f"expected 1000 RL records, found {len(rows)}")
    required = {"image", "problem", "solution"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError("RL1000 record schema mismatch")
    missing_images = [row["image"] for row in rows if not Path(row["image"]).is_file()]
    if missing_images:
        raise FileNotFoundError(f"missing training images: {missing_images[:5]}")
    records = [
        {
            "messages": [
                {"role": "user", "content": f"<image>{row['problem']}"},
                {"role": "assistant", "content": row["solution"]},
            ],
            "images": [row["image"]],
        }
        for row in rows
    ]
    data_dir.mkdir(parents=True, exist_ok=False)
    rendered = data_dir / "pathvlm_rl1000_as_sft.json"
    rendered.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    info = data_dir / "dataset_info.json"
    info.write_text(
        json.dumps(
            {
                "pathvlm_rl1000_as_sft": {
                    "file_name": rendered.name,
                    "formatting": "sharegpt",
                    "columns": {"messages": "messages", "images": "images"},
                    "tags": {
                        "role_tag": "role",
                        "content_tag": "content",
                        "user_tag": "user",
                        "assistant_tag": "assistant",
                    },
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "rendered": str(rendered.resolve()),
        "rendered_sha256": sha256(rendered),
        "record_count": len(records),
        "unique_image_count": len({row["image"] for row in rows}),
        "all_images_present": True,
    }


def build_config(
    model: Path,
    parent: Path,
    data_dir: Path,
    output: Path,
    *,
    world_size: int,
    micro_batch: int,
    gradient_accumulation: int,
    epochs: float,
    max_steps: int | None,
) -> dict:
    config = {
        "model_name_or_path": str(model.resolve()),
        "adapter_name_or_path": str(parent.resolve()),
        "create_new_adapter": False,
        "trust_remote_code": True,
        "image_max_pixels": 65536,
        "video_max_pixels": 8192,
        "flash_attn": "sdpa",
        "use_cache": False,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": True,
        "freeze_language_model": False,
        "dataset_dir": str(data_dir.resolve()),
        "dataset": "pathvlm_rl1000_as_sft",
        "template": "qwen2_vl",
        "cutoff_len": 1024,
        "max_samples": 1000,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        "output_dir": str(output.resolve()),
        "logging_steps": 1,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "save_only_model": True,
        "plot_loss": True,
        "overwrite_output_dir": False,
        "report_to": "none",
        "per_device_train_batch_size": micro_batch,
        "gradient_accumulation_steps": gradient_accumulation,
        "learning_rate": 5.0e-5,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "bf16": True,
        "gradient_checkpointing": False,
        "disable_gradient_checkpointing": True,
        "optim": "adamw_torch_fused",
        "seed": 42,
        "data_seed": 42,
        "disable_shuffling": False,
        "ddp_timeout": 180000000,
        "ddp_find_unused_parameters": False,
        "num_train_epochs": epochs,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
    }
    if max_steps is not None:
        config["max_steps"] = max_steps
        config["save_strategy"] = "no"
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--micro-batch", type=int, default=12)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    if not (args.model / "config.json").is_file():
        raise FileNotFoundError(args.model / "config.json")
    if (
        not 1 <= args.world_size <= 8
        or args.micro_batch < 1
        or args.gradient_accumulation < 1
        or args.epochs <= 0
    ):
        raise ValueError("invalid training shape")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps must be positive")

    parent = verify_parent(args.parent)
    args.run_dir.mkdir(parents=True)
    data = render_dataset(args.source, args.run_dir / "input_data")
    output = args.run_dir / "output"
    config = build_config(
        args.model,
        args.parent,
        args.run_dir / "input_data",
        output,
        world_size=args.world_size,
        micro_batch=args.micro_batch,
        gradient_accumulation=args.gradient_accumulation,
        epochs=args.epochs,
        max_steps=args.max_steps,
    )
    config_path = args.run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    global_batch = args.world_size * args.micro_batch * args.gradient_accumulation
    steps_per_epoch = math.ceil(1000 / global_batch)
    manifest = {
        "schema_version": 1,
        "status": "frozen_before_training",
        "scientific_role": "L-r16 SFT3000 parent plus exact RL1000 trained with supervised loss",
        "comparison_target": "same-parent 3000-SFT plus 1000 rule-RL n4/n8",
        "parent": parent,
        "data": data,
        "training": {
            "world_size": args.world_size,
            "micro_batch_per_gpu": args.micro_batch,
            "gradient_accumulation_steps": args.gradient_accumulation,
            "global_batch": global_batch,
            "epochs_on_additional_1000": args.epochs,
            "steps_per_epoch": steps_per_epoch,
            "expected_steps": args.max_steps or math.ceil(steps_per_epoch * args.epochs),
            "prompt_exposure": 1000 * args.epochs,
            "learning_rate": 5.0e-5,
            "scheduler": "cosine_with_fresh_optimizer",
            "vision": "frozen",
            "projector": "frozen",
            "language": "continue_same_LoRA_r16_alpha32",
            "save_strategy": config["save_strategy"],
        },
        "config_sha256": sha256(config_path),
        "model_config_sha256": sha256(args.model / "config.json"),
        "test_used_for_training_or_selection": False,
    }
    (args.run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
