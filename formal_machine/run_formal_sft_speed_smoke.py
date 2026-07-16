#!/usr/bin/env python3
"""Run an audited, training-only SFT throughput smoke on the formal GPUs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

from run_formal_sft import (
    capture,
    gpu_snapshot,
    parse_trainability,
    require_master_port_free,
    run_logged,
    sha256,
    summarize_resources,
    write_yaml,
)


FORMAL_GPU_IDS = list(range(8))
BASE_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
VARIANTS = {
    "z2_gpu_nogc_fused": {
        "deepspeed": "configs/deepspeed/ds_z2_gpu_torch_adamw.json",
        "gradient_checkpointing": False,
        "optim": "adamw_torch_fused",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--install-root", required=True, type=Path)
    parser.add_argument("--llamafactory-src", required=True, type=Path)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--master-port", type=int, default=29710)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    install = args.install_root.resolve()
    llamafactory = args.llamafactory_src.resolve()
    run_dir = args.run_dir.resolve()
    variant = VARIANTS[args.variant]
    if not 5 <= args.max_steps <= 50:
        parser.error("max-steps must be between 5 and 50")
    if run_dir.exists():
        raise FileExistsError(f"speed-smoke run directory already exists: {run_dir}")

    preflight = install / "reports/preflight_report.json"
    preflight_payload = json.loads(preflight.read_text(encoding="utf-8"))
    if preflight_payload.get("passed") is not True:
        raise RuntimeError("formal preflight is not passing")
    model_dir = install / f"models/Qwen2.5-VL-7B-Instruct-{BASE_REVISION}"
    adapter = install / "data/pathmmu_image_disjoint_v1/llamafactory/pathvlm_sft_n0500.json"
    dataset_info = install / "data/pathmmu_image_disjoint_v1/llamafactory/dataset_info.json"
    deepspeed = (repo / variant["deepspeed"]).resolve()
    for required in (model_dir / "config.json", adapter, dataset_info, deepspeed):
        if not required.exists():
            raise FileNotFoundError(required)

    before = gpu_snapshot(FORMAL_GPU_IDS)
    require_master_port_free(args.master_port)
    if shutil.disk_usage(install).free < 100 * 1024**3:
        raise RuntimeError("less than 100 GiB free disk before speed smoke")

    run_dir.mkdir(parents=True)
    output = run_dir / "output"
    snapshots = run_dir / "env_snapshot"
    snapshots.mkdir()
    resolved = {
        "model_name_or_path": str(model_dir),
        "trust_remote_code": True,
        "image_max_pixels": 65536,
        "video_max_pixels": 8192,
        "flash_attn": "sdpa",
        "use_cache": False,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "full",
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": True,
        "freeze_language_model": False,
        "deepspeed": str(deepspeed),
        "dataset_dir": str(adapter.parent),
        "dataset": "pathvlm_sft_n0500",
        "template": "qwen2_vl",
        "cutoff_len": 512,
        "max_samples": 64,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        "output_dir": str(output),
        "logging_steps": 1,
        "save_strategy": "no",
        "plot_loss": False,
        "overwrite_output_dir": False,
        "report_to": "none",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 2.0e-5,
        "max_steps": args.max_steps,
        "num_train_epochs": 1,
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
        "bf16": True,
        "gradient_checkpointing": variant["gradient_checkpointing"],
        "optim": variant["optim"],
        "seed": 42,
        "data_seed": 42,
        "ddp_timeout": 180000000,
    }
    resolved_path = run_dir / "resolved_config.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
    command = [
        str(install / "envs/sft/bin/python"), "-m", "torch.distributed.run",
        "--nproc_per_node=8", f"--master_port={args.master_port}",
        str(llamafactory / "src/llamafactory/launcher.py"), str(resolved_path),
    ]
    (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    manifest = {
        "run_id": run_dir.name,
        "stage": "stage1_sft_speed_smoke",
        "status": "running",
        "formal_result": False,
        "started_at": datetime.now().astimezone().isoformat(),
        "purpose": "engineering throughput and memory limit measurement only",
        "variant": args.variant,
        "training": {
            "dataset": "pathvlm_sft_n0500",
            "max_samples": 64,
            "max_steps": args.max_steps,
            "seed": 42,
            "global_batch_size": 8,
            "zero_stage": 2,
            "optimizer_offload": False,
            "gradient_checkpointing": variant["gradient_checkpointing"],
            "optim": variant["optim"],
            "language_model_trainable": True,
            "vision_tower_frozen": True,
            "multimodal_projector_frozen": True,
        },
        "model": {"path": str(model_dir), "revision": BASE_REVISION},
        "data": {"adapter": str(adapter), "adapter_sha256": sha256(adapter)},
        "hardware": {"gpu_ids": FORMAL_GPU_IDS, "before": before},
        "provenance": {
            "repo_commit": subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
                capture_output=True, check=True,
            ).stdout.strip(),
            "resolved_config_sha256": sha256(resolved_path),
            "deepspeed_config": str(deepspeed),
            "deepspeed_config_sha256": sha256(deepspeed),
        },
        "test_accessed": False,
    }
    manifest_path = run_dir / "run_manifest.yaml"
    write_yaml(manifest_path, manifest)
    capture([sys.executable, "-m", "pip", "freeze"], snapshots / "pip_freeze.txt")
    capture(["nvidia-smi"], snapshots / "nvidia_smi_before.txt")
    capture(["git", "-C", str(repo), "status", "--short", "--branch"], snapshots / "git_status.txt")

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "WANDB_MODE": "disabled",
        "HF_HUB_OFFLINE": "1",
    })
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        env.pop(key, None)
    resources = run_dir / "resources.jsonl"
    try:
        run_logged(command, run_dir / "train.log", env, llamafactory, resources, set(FORMAL_GPU_IDS), install)
        train_results = json.loads((output / "train_results.json").read_text(encoding="utf-8"))
        trainer_state = json.loads((output / "trainer_state.json").read_text(encoding="utf-8"))
        trainability = parse_trainability(run_dir / "train.log")
        steps_per_second = float(train_results["train_steps_per_second"])
        gates = {
            "completed_requested_steps": trainer_state.get("global_step") == args.max_steps,
            "positive_finite_throughput": steps_per_second > 0,
            "trainability_freeze_gate": trainability["passed"],
            "test_not_accessed": True,
        }
        manifest.update({
            "status": "completed" if all(gates.values()) else "failed_gate",
            "completed_at": datetime.now().astimezone().isoformat(),
            "formal_result": False,
            "metrics": {
                "train_runtime_seconds": float(train_results["train_runtime"]),
                "train_steps_per_second": steps_per_second,
                "seconds_per_step": 1.0 / steps_per_second,
                "train_samples_per_second": float(train_results["train_samples_per_second"]),
                "train_loss": float(train_results["train_loss"]),
            },
            "trainability": trainability,
            "resources": summarize_resources(resources),
            "gates": gates,
            "outputs": {"output_dir": str(output), "log": str(run_dir / "train.log")},
        })
        write_yaml(manifest_path, manifest)
        if not all(gates.values()):
            raise RuntimeError(f"speed-smoke gates failed: {gates}")
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["formal_result"] = False
        manifest["completed_at"] = datetime.now().astimezone().isoformat()
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        if resources.exists() and resources.stat().st_size:
            manifest["resources"] = summarize_resources(resources)
        write_yaml(manifest_path, manifest)
        raise
    finally:
        capture(["nvidia-smi"], snapshots / "nvidia_smi_after.txt")

    print(json.dumps({"manifest": str(manifest_path), "metrics": manifest["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
