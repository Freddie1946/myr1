#!/usr/bin/env python3
"""Run the frozen n3000-parent + RL1000 two-epoch SFT control."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

from checkpoint_retention import TwoTierCheckpointArchiver, checkpoint_step
from run_formal_sft import (
    gpu_snapshot,
    parse_trainability,
    require_master_port_free,
    run_logged,
    sha256,
    summarize_resources,
    write_yaml,
)


DATA_VERSION = "pathmmu_image_disjoint_v2"
SAMPLE_COUNT = 1000
SEED = 42
EPOCHS = 2
GLOBAL_BATCH_SIZE = 8
GPU_IDS = list(range(8))
MASTER_PORT = 29740
LEARNING_RATE = 2e-5
EXPECTED_PARENT_MANIFEST_SHA256 = (
    "6c8708dd677f5ab1563e77317d86cd1b53cb9b969ebda09e4a312581a2b9f1f1"
)
EXPECTED_FROZEN_DATA_SHA256 = (
    "5ca2160068ab11f9908701e2eaa69ab1ebf6107a9b6b4fc265123eae40d88f44"
)
EXPECTED_REWRITTEN_DATA_SHA256 = (
    "5a818b3b4f3c9e5e1a17579111ff3e21008cf8124c4a05cb74eb2dcd326bbd73"
)
DISK_RESERVE_BYTES = 550 * 1024**3
PROJECTED_START_FREE_BYTES = DISK_RESERVE_BYTES + 400 * 1024**3


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_checkpoint_identity(parent: Path) -> dict:
    required = {
        "config.json",
        "model.safetensors.index.json",
        "snapshot_manifest.json",
        "trainer_state.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
    }
    missing = sorted(name for name in required if not (parent / name).is_file())
    if missing:
        raise FileNotFoundError(f"parent checkpoint lacks required files: {missing}")
    manifest_path = parent / "snapshot_manifest.json"
    if sha256(manifest_path) != EXPECTED_PARENT_MANIFEST_SHA256:
        raise ValueError("parent snapshot-manifest SHA-256 mismatch")
    manifest = read_json(manifest_path)
    state = read_json(parent / "trainer_state.json")
    if manifest.get("global_step") != 1125 or float(manifest.get("epoch")) != 3.0:
        raise ValueError("parent snapshot identity is not epoch 3 / step 1125")
    if state.get("global_step") != 1125 or float(state.get("epoch")) != 3.0:
        raise ValueError("parent trainer state is not epoch 3 / step 1125")
    config = read_json(parent / "config.json")
    if config.get("model_type") != "qwen2_5_vl":
        raise ValueError(f"unexpected parent model type: {config.get('model_type')}")
    index = read_json(parent / "model.safetensors.index.json")
    shards = sorted(set(index.get("weight_map", {}).values()))
    if not shards:
        raise ValueError("parent checkpoint index has no weight shards")
    bad_shards = [
        name for name in shards
        if not (parent / name).is_file() or (parent / name).stat().st_size <= 0
    ]
    if bad_shards:
        raise FileNotFoundError(f"parent checkpoint shards missing/empty: {bad_shards}")
    return {
        "path": str(parent),
        "snapshot_manifest_sha256": sha256(manifest_path),
        "epoch": manifest["epoch"],
        "global_step": manifest["global_step"],
        "model_type": config["model_type"],
        "weight_shards": shards,
        "indexed_total_size": index.get("metadata", {}).get("total_size"),
        "all_indexed_shards_present": True,
    }


def verify_and_render_data(repo: Path, install: Path, output: Path) -> dict:
    frozen_path = repo / f"data/{DATA_VERSION}/subsets/rl/rl_1000_with_cot.json"
    rewritten_path = install / f"data/{DATA_VERSION}/rewritten_records/rl_1000.json"
    if sha256(frozen_path) != EXPECTED_FROZEN_DATA_SHA256:
        raise ValueError("frozen RL1000 SHA-256 mismatch")
    if sha256(rewritten_path) != EXPECTED_REWRITTEN_DATA_SHA256:
        raise ValueError("rewritten RL1000 SHA-256 mismatch")
    frozen = read_json(frozen_path)
    rewritten = read_json(rewritten_path)
    if len(frozen) != SAMPLE_COUNT or len(rewritten) != SAMPLE_COUNT:
        raise ValueError("RL1000 record count mismatch")
    frozen_keys = [
        (Path(row["image"]).name, row["problem"], row["solution"]) for row in frozen
    ]
    rewritten_keys = [
        (Path(row["image"]).name, row["problem"], row["solution"]) for row in rewritten
    ]
    if frozen_keys != rewritten_keys:
        raise ValueError("rewritten RL1000 order/content differs from frozen source")
    missing_images = [
        row["image"] for row in rewritten if not Path(row["image"]).is_file()
    ]
    if missing_images:
        raise FileNotFoundError(f"RL1000 images missing: {missing_images[:10]}")

    records = [
        {
            "messages": [
                {"role": "user", "content": f"<image>{row['problem']}"},
                {"role": "assistant", "content": row["solution"]},
            ],
            "images": [row["image"]],
        }
        for row in rewritten
    ]
    output.mkdir(parents=True, exist_ok=False)
    adapter = output / "pathvlm_sft4000_rl_n1000.json"
    adapter.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dataset_info = output / "dataset_info.json"
    dataset_info.write_text(
        json.dumps(
            {
                "pathvlm_sft4000_rl_n1000": {
                    "file_name": adapter.name,
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
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "version": DATA_VERSION,
        "scientific_role": "same 1000 records used by Stage2 RL, now with gold CoT SFT targets",
        "count": SAMPLE_COUNT,
        "frozen_source": str(frozen_path),
        "frozen_source_sha256": sha256(frozen_path),
        "rewritten_source": str(rewritten_path),
        "rewritten_source_sha256": sha256(rewritten_path),
        "adapter": str(adapter),
        "adapter_sha256": sha256(adapter),
        "dataset_info": str(dataset_info),
        "dataset_info_sha256": sha256(dataset_info),
        "source_order_and_content_exact": True,
        "all_image_paths_exist": True,
    }


def resolved_config(parent: Path, data_dir: Path, output: Path, deepspeed: Path) -> dict:
    return {
        "model_name_or_path": str(parent),
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
        "dataset_dir": str(data_dir),
        "dataset": "pathvlm_sft4000_rl_n1000",
        "template": "qwen2_vl",
        "cutoff_len": 512,
        "max_samples": SAMPLE_COUNT,
        "overwrite_cache": True,
        "preprocessing_num_workers": 16,
        "output_dir": str(output),
        "logging_steps": 1,
        "save_strategy": "epoch",
        "save_only_model": False,
        "save_total_limit": 2,
        "plot_loss": True,
        "overwrite_output_dir": False,
        "report_to": "none",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": LEARNING_RATE,
        "num_train_epochs": EPOCHS,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "bf16": True,
        "gradient_checkpointing": True,
        "disable_gradient_checkpointing": False,
        "optim": "adamw_torch_fused",
        "seed": SEED,
        "data_seed": SEED,
        "ddp_timeout": 180000000,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--install-root", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--llamafactory-src", required=True, type=Path)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    install = args.install_root.resolve()
    parent = args.parent.resolve()
    llamafactory = args.llamafactory_src.resolve()
    python = args.python.resolve()
    protocol_path = args.protocol.resolve()
    deepspeed = (repo / "configs/deepspeed/ds_z2_gpu_torch_adamw.json").resolve()
    for path in (parent, llamafactory, python, protocol_path, deepspeed):
        if not path.exists():
            raise FileNotFoundError(path)
    protocol = read_json(protocol_path)
    expected_protocol = {
        "status": "frozen_before_execution",
        "parent_snapshot_manifest_sha256": EXPECTED_PARENT_MANIFEST_SHA256,
        "frozen_rl1000_sha256": EXPECTED_FROZEN_DATA_SHA256,
        "epochs": EPOCHS,
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "global_batch_size": GLOBAL_BATCH_SIZE,
    }
    mismatches = {
        key: {"expected": value, "actual": protocol.get(key)}
        for key, value in expected_protocol.items()
        if protocol.get(key) != value
    }
    if mismatches:
        raise ValueError(f"protocol mismatch: {mismatches}")
    if sha256(Path(__file__)) != protocol.get("runner_sha256"):
        raise ValueError("runner SHA-256 differs from frozen protocol")

    free_before = shutil.disk_usage(install).free
    if free_before < PROJECTED_START_FREE_BYTES:
        raise RuntimeError(
            f"insufficient disk: free={free_before}, required={PROJECTED_START_FREE_BYTES}"
        )
    hardware_before = gpu_snapshot(GPU_IDS)
    require_master_port_free(MASTER_PORT)
    parent_identity = verify_checkpoint_identity(parent)

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_id = f"sft4000_control_rl1000_seed0042_epoch02_{stamp}"
    run_dir = (
        install
        / "runs/stage2_control_sft4000/n1000_seed0042"
        / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    output = run_dir / "output"
    data = verify_and_render_data(repo, install, run_dir / "input_data")
    config = resolved_config(parent, Path(data["dataset_info"]).parent, output, deepspeed)
    config_path = run_dir / "resolved_config.yaml"
    write_yaml(config_path, config)

    launcher = llamafactory / "src/llamafactory/launcher.py"
    if not launcher.is_file():
        raise FileNotFoundError(launcher)
    command = [
        str(python),
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={len(GPU_IDS)}",
        f"--master_port={MASTER_PORT}",
        str(launcher),
        str(config_path),
    ]
    (run_dir / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )
    manifest_path = run_dir / "run_manifest.yaml"
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "formal_result": False,
        "scientific_role": (
            "causal control: selected n3000 SFT parent plus the exact Stage2 RL1000 "
            "records trained with ordinary supervised loss for two epochs"
        ),
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": parent_identity,
        "data": data,
        "training": {
            "optimizer_state": "fresh branch optimizer; model weights continue from n3000 parent",
            "epochs": EPOCHS,
            "steps_per_epoch": math.ceil(SAMPLE_COUNT / GLOBAL_BATCH_SIZE),
            "expected_global_steps": EPOCHS
            * math.ceil(SAMPLE_COUNT / GLOBAL_BATCH_SIZE),
            "global_batch_size": GLOBAL_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "scheduler": "cosine",
            "warmup_ratio": 0.03,
            "seed": SEED,
            "language_model_trainable": True,
            "vision_tower_frozen": True,
            "multimodal_projector_frozen": True,
            "checkpoint_selection": "epoch 2 fixed by exposure-matched protocol; no validation/test selection",
        },
        "hardware": {
            "host": socket.gethostname(),
            "cuda_visible_devices": GPU_IDS,
            "before": hardware_before,
        },
        "provenance": {
            "protocol": str(protocol_path),
            "protocol_sha256": sha256(protocol_path),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256(Path(__file__)),
            "resolved_config_sha256": sha256(config_path),
            "llamafactory": str(llamafactory),
        },
        "pathmmu_test_used_for_training_or_selection": False,
        "stage3_started": False,
    }
    write_yaml(manifest_path, manifest)
    snapshots = run_dir / "env_snapshot"
    snapshots.mkdir()
    subprocess.run(
        [str(python), "-m", "pip", "freeze"],
        text=True,
        stdout=(snapshots / "pip_freeze.txt").open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        check=False,
    )
    subprocess.run(
        ["nvidia-smi"],
        text=True,
        stdout=(snapshots / "nvidia_smi.txt").open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        check=False,
    )

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(map(str, GPU_IDS)),
            "NCCL_P2P_DISABLE": env.get("NCCL_P2P_DISABLE", "1"),
            "NCCL_IB_DISABLE": env.get("NCCL_IB_DISABLE", "1"),
            "PYTORCH_CUDA_ALLOC_CONF": env.get(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
            ),
            "WANDB_MODE": "disabled",
            "HF_HUB_OFFLINE": "1",
        }
    )
    for key in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "all_proxy",
    ):
        env.pop(key, None)

    try:
        archiver = TwoTierCheckpointArchiver(
            output,
            run_dir / "epoch_snapshots",
            run_dir / "checkpoint_retention_events.jsonl",
            minimum_free_bytes=DISK_RESERVE_BYTES,
        )
        archiver.start()
        try:
            run_logged(
                command,
                run_dir / "train.log",
                env,
                llamafactory,
                run_dir / "resources.jsonl",
                set(GPU_IDS),
                install,
            )
        finally:
            retention = archiver.stop_and_validate()

        expected_steps = [125, 250]
        observed_steps = [item["global_step"] for item in retention["snapshots"]]
        state = read_json(output / "trainer_state.json")
        load_report = run_dir / "final_checkpoint_load.json"
        load_result = subprocess.run(
            [
                str(python),
                str(repo / "formal_machine/verify_sft_checkpoint_load.py"),
                "--checkpoint",
                str(output),
                "--output",
                str(load_report),
            ],
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        (run_dir / "final_checkpoint_load.log").write_text(
            load_result.stdout + load_result.stderr, encoding="utf-8"
        )
        if load_result.returncode:
            raise RuntimeError("final checkpoint independent load failed")
        delta_path = run_dir / "parent_to_final_delta.json"
        delta_result = subprocess.run(
            [
                str(python),
                str(repo / "scripts/compare_checkpoint_tensors.py"),
                "--parent",
                str(parent),
                "--child",
                str(output),
                "--output",
                str(delta_path),
            ],
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        (run_dir / "parent_to_final_delta.log").write_text(
            delta_result.stdout + delta_result.stderr, encoding="utf-8"
        )
        if delta_result.returncode:
            raise RuntimeError("parent/final tensor comparison failed")
        delta = read_json(delta_path)
        history = state.get("log_history", [])
        losses = [float(row["loss"]) for row in history if "loss" in row]
        gradients = [float(row["grad_norm"]) for row in history if "grad_norm" in row]
        trainability = parse_trainability(run_dir / "train.log")
        resource_summary = summarize_resources(run_dir / "resources.jsonl")
        checkpoints = sorted(
            (path for path in output.glob("checkpoint-*") if path.is_dir()),
            key=checkpoint_step,
        )
        gates = {
            "training_completed": True,
            "global_step_250": state.get("global_step") == 250,
            "both_epoch_snapshots_saved": observed_steps == expected_steps,
            "both_full_resume_checkpoints_retained": [
                checkpoint_step(path) for path in checkpoints
            ]
            == expected_steps,
            "final_checkpoint_reloaded": read_json(load_report).get("passed") is True,
            "trainability_freeze_gate": trainability["passed"],
            "losses_finite": bool(losses)
            and all(math.isfinite(value) for value in losses),
            "gradients_finite_nonzero": bool(gradients)
            and all(math.isfinite(value) and value > 0 for value in gradients),
            "language_tensor_changed": (
                delta["tensors"]["language"]["changed_elements"] > 0
            ),
            "visual_tensor_exactly_equal": (
                delta["tensors"]["visual"]["exactly_equal"] is True
            ),
            "disk_reserve_maintained": (
                resource_summary["minimum_disk_free_bytes"] >= DISK_RESERVE_BYTES
                and shutil.disk_usage(install).free >= DISK_RESERVE_BYTES
            ),
            "pathmmu_test_not_used_for_training_or_selection": True,
            "stage3_not_started": True,
        }
        manifest.update(
            {
                "status": "completed" if all(gates.values()) else "failed_gate",
                "formal_result": all(gates.values()),
                "completed_at": datetime.now().astimezone().isoformat(),
                "trainer_state": {
                    "global_step": state.get("global_step"),
                    "epoch": state.get("epoch"),
                    "loss_history": losses,
                    "gradient_history": gradients,
                },
                "trainability": trainability,
                "retention": retention,
                "resources": resource_summary,
                "gates": gates,
                "outputs": {
                    "run_dir": str(run_dir),
                    "final_checkpoint": str(output),
                    "fixed_evaluation_checkpoint": str(
                        run_dir / "epoch_snapshots/checkpoint-250"
                    ),
                },
            }
        )
        write_yaml(manifest_path, manifest)
        if not all(gates.values()):
            raise RuntimeError(f"SFT4000 completion gates failed: {gates}")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["formal_result"] = False
        manifest["completed_at"] = datetime.now().astimezone().isoformat()
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        write_yaml(manifest_path, manifest)
        raise

    print(
        json.dumps(
            {
                "status": manifest["status"],
                "run_dir": str(run_dir),
                "fixed_evaluation_checkpoint": manifest["outputs"][
                    "fixed_evaluation_checkpoint"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
