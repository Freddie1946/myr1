#!/usr/bin/env python3
"""Run one audited formal seed-42 SFT scale experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import yaml

from checkpoint_retention import TwoTierCheckpointArchiver, checkpoint_step


AUTHORIZED_CONFIGS = {
    "sft_n0500_seed0042.yaml": ("pathvlm_sft_n0500", 500, 42),
    "sft_n1000_seed0042.yaml": ("pathvlm_sft_n1000", 1000, 42),
    "sft_n2000_seed0042.yaml": ("pathvlm_sft_n2000", 2000, 42),
    "sft_n3000_seed0042.yaml": ("pathvlm_sft_n3000", 3000, 42),
}
DISK_RESERVE_BYTES = 550 * 1024**3
FULL_CHECKPOINT_BUDGET_BYTES = 110 * 1024**3
MODEL_SNAPSHOT_BUDGET_BYTES = 18 * 1024**3
FORMAL_EPOCHS = 10
PROJECTED_START_FREE_BYTES = (
    DISK_RESERVE_BYTES
    # Trainer writes the new checkpoint before rotating the oldest of the two
    # retained checkpoints, so budget for three full checkpoints transiently.
    + 3 * FULL_CHECKPOINT_BUDGET_BYTES
    + (FORMAL_EPOCHS + 1) * MODEL_SNAPSHOT_BUDGET_BYTES
)
FORMAL_GPU_IDS = list(range(8))
MASTER_PORT = 29700


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def capture(command: list[str], output: Path, cwd: Path | None = None) -> int:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    output.write_text(result.stdout + result.stderr, encoding="utf-8")
    return result.returncode


def gpu_snapshot(selected: list[int]) -> dict:
    inventory = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    if inventory.returncode:
        raise RuntimeError(f"nvidia-smi inventory failed: {inventory.stderr.strip()}")
    rows = []
    uuid_to_index = {}
    for line in inventory.stdout.splitlines():
        index, uuid, name, total, used, free, util = [item.strip() for item in line.split(",")]
        row = {
            "index": int(index), "uuid": uuid, "name": name,
            "memory_total_mib": int(total), "memory_used_mib": int(used),
            "memory_free_mib": int(free), "utilization_percent": int(util),
        }
        rows.append(row)
        uuid_to_index[uuid] = row["index"]
    if any(index not in {row["index"] for row in rows} for index in selected):
        raise ValueError(f"selected GPU is not visible: {selected}")
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    if processes.returncode:
        raise RuntimeError(f"nvidia-smi process query failed: {processes.stderr.strip()}")
    selected_processes = []
    for line in processes.stdout.splitlines():
        if not line.strip():
            continue
        uuid, pid, name, used = [item.strip() for item in line.split(",")]
        index = uuid_to_index.get(uuid)
        if index in selected:
            selected_processes.append({
                "gpu_index": index, "pid": int(pid), "process_name": name,
                "used_memory_mib": int(used),
            })
    if selected_processes:
        raise RuntimeError(f"selected GPUs have compute processes: {selected_processes}")
    return {"gpus": [row for row in rows if row["index"] in selected], "compute_processes": []}


def require_master_port_free(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"distributed master port {port} is unavailable: {exc}") from exc


def monitor_resources(path: Path, stop: threading.Event, selected: set[int], install: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        while not stop.is_set():
            sample = {
                "timestamp": datetime.now().astimezone().isoformat(),
                "gpus": [],
                "disk_free_bytes": shutil.disk_usage(install).free,
            }
            gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                text=True, capture_output=True, check=False,
            )
            if gpu.returncode == 0:
                for line in gpu.stdout.splitlines():
                    index, used, util = [item.strip() for item in line.split(",")]
                    if int(index) in selected:
                        sample["gpus"].append({
                            "index": int(index), "memory_used_mib": int(used),
                            "utilization_percent": int(util),
                        })
            meminfo = {}
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                key, value = line.split(":", 1)
                if key in {"MemAvailable", "SwapTotal", "SwapFree"}:
                    meminfo[key] = int(value.strip().split()[0])
            sample["host_kib"] = meminfo
            handle.write(json.dumps(sample) + "\n")
            handle.flush()
            stop.wait(10.0)


def run_logged(command: list[str], log: Path, env: dict[str, str], cwd: Path,
               monitor: Path, selected: set[int], install: Path) -> None:
    stop = threading.Event()
    thread = threading.Thread(
        target=monitor_resources, args=(monitor, stop, selected, install), daemon=True,
    )
    thread.start()
    try:
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            returncode = process.wait()
        if returncode:
            raise RuntimeError(f"command failed with exit {returncode}: {shlex.join(command)}")
    finally:
        stop.set()
        thread.join(timeout=15)


def parse_trainability(log: Path) -> dict:
    text = log.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"trainable params:\s*([0-9,]+)\s*\|\|\s*all params:\s*([0-9,]+)\s*\|\|\s*trainable%:\s*([0-9.]+)",
        text,
    )
    if not matches:
        raise ValueError("trainable-parameter report not found")
    trainable, total, percent = matches[-1]
    result = {
        "trainable_parameters": int(trainable.replace(",", "")),
        "total_parameters": int(total.replace(",", "")),
        "trainable_percent": float(percent),
        "vision_freeze_logged": "Set vision model not trainable" in text,
        "projector_freeze_logged": "Set multi model projector not trainable" in text,
    }
    result["passed"] = (
        result["trainable_parameters"] > 7_000_000_000
        and result["total_parameters"] > result["trainable_parameters"]
        and result["vision_freeze_logged"] and result["projector_freeze_logged"]
    )
    return result


def summarize_resources(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    peaks = {}
    minimum_memory = None
    maximum_swap = 0
    minimum_disk = None
    for row in rows:
        for gpu in row.get("gpus", []):
            key = str(gpu["index"])
            peaks[key] = max(peaks.get(key, 0), gpu["memory_used_mib"])
        available = row.get("host_kib", {}).get("MemAvailable")
        if available is not None:
            minimum_memory = available if minimum_memory is None else min(minimum_memory, available)
        host = row.get("host_kib", {})
        maximum_swap = max(maximum_swap, host.get("SwapTotal", 0) - host.get("SwapFree", 0))
        disk = row.get("disk_free_bytes")
        if disk is not None:
            minimum_disk = disk if minimum_disk is None else min(minimum_disk, disk)
    return {
        "samples": len(rows), "peak_gpu_memory_used_mib": peaks,
        "minimum_host_mem_available_kib": minimum_memory,
        "peak_swap_used_kib": maximum_swap, "minimum_disk_free_bytes": minimum_disk,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--install-root", required=True, type=Path)
    parser.add_argument("--llamafactory-src", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--nproc-per-node", required=True, type=int)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    install = args.install_root.resolve()
    llamafactory = args.llamafactory_src.resolve()
    config_path = args.config.resolve()
    if config_path.name not in AUTHORIZED_CONFIGS:
        raise ValueError(f"unauthorized formal SFT config: {config_path.name}")
    expected_config_path = (install / "generated_configs" / "sft" / config_path.name).resolve()
    if config_path != expected_config_path:
        raise ValueError(f"config must be the generated formal config: {expected_config_path}")
    dataset_name, sample_count, seed = AUTHORIZED_CONFIGS[config_path.name]
    gpu_ids = [int(item.strip()) for item in args.cuda_visible_devices.split(",") if item.strip()]
    if len(gpu_ids) != args.nproc_per_node or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("CUDA device count must equal nproc-per-node and contain no duplicates")
    if gpu_ids != FORMAL_GPU_IDS or args.nproc_per_node != len(FORMAL_GPU_IDS):
        raise ValueError(
            "formal SFT topology is frozen to physical GPUs 0,1,2,3,4,5,6,7 "
            "with nproc-per-node=8"
        )
    free_before = shutil.disk_usage(install).free
    if free_before < PROJECTED_START_FREE_BYTES:
        raise RuntimeError(
            "insufficient disk for two-tier SFT retention: "
            f"free={free_before}, required={PROJECTED_START_FREE_BYTES}"
        )
    hardware_before = gpu_snapshot(gpu_ids)
    require_master_port_free(MASTER_PORT)

    base = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected = {
        "stage": "sft", "finetuning_type": "full", "freeze_vision_tower": True,
        "freeze_multi_modal_projector": True, "freeze_language_model": False,
        "dataset": dataset_name, "max_samples": sample_count, "seed": seed, "data_seed": seed,
        "learning_rate": 2e-5, "num_train_epochs": FORMAL_EPOCHS,
        "per_device_train_batch_size": 1, "gradient_accumulation_steps": 1,
        "lr_scheduler_type": "cosine", "warmup_ratio": 0.03, "bf16": True,
        "gradient_checkpointing": True, "cutoff_len": 512,
        "disable_gradient_checkpointing": False, "optim": "adamw_torch_fused",
        "save_strategy": "epoch", "save_only_model": False,
    }
    mismatches = {key: {"expected": value, "actual": base.get(key)}
                  for key, value in expected.items() if base.get(key) != value}
    if mismatches:
        raise ValueError(f"formal config mismatch: {mismatches}")
    if "test" in json.dumps(base).lower():
        raise ValueError("test reference is forbidden in formal SFT config")
    model_dir = Path(base["model_name_or_path"]).resolve()
    exact_model = install / "models" / "Qwen2.5-VL-7B-Instruct-cc594898137f460bfe9f0759e9844b3ce807cfb5"
    if model_dir != exact_model.resolve():
        raise ValueError(f"unexpected base model: {model_dir}")
    expected_deepspeed = (repo / "configs/deepspeed/ds_z2_gpu_torch_adamw.json").resolve()
    if Path(base["deepspeed"]).resolve() != expected_deepspeed:
        raise ValueError(f"unexpected DeepSpeed config: {base['deepspeed']}")
    dataset_dir = Path(base["dataset_dir"]).resolve()
    expected_dataset_dir = (install / "data/pathmmu_image_disjoint_v2/llamafactory").resolve()
    if dataset_dir != expected_dataset_dir:
        raise ValueError(f"formal SFT dataset root must be v2: {dataset_dir}")
    info = json.loads((dataset_dir / "dataset_info.json").read_text(encoding="utf-8"))
    adapter = dataset_dir / info[dataset_name]["file_name"]
    adapter_records = json.loads(adapter.read_text(encoding="utf-8"))
    if len(adapter_records) != sample_count:
        raise ValueError("adapter sample count mismatch")
    adapter_images = [Path(image) for record in adapter_records for image in record.get("images", [])]
    missing_images = [str(image) for image in adapter_images if not image.is_file()]
    if missing_images:
        raise FileNotFoundError(f"adapter contains missing images: {missing_images[:10]}")
    if not adapter_images:
        raise ValueError("adapter contains no image paths")
    preflight_path = install / "reports/preflight_report.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("passed") is not True:
        raise RuntimeError("formal-machine preflight report is not passing")
    if preflight.get("data", {}).get("data_version") != "pathmmu_image_disjoint_v2":
        raise RuntimeError("formal-machine preflight is not for PathMMU v2")
    required_preflight_gates = {
        "frozen_split_verification_passes", "no_image_overlap", "no_exact_content_overlap",
        "picked_json_unused",
        "adapter_source_counts_match", "model_id_matches", "model_revision_matches",
    }
    failed_preflight = {
        key: preflight.get("gates", {}).get(key)
        for key in required_preflight_gates
        if preflight.get("gates", {}).get(key) is not True
    }
    if failed_preflight:
        raise RuntimeError(f"required preflight gates failed: {failed_preflight}")

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_id = f"formal_sft_n{sample_count:04d}_seed{seed:04d}_{timestamp}"
    run_dir = install / "runs" / "stage1_sft" / f"n{sample_count:04d}_seed{seed:04d}" / run_id
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    output = run_dir / "output"
    epoch_snapshots = run_dir / "epoch_snapshots"
    snapshots = run_dir / "env_snapshot"
    snapshots.mkdir()
    resolved_config = {
        **base,
        "output_dir": str(output),
        "save_total_limit": 2,
        "overwrite_output_dir": False,
        "report_to": "none",
    }
    resolved_config.pop("resume_from_checkpoint", None)
    resolved_path = run_dir / "resolved_config.yaml"
    write_yaml(resolved_path, resolved_config)

    launcher = llamafactory / "src" / "llamafactory" / "launcher.py"
    command = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={args.nproc_per_node}", f"--master_port={MASTER_PORT}",
        str(launcher), str(resolved_path),
    ]
    (run_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1, "run_id": run_id, "status": "running",
        "stage": "stage1_sft", "formal_result": False,
        "created_at": datetime.now().astimezone().isoformat(),
        "data": {"version": "pathmmu_image_disjoint_v2", "dataset": dataset_name,
                 "qa_count": sample_count, "image_reference_count": len(adapter_images),
                 "all_image_paths_exist": True, "adapter": str(adapter),
                 "adapter_sha256": sha256(adapter)},
        "model": {"base_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                  "base_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
                  "base_path": str(model_dir)},
        "training": {"seed": seed, "epochs": FORMAL_EPOCHS, "learning_rate": 2e-5,
                     "finetuning_type": "full", "language_model_trainable": True,
                     "vision_tower_frozen": True, "multimodal_projector_frozen": True,
                     "backend": "deepspeed_zero2_gpu_fused_adamw_gc",
                     "checkpoint_policy": {
                         "save_strategy": "epoch",
                         "model_only_epoch_snapshots": FORMAL_EPOCHS,
                         "full_resume_checkpoints_retained": 2,
                         "final_model_retained": True,
                         "automatic_scientific_snapshot_pruning": False,
                     }},
        "disk_budget": {
            "free_before_bytes": free_before,
            "reserve_bytes": DISK_RESERVE_BYTES,
            "full_checkpoint_budget_bytes": FULL_CHECKPOINT_BUDGET_BYTES,
            "model_snapshot_budget_bytes": MODEL_SNAPSHOT_BUDGET_BYTES,
            "projected_start_free_required_bytes": PROJECTED_START_FREE_BYTES,
        },
        "hardware": {"host": socket.gethostname(), "cuda_visible_devices": gpu_ids,
                     "gpu_count": args.nproc_per_node, "before": hardware_before},
        "provenance": {
            "repo_code_manifest": str(repo / "protocol" / "code_hash_manifest_20260719_172547.json"),
            "base_model_manifest": str(repo / "protocol" / "base_model_manifest.json"),
            "formal_data_manifest": str(install / "data/pathmmu_image_disjoint_v2/formal_data_manifest.json"),
            "preflight_report": str(preflight_path), "preflight_report_sha256": sha256(preflight_path),
            "source_config": str(config_path), "source_config_sha256": sha256(config_path),
            "resolved_config_sha256": sha256(resolved_path),
        },
        "test_accessed": False,
    }
    manifest_path = run_dir / "run_manifest.yaml"
    write_yaml(manifest_path, manifest)

    capture([sys.executable, "-m", "pip", "freeze"], snapshots / "pip_freeze.txt")
    capture(["nvidia-smi"], snapshots / "nvidia_smi.txt")
    capture(["git", "-C", str(repo), "status", "--short", "--branch"], snapshots / "git_status.txt")
    capture(["git", "-C", str(repo), "rev-parse", "HEAD"], snapshots / "git_commit.txt")
    capture(["git", "-C", str(llamafactory), "rev-parse", "HEAD"], snapshots / "llamafactory_git.txt")
    for source in (
        repo / "protocol/code_hash_manifest_20260719_172547.json",
        repo / "protocol/base_model_manifest.json",
        install / "data/pathmmu_image_disjoint_v2/formal_data_manifest.json",
        preflight_path,
        Path(base["deepspeed"]), config_path,
    ):
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, snapshots / source.name)

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
        "NCCL_P2P_DISABLE": env.get("NCCL_P2P_DISABLE", "1"),
        "NCCL_IB_DISABLE": env.get("NCCL_IB_DISABLE", "1"),
        "PYTORCH_CUDA_ALLOC_CONF": env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        "WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1",
    })
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        env.pop(key, None)
    try:
        resources = run_dir / "resources.jsonl"
        archiver = TwoTierCheckpointArchiver(
            output,
            epoch_snapshots,
            run_dir / "checkpoint_retention_events.jsonl",
            minimum_free_bytes=DISK_RESERVE_BYTES,
        )
        archiver.start()
        try:
            run_logged(
                command, run_dir / "train.log", env, llamafactory, resources,
                set(gpu_ids), install,
            )
        finally:
            retention = archiver.stop_and_validate()
        if not (output / "model.safetensors.index.json").is_file():
            raise FileNotFoundError("final gathered model index is missing")
        load_report = run_dir / "final_checkpoint_load.json"
        load_command = [
            sys.executable, str(repo / "formal_machine/verify_sft_checkpoint_load.py"),
            "--checkpoint", str(output), "--output", str(load_report),
        ]
        load_result = subprocess.run(load_command, cwd=repo, env=env, text=True, capture_output=True, check=False)
        (run_dir / "final_checkpoint_load.log").write_text(
            load_result.stdout + load_result.stderr, encoding="utf-8"
        )
        if load_result.returncode:
            raise RuntimeError("final checkpoint independent load failed")
        delta_path = run_dir / "base_to_final_delta.json"
        delta_command = [
            sys.executable, str(repo / "scripts/compare_checkpoint_tensors.py"),
            "--parent", str(model_dir), "--child", str(output), "--output", str(delta_path),
        ]
        delta_result = subprocess.run(delta_command, cwd=repo, env=env, text=True, capture_output=True, check=False)
        (run_dir / "base_to_final_delta.log").write_text(
            delta_result.stdout + delta_result.stderr, encoding="utf-8"
        )
        if delta_result.returncode:
            raise RuntimeError("final tensor comparison failed")
        delta = json.loads(delta_path.read_text(encoding="utf-8"))
        trainer_state = json.loads((output / "trainer_state.json").read_text(encoding="utf-8"))
        history = trainer_state.get("log_history", [])
        losses = [float(row["loss"]) for row in history if "loss" in row]
        gradients = [float(row["grad_norm"]) for row in history if "grad_norm" in row]
        trainability = parse_trainability(run_dir / "train.log")
        log_text = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace")
        actual_gradient_checkpointing = "Gradient checkpointing enabled." in log_text
        full_checkpoints = sorted(
            (path for path in output.glob("checkpoint-*") if path.is_dir()),
            key=checkpoint_step,
        )
        retention_steps = [item["global_step"] for item in retention["snapshots"]]
        retention_epochs = [item.get("epoch") for item in retention["snapshots"]]
        expected_steps_per_epoch = math.ceil(sample_count / len(FORMAL_GPU_IDS))
        expected_retention_steps = [
            expected_steps_per_epoch * epoch for epoch in range(1, FORMAL_EPOCHS + 1)
        ]
        resource_summary = summarize_resources(resources)
        gates = {
            "training_completed": True,
            "final_checkpoint_saved": True,
            "final_checkpoint_reloaded": json.loads(load_report.read_text(encoding="utf-8"))["passed"],
            "trainability_freeze_gate": trainability["passed"],
            "losses_finite": bool(losses) and all(math.isfinite(value) for value in losses),
            "gradients_finite_nonzero": bool(gradients) and all(math.isfinite(value) and value > 0 for value in gradients),
            "language_tensor_changed": delta["tensors"]["language"]["changed_elements"] > 0,
            "visual_tensor_exactly_equal": delta["tensors"]["visual"]["exactly_equal"] is True,
            "gradient_checkpointing_observed": actual_gradient_checkpointing,
            "all_epoch_model_snapshots_saved": retention_steps == expected_retention_steps,
            "all_epoch_snapshot_metadata_present": (
                len(retention_epochs) == FORMAL_EPOCHS
                and all(epoch is not None for epoch in retention_epochs)
            ),
            "latest_two_full_resume_checkpoints_retained": (
                len(full_checkpoints) == 2
                and [checkpoint_step(path) for path in full_checkpoints]
                == expected_retention_steps[-2:]
            ),
            "disk_reserve_maintained": (
                resource_summary["minimum_disk_free_bytes"] is not None
                and resource_summary["minimum_disk_free_bytes"] >= DISK_RESERVE_BYTES
                and shutil.disk_usage(install).free >= DISK_RESERVE_BYTES
            ),
            "test_not_accessed": True,
        }
        manifest.update({
            "trainability": trainability, "trainer_state": {"global_step": trainer_state.get("global_step"),
            "loss_history": losses, "gradient_history": gradients},
            "observed_execution": {
                "gradient_checkpointing_enabled": actual_gradient_checkpointing,
                "full_resume_checkpoints": [str(path) for path in full_checkpoints],
            },
            "retention": retention,
            "resources": resource_summary, "gates": gates,
            "outputs": {"final_checkpoint": str(output), "run_dir": str(run_dir),
                        "epoch_snapshots": str(epoch_snapshots),
                        "retention_events": str(run_dir / "checkpoint_retention_events.jsonl")},
            "status": "completed" if all(gates.values()) else "failed_gate",
            "formal_result": all(gates.values()),
            "completed_at": datetime.now().astimezone().isoformat(),
        })
        write_yaml(manifest_path, manifest)
        if not all(gates.values()):
            raise RuntimeError(f"formal SFT gates failed: {gates}")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["formal_result"] = False
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        manifest["completed_at"] = datetime.now().astimezone().isoformat()
        write_yaml(manifest_path, manifest)
        raise

    print(json.dumps({"run_dir": str(run_dir), "manifest": str(manifest_path),
                      "status": manifest["status"]}, indent=2))


if __name__ == "__main__":
    main()
