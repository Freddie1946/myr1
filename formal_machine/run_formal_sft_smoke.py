#!/usr/bin/env python3
"""Run the formal 7B SFT one-step save/load/resume engineering gate."""

from __future__ import annotations

import argparse
import hashlib
import json
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


DISK_RESERVE_BYTES = 550 * 1024**3
SMOKE_START_FREE_BYTES = DISK_RESERVE_BYTES + (3 * 110 + 4 * 18) * 1024**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture(command: list[str], output: Path, cwd: Path | None = None) -> int:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    output.write_text(result.stdout + result.stderr, encoding="utf-8")
    return result.returncode


def monitor_resources(path: Path, stop: threading.Event) -> None:
    with path.open("w", encoding="utf-8") as handle:
        while not stop.is_set():
            sample: dict = {"timestamp": datetime.now().astimezone().isoformat(), "gpus": []}
            gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                text=True, capture_output=True, check=False,
            )
            if gpu.returncode == 0:
                for line in gpu.stdout.splitlines():
                    index, memory_used, utilization = [item.strip() for item in line.split(",")]
                    sample["gpus"].append({
                        "index": int(index), "memory_used_mib": int(memory_used),
                        "utilization_percent": int(utilization),
                    })
            meminfo = {}
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                key, value = line.split(":", 1)
                if key in {"MemAvailable", "SwapTotal", "SwapFree"}:
                    meminfo[key] = int(value.strip().split()[0])
            sample["host_kib"] = meminfo
            handle.write(json.dumps(sample) + "\n")
            handle.flush()
            stop.wait(2.0)


def run_logged(command: list[str], log: Path, env: dict[str, str], cwd: Path, monitor: Path | None = None) -> None:
    stop = threading.Event()
    thread = None
    if monitor is not None:
        thread = threading.Thread(target=monitor_resources, args=(monitor, stop), daemon=True)
        thread.start()
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
    stop.set()
    if thread is not None:
        thread.join(timeout=5)
    if returncode:
        raise RuntimeError(f"command failed with exit {returncode}: {shlex.join(command)}")


def write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def find_checkpoint(output: Path, step: int) -> Path:
    checkpoint = output / f"checkpoint-{step}"
    if not (checkpoint / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"missing gathered checkpoint: {checkpoint}")
    return checkpoint


def parse_trainability(log: Path) -> dict:
    text = log.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"trainable params:\s*([0-9,]+)\s*\|\|\s*all params:\s*([0-9,]+)\s*\|\|\s*trainable%:\s*([0-9.]+)",
        text,
    )
    if not matches:
        raise ValueError("LLaMA-Factory trainable-parameter report not found in training log")
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
        and result["vision_freeze_logged"]
        and result["projector_freeze_logged"]
    )
    if not result["passed"]:
        raise ValueError(f"trainability/freeze gate failed: {result}")
    return result


def checkpoint_metrics(checkpoint: Path) -> dict:
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    histories = [item for item in state.get("log_history", []) if "loss" in item or "train_loss" in item]
    return {"global_step": state.get("global_step"), "log_history": histories}


def resource_summary(path: Path, visible_gpu_ids: list[str]) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    visible = {int(item) for item in visible_gpu_ids}
    gpu_peaks = {}
    min_mem_available = None
    max_swap_used = 0
    for row in rows:
        for gpu in row.get("gpus", []):
            if gpu["index"] in visible:
                key = str(gpu["index"])
                gpu_peaks[key] = max(gpu_peaks.get(key, 0), gpu["memory_used_mib"])
        host = row.get("host_kib", {})
        available = host.get("MemAvailable")
        if available is not None:
            min_mem_available = available if min_mem_available is None else min(min_mem_available, available)
        max_swap_used = max(max_swap_used, host.get("SwapTotal", 0) - host.get("SwapFree", 0))
    return {
        "samples": len(rows),
        "peak_gpu_memory_used_mib": gpu_peaks,
        "minimum_host_mem_available_kib": min_mem_available,
        "peak_swap_used_kib": max_swap_used,
    }


def check_delta(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    language = payload["tensors"]["language"]
    visual = payload["tensors"]["visual"]
    passed = language["changed_elements"] > 0 and visual["exactly_equal"] is True
    return {"passed": passed, "language": language, "visual": visual}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--install-root", required=True, type=Path)
    parser.add_argument("--llamafactory-src", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--nproc-per-node", required=True, type=int)
    parser.add_argument("--master-port", required=True, type=int)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    install = args.install_root.resolve()
    llamafactory = args.llamafactory_src.resolve()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse run directory: {run_dir}")
    if shutil.disk_usage(install).free < SMOKE_START_FREE_BYTES:
        raise RuntimeError("insufficient disk for two-tier save/rotate/resume smoke")
    gpu_ids = [item.strip() for item in args.cuda_visible_devices.split(",") if item.strip()]
    if len(gpu_ids) != args.nproc_per_node:
        raise ValueError("CUDA_VISIBLE_DEVICES count must equal nproc-per-node")

    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    required = {
        "stage": "sft", "finetuning_type": "full", "freeze_vision_tower": True,
        "freeze_multi_modal_projector": True, "freeze_language_model": False,
        "dataset": "pathvlm_sft_smoke_n0008", "max_samples": 8, "seed": 42, "data_seed": 42,
    }
    mismatches = {key: {"expected": value, "actual": base.get(key)} for key, value in required.items() if base.get(key) != value}
    if mismatches:
        raise ValueError(f"formal smoke config mismatch: {mismatches}")
    backend_required = {
        "gradient_checkpointing": True,
        "disable_gradient_checkpointing": False,
        "optim": "adamw_torch_fused",
    }
    backend_mismatches = {
        key: {"expected": value, "actual": base.get(key)}
        for key, value in backend_required.items() if base.get(key) != value
    }
    if backend_mismatches:
        raise ValueError(f"formal smoke backend mismatch: {backend_mismatches}")
    expected_deepspeed = (repo / "configs/deepspeed/ds_z2_gpu_torch_adamw.json").resolve()
    if Path(base["deepspeed"]).resolve() != expected_deepspeed:
        raise ValueError(f"formal smoke must use audited ZeRO-2 backend: {base['deepspeed']}")
    model_dir = Path(base["model_name_or_path"]).resolve()
    dataset_dir = Path(base["dataset_dir"]).resolve()
    dataset_info = json.loads((dataset_dir / "dataset_info.json").read_text(encoding="utf-8"))
    adapter = dataset_dir / dataset_info[base["dataset"]]["file_name"]
    adapter_rows = json.loads(adapter.read_text(encoding="utf-8"))
    if len(adapter_rows) != 8:
        raise ValueError(f"expected 8 smoke samples, found {len(adapter_rows)}")

    run_dir.mkdir(parents=True)
    output = run_dir / "output"
    snapshots = run_dir / "env_snapshot"
    snapshots.mkdir()
    configs = run_dir / "configs"
    configs.mkdir()
    step1 = {**base, "output_dir": str(output), "max_steps": 1, "num_train_epochs": 1,
             "save_strategy": "steps", "save_steps": 1, "save_only_model": False,
             "overwrite_output_dir": False}
    step1.pop("resume_from_checkpoint", None)
    # The engineering smoke retains one full checkpoint to cap temporary disk;
    # the production runner retains two. Rotation semantics are identical.
    step1["save_total_limit"] = 1
    resume = {**step1, "max_steps": 2}
    rotate = {**step1, "max_steps": 3}
    write_yaml(configs / "step1.yaml", step1)
    write_yaml(configs / "resume.yaml", resume)
    write_yaml(configs / "rotate.yaml", rotate)

    launcher = llamafactory / "src/llamafactory/launcher.py"
    common = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={args.nproc_per_node}"]
    command1 = common + [f"--master_port={args.master_port}", str(launcher), str(configs / "step1.yaml")]
    command2 = common + [f"--master_port={args.master_port + 1}", str(launcher), str(configs / "resume.yaml")]
    command3 = common + [f"--master_port={args.master_port + 2}", str(launcher), str(configs / "rotate.yaml")]
    (run_dir / "command.txt").write_text(
        "STEP1\n" + shlex.join(command1)
        + "\n\nRESUME\n" + shlex.join(command2)
        + "\n\nROTATE\n" + shlex.join(command3) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "status": "running",
        "stage": "stage1_sft_smoke",
        "formal_result": False,
        "created_at": datetime.now().astimezone().isoformat(),
        "data": {"version": "pathmmu_image_disjoint_v1", "split": "sft_train", "qa_count": 8,
                 "adapter": str(adapter), "adapter_sha256": sha256(adapter)},
        "model": {"base_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                  "base_revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
                  "base_path": str(model_dir)},
        "training": {"finetuning_type": "full", "language_model_trainable": True,
                     "vision_tower_frozen": True, "multimodal_projector_frozen": True,
                     "seed": 42, "initial_max_steps": 1, "resumed_total_max_steps": 2,
                     "rotated_total_max_steps": 3,
                     "checkpoint_policy": "three model-only snapshots plus latest one full resume checkpoint"},
        "provenance": {
            "repo_code_manifest": str(repo / "protocol/code_hash_manifest_20260717_010146.json"),
            "base_model_manifest": str(repo / "protocol/base_model_manifest.json"),
            "formal_data_manifest": str(install / "data/pathmmu_image_disjoint_v1/formal_data_manifest.json"),
        },
        "hardware": {"host": socket.gethostname(), "cuda_visible_devices": gpu_ids,
                     "gpu_count": args.nproc_per_node},
        "resume_compatibility": {
            "torch_force_no_weights_only_load": True,
            "scope": "trusted locally generated DeepSpeed optimizer checkpoint only",
        },
        "test_accessed": False,
    }
    manifest_path = run_dir / "run_manifest.yaml"
    write_yaml(manifest_path, manifest)

    capture([sys.executable, "-m", "pip", "freeze"], snapshots / "pip_freeze.txt")
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda:
        capture([conda, "list", "--prefix", sys.prefix, "--explicit"], snapshots / "conda_explicit.txt")
    capture(["nvidia-smi"], snapshots / "nvidia_smi.txt")
    capture(["git", "-C", str(repo), "status", "--short", "--branch"], snapshots / "git_status.txt")
    capture(["git", "-C", str(repo), "rev-parse", "HEAD"], snapshots / "git_commit.txt")
    capture(["git", "-C", str(llamafactory), "rev-parse", "HEAD"], snapshots / "llamafactory_git.txt")
    capture([sys.executable, "-c", "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"], snapshots / "torch_cuda_versions.txt")
    for source in (
        repo / "protocol/code_hash_manifest_20260717_010146.json",
        repo / "protocol/base_model_manifest.json",
        install / "data/pathmmu_image_disjoint_v1/formal_data_manifest.json",
        Path(base["deepspeed"]),
    ):
        if not source.is_file():
            raise FileNotFoundError(f"required provenance file missing: {source}")
        shutil.copy2(source, snapshots / source.name)

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
        "NCCL_P2P_DISABLE": env.get("NCCL_P2P_DISABLE", "1"),
        "NCCL_IB_DISABLE": env.get("NCCL_IB_DISABLE", "1"),
        "PYTORCH_CUDA_ALLOC_CONF": env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        "WANDB_MODE": "disabled",
        "HF_HUB_OFFLINE": "1",
    })
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        env.pop(key, None)
    try:
        archiver = TwoTierCheckpointArchiver(
            output,
            run_dir / "epoch_snapshots",
            run_dir / "checkpoint_retention_events.jsonl",
            minimum_free_bytes=DISK_RESERVE_BYTES,
            poll_seconds=2.0,
        )
        archiver.start()
        run_logged(command1, run_dir / "step1.log", env, llamafactory, run_dir / "step1_resources.jsonl")
        checkpoint1 = find_checkpoint(output, 1)
        manifest["trainability"] = parse_trainability(run_dir / "step1.log")
        manifest["step1_metrics"] = checkpoint_metrics(checkpoint1)
        manifest["step1_resources"] = resource_summary(run_dir / "step1_resources.jsonl", gpu_ids)

        load1 = [sys.executable, str(repo / "formal_machine/verify_sft_checkpoint_load.py"),
                 "--checkpoint", str(checkpoint1), "--output", str(run_dir / "checkpoint1_load.json")]
        run_logged(load1, run_dir / "checkpoint1_load.log", env, repo)

        resume["resume_from_checkpoint"] = str(checkpoint1)
        write_yaml(configs / "resume.yaml", resume)
        resume_env = env.copy()
        resume_env.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)
        resume_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        run_logged(
            command2,
            run_dir / "resume.log",
            resume_env,
            llamafactory,
            run_dir / "resume_resources.jsonl",
        )
        checkpoint2 = find_checkpoint(output, 2)
        manifest["resume_metrics"] = checkpoint_metrics(checkpoint2)
        manifest["resume_resources"] = resource_summary(run_dir / "resume_resources.jsonl", gpu_ids)

        rotate["resume_from_checkpoint"] = str(checkpoint2)
        write_yaml(configs / "rotate.yaml", rotate)
        run_logged(
            command3,
            run_dir / "rotate.log",
            resume_env,
            llamafactory,
            run_dir / "rotate_resources.jsonl",
        )
        checkpoint3 = find_checkpoint(output, 3)
        manifest["rotate_metrics"] = checkpoint_metrics(checkpoint3)
        manifest["rotate_resources"] = resource_summary(run_dir / "rotate_resources.jsonl", gpu_ids)
        retention = archiver.stop_and_validate()
        snapshot1 = run_dir / "epoch_snapshots/checkpoint-1"
        snapshot2 = run_dir / "epoch_snapshots/checkpoint-2"
        snapshot3 = run_dir / "epoch_snapshots/checkpoint-3"
        remaining_full = sorted(
            (path for path in output.glob("checkpoint-*") if path.is_dir()), key=checkpoint_step,
        )

        load2 = [sys.executable, str(repo / "formal_machine/verify_sft_checkpoint_load.py"),
                 "--checkpoint", str(snapshot2), "--output", str(run_dir / "checkpoint2_load.json")]
        run_logged(load2, run_dir / "checkpoint2_load.log", env, repo)
        load3 = [sys.executable, str(repo / "formal_machine/verify_sft_checkpoint_load.py"),
                 "--checkpoint", str(snapshot3), "--output", str(run_dir / "checkpoint3_load.json")]
        run_logged(load3, run_dir / "checkpoint3_load.log", env, repo)

        delta1 = run_dir / "base_to_checkpoint1_delta.json"
        delta2 = run_dir / "checkpoint1_to_checkpoint2_delta.json"
        compare1 = [sys.executable, str(repo / "scripts/compare_checkpoint_tensors.py"),
                    "--parent", str(model_dir), "--child", str(snapshot1), "--output", str(delta1)]
        compare2 = [sys.executable, str(repo / "scripts/compare_checkpoint_tensors.py"),
                    "--parent", str(snapshot1), "--child", str(snapshot2), "--output", str(delta2)]
        run_logged(compare1, run_dir / "base_to_checkpoint1_delta.log", env, repo)
        run_logged(compare2, run_dir / "checkpoint1_to_checkpoint2_delta.log", env, repo)
        gates = {
            "optimizer_step_completed": True,
            "checkpoint1_saved": True,
            "checkpoint1_reloaded": json.loads((run_dir / "checkpoint1_load.json").read_text())["passed"],
            "resume_step_completed": True,
            "checkpoint2_saved": True,
            "checkpoint2_reloaded": json.loads((run_dir / "checkpoint2_load.json").read_text())["passed"],
            "second_resume_step_completed": True,
            "checkpoint3_saved": True,
            "checkpoint3_reloaded": json.loads((run_dir / "checkpoint3_load.json").read_text())["passed"],
            "all_three_model_snapshots_retained": (
                retention["count"] == 3
                and [item["global_step"] for item in retention["snapshots"]] == [1, 2, 3]
            ),
            "oldest_full_checkpoint_rotated": not (output / "checkpoint-1").exists(),
            "latest_full_checkpoint_retained": (
                [checkpoint_step(path) for path in remaining_full] == [3]
            ),
            "base_to_checkpoint1_delta": check_delta(delta1)["passed"],
            "checkpoint1_to_checkpoint2_delta": check_delta(delta2)["passed"],
            "trainability_freeze_gate": manifest["trainability"]["passed"],
            "gradient_checkpointing_observed": all(
                "Gradient checkpointing enabled." in (run_dir / name).read_text(
                    encoding="utf-8", errors="replace"
                )
                for name in ("step1.log", "resume.log", "rotate.log")
            ),
            "zero2_gpu_optimizer_observed": all(
                "Creating torch.bfloat16 ZeRO stage 2 optimizer" in (run_dir / name).read_text(
                    encoding="utf-8", errors="replace"
                )
                and "CPU Offload: False" in (run_dir / name).read_text(
                    encoding="utf-8", errors="replace"
                )
                for name in ("step1.log", "resume.log", "rotate.log")
            ),
            "test_not_accessed": True,
        }
        manifest["gates"] = gates
        manifest["retention"] = retention
        manifest["outputs"] = {
            "snapshot1": str(snapshot1), "snapshot2": str(snapshot2),
            "snapshot3": str(snapshot3),
            "full_checkpoint3": str(checkpoint3),
        }
        manifest["status"] = "completed" if all(gates.values()) else "failed_gate"
        manifest["completed_at"] = datetime.now().astimezone().isoformat()
        write_yaml(manifest_path, manifest)
        if not all(gates.values()):
            raise RuntimeError(f"formal SFT smoke gates failed: {gates}")
    except Exception as exc:
        if "archiver" in locals():
            try:
                manifest["retention_on_failure"] = archiver.stop_and_validate()
            except Exception as retention_exc:
                manifest["retention_stop_failure"] = {
                    "type": type(retention_exc).__name__, "message": str(retention_exc),
                }
        manifest["status"] = "failed"
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        manifest["completed_at"] = datetime.now().astimezone().isoformat()
        write_yaml(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
