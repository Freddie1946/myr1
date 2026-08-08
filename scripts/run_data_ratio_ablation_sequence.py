#!/usr/bin/env python3
"""Fail-closed sequential runner for the sparse SFT/rule-RL ablation."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml


GPU_IDS = tuple(range(8))
SFT_MASTER_PORT = 29820
RL_MASTER_PORT = 29821
DISK_RESERVE_BYTES = 700 * 1024**3
BASE_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"


class SequenceStop(RuntimeError):
    pass


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def acquire_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise SequenceStop(f"another sequence owns lock: {path}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} started_at={now_iso()}\n")
    handle.flush()
    return handle


def require_port_free(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise SequenceStop(f"distributed port {port} is unavailable: {exc}") from exc


def model_files(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    required = ("config.json", "preprocessor_config.json", "tokenizer_config.json")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    index = checkpoint / "model.safetensors.index.json"
    single = checkpoint / "model.safetensors"
    weight_files: list[Path]
    if index.is_file():
        payload = json.loads(index.read_text(encoding="utf-8"))
        names = sorted(set(payload.get("weight_map", {}).values()))
        weight_files = [checkpoint / name for name in names]
        if not names:
            missing.append("nonempty model.safetensors.index.json")
    elif single.is_file():
        weight_files = [single]
    else:
        weight_files = []
        missing.append("model weights")
    missing.extend(str(path.name) for path in weight_files if not path.is_file())
    if missing:
        raise SequenceStop(f"checkpoint is structurally incomplete: {checkpoint}: {missing[:8]}")
    empty = [path.name for path in weight_files if path.stat().st_size <= 0]
    if empty:
        raise SequenceStop(f"checkpoint has empty weight files: {checkpoint}: {empty}")
    return {
        "path": str(checkpoint),
        "weight_file_count": len(weight_files),
        "weight_bytes": sum(path.stat().st_size for path in weight_files),
        "config_sha256": sha256_file(checkpoint / "config.json"),
    }


def gpu_preflight() -> dict[str, Any]:
    inventory = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    if inventory.returncode:
        raise SequenceStop(f"nvidia-smi inventory failed: {inventory.stderr.strip()}")
    rows = []
    for line in inventory.stdout.splitlines():
        index, name, total, used, util = [item.strip() for item in line.split(",")]
        rows.append({
            "index": int(index), "name": name, "memory_total_mib": int(total),
            "memory_used_mib": int(used), "utilization_percent": int(util),
        })
    if [row["index"] for row in rows] != list(GPU_IDS):
        raise SequenceStop(f"expected physical GPUs 0-7, found {[row['index'] for row in rows]}")
    if any("A100" not in row["name"] for row in rows):
        raise SequenceStop("formal ratio queue requires eight A100 GPUs")
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    if processes.returncode:
        raise SequenceStop(f"nvidia-smi process query failed: {processes.stderr.strip()}")
    active = [line.strip() for line in processes.stdout.splitlines() if line.strip()]
    if active:
        raise SequenceStop(f"GPU compute processes are already active: {active}")
    return {"gpus": rows, "compute_processes": []}


def latest_checkpoint(output: Path) -> Path | None:
    candidates = []
    for path in output.glob("checkpoint-*") if output.is_dir() else ():
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        if path.is_dir() and (path / "trainer_state.json").is_file():
            candidates.append((step, path))
    return max(candidates, default=(0, None))[1]


@dataclass(frozen=True)
class Task:
    task_id: str
    stage: str
    dataset_name: str | None = None
    dataset_yaml: str | None = None
    parent_kind: str = "base"
    parent_task: str | None = None
    sample_count: int = 0
    max_steps: int | None = None
    epochs: int | None = None
    gate_only: bool = False


def build_tasks(mode: str, data_manifest: dict[str, Any]) -> list[Task]:
    arms = data_manifest["arms"]
    if mode == "smoke":
        selected = ("sft0750_rl0250", "sft0250_rl0750")
        tasks: list[Task] = []
        for arm in selected:
            sft_id = f"smoke_{arm}_sft"
            tasks.extend([
                Task(
                    sft_id, "sft",
                    dataset_name=f"pathvlm_ratio_smoke_{arm}_sft",
                    sample_count=8, max_steps=1,
                ),
                Task(
                    f"smoke_{arm}_rule_rl", "rule_rl",
                    dataset_yaml=f"grpo/pathvlm_ratio_smoke_{arm}_rl.yaml",
                    parent_kind="task", parent_task=sft_id,
                    sample_count=8, max_steps=1,
                ),
            ])
        tasks.extend([
            Task(
                "smoke_base_rule_rl4000", "rule_rl",
                dataset_yaml="grpo/pathvlm_ratio_smoke_base_rule_rl4000.yaml",
                parent_kind="base", sample_count=8, max_steps=1,
            ),
            Task(
                "smoke_stage2_continue_rule_rl1000", "rule_rl",
                dataset_yaml="grpo/pathvlm_ratio_smoke_stage2_continue_rule_rl1000.yaml",
                parent_kind="stage2", sample_count=8, max_steps=1,
            ),
        ])
        return tasks

    tasks = []
    for arm in ("sft0750_rl0250", "sft0500_rl0500", "sft0250_rl0750"):
        sft = arms[arm]["sft"]
        rl = arms[arm]["rl"]
        sft_id = f"{arm}_sft"
        tasks.extend([
            Task(
                sft_id, "sft", dataset_name=sft["dataset_name"],
                sample_count=int(sft["qa_count"]), epochs=10,
            ),
            Task(
                f"{arm}_rule_rl", "rule_rl", dataset_yaml=rl["dataset_yaml"],
                parent_kind="task", parent_task=sft_id,
                sample_count=int(rl["qa_count"]),
                max_steps={250: 375, 500: 750, 750: 1125}[int(rl["qa_count"])],
            ),
        ])
    tasks.extend([
        Task(
            "stage2_continue_rule_rl1000", "rule_rl",
            dataset_yaml="grpo/pathvlm_ratio_stage2_continue_rule_rl1000.yaml",
            parent_kind="stage2", sample_count=1000, max_steps=1500,
        ),
        Task(
            "base_rule_rl4000_gate50", "rule_rl",
            dataset_yaml="grpo/pathvlm_ratio_base_rule_rl4000.yaml",
            parent_kind="base", sample_count=4000, max_steps=50, gate_only=True,
        ),
        Task(
            "base_rule_rl4000", "rule_rl",
            dataset_yaml="grpo/pathvlm_ratio_base_rule_rl4000.yaml",
            parent_kind="base", sample_count=4000, max_steps=6000,
        ),
    ])
    return tasks


def clean_training_env(extra: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy",
        "OPENROUTER_API_KEY", "AIGCBEST_API_KEY", "PATHVLM_STAGE3_JUDGE_BACKEND",
        "PATHVLM_STAGE3_JUDGE_MODEL", "PATHVLM_STAGE3_JUDGE_CACHE",
    ):
        env.pop(key, None)
    env.update({
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "WANDB_MODE": "disabled",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        **extra,
    })
    return env


def run_logged(command: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{now_iso()}] COMMAND {shlex.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            command, cwd=cwd, env=env, text=True,
            stdout=handle, stderr=subprocess.STDOUT,
        )
        previous_handlers = {}

        def forward(signum, _frame):
            if process.poll() is None:
                process.send_signal(signum)

        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, forward)
        try:
            return int(process.wait())
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def reward_audit_summary(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.glob("rank_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    types = Counter(str(row.get("reward_type")) for row in rows)
    rewards = [float(row["reward"]) for row in rows]
    if not rows or not {"accuracy", "format"}.issubset(types):
        raise SequenceStop(f"rule-RL reward audit is incomplete: {root}")
    if any(row.get("reward_type") == "process" for row in rows):
        raise SequenceStop("process/Judge reward appeared in a rule-only arm")
    return {
        "event_count": len(rows),
        "reward_types": dict(sorted(types.items())),
        "minimum_reward": min(rewards),
        "maximum_reward": max(rewards),
        "nonzero_count": sum(value != 0 for value in rewards),
    }


def gate_rule_rl(task: Task, task_dir: Path, output: Path, reward_dir: Path) -> dict[str, Any]:
    audit_path = output / "pathvlm_ratio_train_state_audit.json"
    if not audit_path.is_file():
        raise SequenceStop(f"missing train-state audit: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    global_step = int(audit.get("global_step", -1))
    if task.max_steps is not None and global_step != task.max_steps:
        raise SequenceStop(f"{task.task_id} reached step {global_step}, expected {task.max_steps}")
    if audit.get("trainability", {}).get("passed") is not True:
        raise SequenceStop(f"trainability/freeze gate failed for {task.task_id}")
    metrics = audit.get("train_metrics", {})
    numeric = [float(value) for value in metrics.values() if isinstance(value, (int, float))]
    if numeric and not all(math.isfinite(value) for value in numeric):
        raise SequenceStop(f"non-finite training metric in {task.task_id}")
    rewards = reward_audit_summary(reward_dir)
    if task.gate_only and rewards["minimum_reward"] == rewards["maximum_reward"]:
        raise SequenceStop("0+4000 gate has zero observed reward variance")
    return {"model": model_files(output), "train_state": str(audit_path), "rewards": rewards}


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.mode = args.mode
        self.repo = args.repo_root.resolve()
        self.install = args.install_root.resolve()
        self.protocol_path = args.protocol.resolve()
        self.protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        self.data_root = Path(self.protocol["prepared_data"]["root"]).resolve()
        self.data_manifest_path = Path(self.protocol["prepared_data"]["manifest"]).resolve()
        self.data_manifest = json.loads(self.data_manifest_path.read_text(encoding="utf-8"))
        self.root = self.install / "runs/data_ratio_rule_rl_ablation_v1" / f"{self.mode}_sequence"
        self.state_path = self.root / "state.json"
        self.sft_python = self.install / "envs/sft/bin/python"
        self.grpo_python = self.install / "envs/grpo/bin/python"
        self.llamafactory = self.install / "sources/LLaMA-Factory"
        self.base = Path(self.protocol["base_model"]["path"]).resolve()
        self.stage2 = Path(self.protocol["stage2_parent"]["path"]).resolve()
        self.tasks = build_tasks(self.mode, self.data_manifest)
        self.current_process: subprocess.Popen | None = None

    def preflight(self) -> dict[str, Any]:
        if self.protocol.get("test_accessed") is not False:
            raise SequenceStop("protocol does not keep test inaccessible")
        if sha256_file(self.data_manifest_path) != self.protocol["prepared_data"]["manifest_sha256"]:
            raise SequenceStop("prepared data manifest hash mismatch")
        data_gates = self.data_manifest.get("gates", {})
        failed_data_gates = {
            key: value for key, value in data_gates.items()
            if key != "test_accessed" and value is not True
        }
        if failed_data_gates or data_gates.get("test_accessed") is not False:
            raise SequenceStop(f"prepared data gates failed: {data_gates}")
        if self.data_manifest.get("test_accessed") is not False:
            raise SequenceStop("prepared data touched test")
        if self.protocol["base_model"]["revision"] != BASE_REVISION:
            raise SequenceStop("base revision changed")
        base = model_files(self.base)
        stage2 = model_files(self.stage2)
        stage2_manifest = self.stage2 / "snapshot_manifest.json"
        if sha256_file(stage2_manifest) != self.protocol["stage2_parent"]["manifest_sha256"]:
            raise SequenceStop("Stage2 parent snapshot manifest hash mismatch")
        for binary in (self.sft_python, self.grpo_python):
            if not os.access(binary, os.X_OK):
                raise SequenceStop(f"missing executable environment: {binary}")
        launcher = self.llamafactory / "src/llamafactory/launcher.py"
        if not launcher.is_file():
            raise SequenceStop(f"missing LLaMA-Factory launcher: {launcher}")
        preflight_report = self.install / "reports/preflight_report.json"
        if json.loads(preflight_report.read_text(encoding="utf-8")).get("passed") is not True:
            raise SequenceStop("formal A100 preflight is not passing")
        disk_free = shutil.disk_usage(self.install).free
        if disk_free < DISK_RESERVE_BYTES:
            raise SequenceStop(f"disk reserve gate failed: {disk_free} < {DISK_RESERVE_BYTES}")
        require_port_free(SFT_MASTER_PORT)
        require_port_free(RL_MASTER_PORT)
        hardware = gpu_preflight()
        return {
            "passed": True,
            "timestamp": now_iso(),
            "mode": self.mode,
            "protocol": str(self.protocol_path),
            "protocol_sha256": sha256_file(self.protocol_path),
            "data_manifest": str(self.data_manifest_path),
            "data_manifest_sha256": sha256_file(self.data_manifest_path),
            "base": base,
            "stage2": stage2,
            "disk_free_bytes": disk_free,
            "hardware": hardware,
            "task_order": [task.task_id for task in self.tasks],
            "test_accessed": False,
        }

    def load_state(self, preflight: dict[str, Any]) -> dict[str, Any]:
        if self.state_path.is_file():
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if state.get("mode") != self.mode:
                raise SequenceStop("state mode mismatch")
            if state.get("protocol_sha256") != preflight["protocol_sha256"]:
                raise SequenceStop("protocol changed after sequence state was created")
            if state.get("data_manifest_sha256") != preflight["data_manifest_sha256"]:
                raise SequenceStop("data manifest changed after sequence state was created")
            return state
        state = {
            "schema_version": 1,
            "sequence_id": f"data_ratio_rule_rl_ablation_v1_{self.mode}",
            "mode": self.mode,
            "status": "prepared",
            "created_at": now_iso(),
            "protocol_sha256": preflight["protocol_sha256"],
            "data_manifest_sha256": preflight["data_manifest_sha256"],
            "supervisor_pid": os.getpid(),
            "task_order": [task.task_id for task in self.tasks],
            "tasks": {},
            "preflight": preflight,
            "test_accessed": False,
        }
        atomic_json(self.state_path, state)
        return state

    def task_output(self, task_id: str) -> Path:
        return self.root / "tasks" / task_id / "output"

    def resolve_parent(self, task: Task) -> Path:
        if task.parent_kind == "base":
            return self.base
        if task.parent_kind == "stage2":
            return self.stage2
        if task.parent_kind == "task" and task.parent_task:
            output = self.task_output(task.parent_task)
            model_files(output)
            return output
        raise SequenceStop(f"invalid parent contract for {task.task_id}")

    def run_sft(self, task: Task, task_dir: Path, attempt: int) -> dict[str, Any]:
        output = task_dir / "output"
        completed_state = output / "trainer_state.json"
        if completed_state.is_file():
            trainer_state = json.loads(completed_state.read_text(encoding="utf-8"))
            reached = (
                task.max_steps is not None
                and int(trainer_state.get("global_step", -1)) == task.max_steps
            ) or (
                task.epochs is not None
                and float(trainer_state.get("epoch", 0.0)) >= task.epochs - 0.01
            )
            if reached:
                model = model_files(output)
                return {
                    "output": str(output), "model": model,
                    "trainer_state": str(completed_state), "resumed_from": None,
                    "recovered_completed_output_without_retraining": True,
                }
        resume = latest_checkpoint(output)
        config = {
            "model_name_or_path": str(self.base),
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
            "deepspeed": str(self.repo / "configs/deepspeed/ds_z2_gpu_torch_adamw.json"),
            "dataset_dir": str(self.data_root / "llamafactory"),
            "dataset": task.dataset_name,
            "template": "qwen2_vl",
            "cutoff_len": 512,
            "max_samples": task.sample_count,
            "overwrite_cache": True,
            "preprocessing_num_workers": 16,
            "output_dir": str(output),
            "logging_steps": 1,
            "save_strategy": "steps",
            "save_steps": 1 if self.mode == "smoke" else 100,
            "save_total_limit": 1 if self.mode == "smoke" else 2,
            "save_only_model": self.mode == "smoke",
            "plot_loss": True,
            "overwrite_output_dir": False,
            "report_to": "none",
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 2.0e-5,
            "lr_scheduler_type": "constant" if self.mode == "smoke" else "cosine",
            "warmup_ratio": 0.0 if self.mode == "smoke" else 0.03,
            "bf16": True,
            "gradient_checkpointing": True,
            "disable_gradient_checkpointing": False,
            "optim": "adamw_torch_fused",
            "seed": 42,
            "data_seed": 42,
            "ddp_timeout": 180000000,
        }
        if task.max_steps is not None:
            config["max_steps"] = task.max_steps
            config["num_train_epochs"] = 1
        else:
            config["num_train_epochs"] = task.epochs
        if resume is not None:
            config["resume_from_checkpoint"] = str(resume)
        config_path = task_dir / f"resolved_config_attempt{attempt:02d}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        command = [
            str(self.sft_python), "-m", "torch.distributed.run",
            "--nproc_per_node=8", f"--master_port={SFT_MASTER_PORT}",
            str(self.llamafactory / "src/llamafactory/launcher.py"), str(config_path),
        ]
        env = clean_training_env({})
        rc = run_logged(
            command, cwd=self.llamafactory, env=env,
            log=task_dir / f"train_attempt{attempt:02d}.log",
        )
        if rc:
            raise SequenceStop(f"SFT task {task.task_id} failed with exit {rc}")
        model = model_files(output)
        state_path = output / "trainer_state.json"
        if not state_path.is_file():
            raise SequenceStop(f"SFT task has no trainer state: {task.task_id}")
        trainer_state = json.loads(state_path.read_text(encoding="utf-8"))
        if task.max_steps is not None and int(trainer_state.get("global_step", -1)) != task.max_steps:
            raise SequenceStop(f"SFT smoke step mismatch for {task.task_id}")
        if task.epochs is not None and float(trainer_state.get("epoch", 0.0)) < task.epochs - 0.01:
            raise SequenceStop(f"SFT epoch mismatch for {task.task_id}")
        return {
            "output": str(output), "model": model,
            "trainer_state": str(state_path), "resumed_from": str(resume) if resume else None,
            "config": str(config_path), "config_sha256": sha256_file(config_path),
        }

    def run_rule_rl(self, task: Task, task_dir: Path, attempt: int) -> dict[str, Any]:
        parent = self.resolve_parent(task)
        output = task_dir / "output"
        alias = task_dir / "parent_Qwen2.5-VL-7B-Instruct"
        if alias.is_symlink():
            if alias.resolve() != parent.resolve():
                raise SequenceStop(f"parent alias changed for {task.task_id}")
        elif alias.exists():
            raise SequenceStop(f"parent alias path is not a symlink: {alias}")
        else:
            alias.parent.mkdir(parents=True, exist_ok=True)
            alias.symlink_to(parent, target_is_directory=True)
        dataset_yaml = (self.data_root / str(task.dataset_yaml)).resolve()
        if not dataset_yaml.is_file() or "test" in str(dataset_yaml).lower():
            raise SequenceStop(f"invalid rule-RL dataset: {dataset_yaml}")
        completed_audit = output / "pathvlm_ratio_train_state_audit.json"
        if completed_audit.is_file():
            audit = json.loads(completed_audit.read_text(encoding="utf-8"))
            if int(audit.get("global_step", -1)) == task.max_steps:
                for previous_reward_dir in sorted(
                    task_dir.glob("online_reward_events_attempt*"), reverse=True
                ):
                    try:
                        result = gate_rule_rl(
                            task, task_dir, output, previous_reward_dir
                        )
                    except SequenceStop:
                        continue
                    result.update({
                        "output": str(output), "parent": str(parent),
                        "resumed_from": None,
                        "dataset_yaml": str(dataset_yaml),
                        "dataset_yaml_sha256": sha256_file(dataset_yaml),
                        "recovered_completed_output_without_retraining": True,
                        "reward_audit_reused": str(previous_reward_dir),
                    })
                    return result
        resume = latest_checkpoint(output)
        command = [
            str(self.grpo_python), "-m", "torch.distributed.run",
            "--nproc_per_node=8", f"--master_port={RL_MASTER_PORT}",
            str(self.repo / "scripts/grpo_pathmmu.py"),
            "--deepspeed", str(self.repo / "configs/deepspeed/ds_z3_optimizer_offload_torch_adamw.json"),
            "--output_dir", str(output),
            "--model_name_or_path", str(alias),
            "--dataset_name", str(dataset_yaml), "--image_root", "/",
            "--reward_funcs", "accuracy", "format",
            "--freeze_vision_modules", "true",
            "--max_pixels", "65536", "--min_pixels", "3136",
            "--num_generations", "4", "--max_completion_length", "192",
            "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "1",
            "--learning_rate", "1.0e-6", "--logging_steps", "1",
            "--bf16", "true", "--torch_dtype", "bfloat16",
            "--gradient_checkpointing", "true", "--attn_implementation", "sdpa",
            "--beta", "0.04", "--num_iterations", "1",
            "--save_strategy", "steps",
            "--save_steps", "1" if self.mode == "smoke" else "100",
            "--save_total_limit", "1" if self.mode == "smoke" else "2",
            "--save_only_model", "true" if self.mode == "smoke" else "false",
            "--report_to", "none", "--seed", "42", "--data_seed", "42",
            "--remove_unused_columns", "false", "--max_steps", str(task.max_steps),
        ]
        reward_dir = task_dir / f"online_reward_events_attempt{attempt:02d}"
        env_extra = {
            "PATHVLM_REWARD_LOG_DIR": str(reward_dir),
            "PATHVLM_TRAINING_SEGMENT": task.task_id,
            "PATHVLM_TRAIN_STATE_AUDIT_NAME": "pathvlm_ratio_train_state_audit.json",
            "PATHVLM_REQUIRE_SOURCE_AUDIT": "true",
            "PATHVLM_IMAGE_HASH_MANIFEST": str(
                self.repo / "data/pathmmu_image_disjoint_v2/image_content_sha256.json"
            ),
            "DEBUG_MODE": "false",
        }
        if resume is not None:
            env_extra["PATHVLM_RESUME_FROM_CHECKPOINT"] = str(resume)
        env = clean_training_env(env_extra)
        rc = run_logged(
            command, cwd=self.repo / "vendor/open-r1-multimodal", env=env,
            log=task_dir / f"train_attempt{attempt:02d}.log",
        )
        if rc:
            raise SequenceStop(f"rule-RL task {task.task_id} failed with exit {rc}")
        result = gate_rule_rl(task, task_dir, output, reward_dir)
        result.update({
            "output": str(output), "parent": str(parent),
            "resumed_from": str(resume) if resume else None,
            "dataset_yaml": str(dataset_yaml), "dataset_yaml_sha256": sha256_file(dataset_yaml),
        })
        return result

    def execute_task(self, task: Task, attempt: int) -> dict[str, Any]:
        task_dir = self.root / "tasks" / task.task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        gpu_preflight()
        require_port_free(SFT_MASTER_PORT if task.stage == "sft" else RL_MASTER_PORT)
        if shutil.disk_usage(self.install).free < DISK_RESERVE_BYTES:
            raise SequenceStop("disk reserve gate failed before task launch")
        (task_dir / f"task_contract_attempt{attempt:02d}.json").write_text(
            json.dumps(asdict(task), indent=2) + "\n", encoding="utf-8"
        )
        if task.stage == "sft":
            return self.run_sft(task, task_dir, attempt)
        if task.stage == "rule_rl":
            return self.run_rule_rl(task, task_dir, attempt)
        raise SequenceStop(f"unsupported task stage: {task.stage}")

    def run(self, preflight_only: bool = False) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        lock = acquire_lock(self.root / "sequence.lock")
        try:
            preflight = self.preflight()
            atomic_json(self.root / "preflight_report.json", preflight)
            if preflight_only:
                print(json.dumps(preflight, indent=2))
                return
            state = self.load_state(preflight)
            state.update({"status": "running", "supervisor_pid": os.getpid(), "updated_at": now_iso()})
            atomic_json(self.state_path, state)
            for task in self.tasks:
                entry = state["tasks"].get(task.task_id, {})
                if entry.get("status") == "completed":
                    model_files(self.task_output(task.task_id))
                    continue
                attempt = int(entry.get("attempt", 0)) + 1
                state["current_task"] = task.task_id
                state["tasks"][task.task_id] = {
                    "status": "running", "attempt": attempt,
                    "started_at": now_iso(), "contract": asdict(task),
                }
                state["updated_at"] = now_iso()
                atomic_json(self.state_path, state)
                try:
                    result = self.execute_task(task, attempt)
                except Exception as exc:
                    state["status"] = "failed"
                    state["tasks"][task.task_id].update({
                        "status": "failed", "failed_at": now_iso(),
                        "failure": {"type": type(exc).__name__, "message": str(exc)},
                    })
                    state["updated_at"] = now_iso()
                    atomic_json(self.state_path, state)
                    raise
                state["tasks"][task.task_id].update({
                    "status": "completed", "completed_at": now_iso(), "result": result,
                })
                state["updated_at"] = now_iso()
                atomic_json(self.state_path, state)
            state.update({
                "status": "completed", "completed_at": now_iso(),
                "current_task": None, "test_accessed": False,
            })
            atomic_json(self.state_path, state)
            print(f"[data-ratio-sequence] COMPLETE {self.state_path}", flush=True)
        finally:
            lock.close()


def run_task_plan_for_test(
    task_ids: list[str], state_path: Path,
    execute: Callable[[str, int], dict[str, Any]],
    validate: Callable[[str, dict[str, Any]], None],
) -> dict[str, Any]:
    """Small dependency-free state machine used by regression tests."""
    state = json.loads(state_path.read_text()) if state_path.is_file() else {"status": "prepared", "tasks": {}}
    state["status"] = "running"
    atomic_json(state_path, state)
    for task_id in task_ids:
        entry = state["tasks"].get(task_id, {})
        if entry.get("status") == "completed":
            validate(task_id, entry["result"])
            continue
        attempt = int(entry.get("attempt", 0)) + 1
        state["tasks"][task_id] = {"status": "running", "attempt": attempt}
        atomic_json(state_path, state)
        try:
            result = execute(task_id, attempt)
            validate(task_id, result)
        except Exception as exc:
            state["status"] = "failed"
            state["tasks"][task_id].update({
                "status": "failed", "failure": {"type": type(exc).__name__, "message": str(exc)}
            })
            atomic_json(state_path, state)
            raise
        state["tasks"][task_id].update({"status": "completed", "result": result})
        atomic_json(state_path, state)
    state["status"] = "completed"
    atomic_json(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--install-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    Runner(args).run(preflight_only=args.preflight_only)


if __name__ == "__main__":
    main()
