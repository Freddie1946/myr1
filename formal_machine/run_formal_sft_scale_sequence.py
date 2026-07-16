#!/usr/bin/env python3
"""Fail-closed supervisor for the formal seed-42 SFT scale sequence."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import yaml


COUNTS = (500, 1000, 2000, 3000)
FORMAL_GPU_IDS = list(range(8))
MIN_FREE_DISK_BYTES = 500 * 1024**3
BASE_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
CORE_MANIFEST_NAME = "code_hash_manifest_20260717_010146.json"
SEQUENCE_MANIFEST_NAME = "sft_scale_sequence_manifest_20260715_235000.json"
SFT_GATES = {
    "training_completed",
    "final_checkpoint_saved",
    "final_checkpoint_reloaded",
    "trainability_freeze_gate",
    "losses_finite",
    "gradients_finite_nonzero",
    "language_tensor_changed",
    "visual_tensor_exactly_equal",
    "gradient_checkpointing_observed",
    "all_epoch_model_snapshots_saved",
    "all_epoch_snapshot_metadata_present",
    "latest_two_full_resume_checkpoints_retained",
    "disk_reserve_maintained",
    "test_not_accessed",
}
VALIDATION_GATES = {
    "inference_completed",
    "exact_validation_count",
    "deterministic_decoding",
    "test_not_accessed",
    "raw_predictions_saved",
}
TERMINAL_FAILURE_STATES = {"failed", "failed_gate", "aborted", "cancelled"}


class SequenceStop(RuntimeError):
    """A condition that must stop automatic progression."""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise SequenceStop(f"manifest is missing: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SequenceStop(f"manifest is not a mapping: {path}")
    return payload


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def require_within(path: Path, root: Path, label: str) -> None:
    path = path.resolve()
    root = root.resolve()
    if path != root and root not in path.parents:
        raise SequenceStop(f"{label} is outside its allowed root: {path} not under {root}")


def require_true_gates(payload: dict, required: set[str], label: str) -> None:
    gates = payload.get("gates")
    if not isinstance(gates, dict):
        raise SequenceStop(f"{label} gates are missing")
    missing = sorted(required.difference(gates))
    false_gates = {key: gates.get(key) for key in sorted(required) if gates.get(key) is not True}
    unexpected_false = {key: value for key, value in gates.items() if value is not True}
    if missing or false_gates or unexpected_false:
        raise SequenceStop(
            f"{label} gates are not all true: missing={missing}, "
            f"required_failures={false_gates}, all_failures={unexpected_false}"
        )


def validate_sft_manifest(path: Path, install: Path, expected_count: int) -> tuple[dict, Path]:
    path = path.resolve()
    expected_root = (
        install / "runs" / "stage1_sft" / f"n{expected_count:04d}_seed0042"
    ).resolve()
    require_within(path, expected_root, "SFT manifest")
    if path.name != "run_manifest.yaml" or path.parent.parent != expected_root:
        raise SequenceStop(f"unexpected SFT manifest layout: {path}")
    payload = read_yaml(path)
    expected_run_prefix = f"formal_sft_n{expected_count:04d}_seed0042_"
    if not str(payload.get("run_id", "")).startswith(expected_run_prefix):
        raise SequenceStop(f"unexpected SFT run id: {payload.get('run_id')}")
    fixed = {
        "status": "completed",
        "stage": "stage1_sft",
        "formal_result": True,
        "test_accessed": False,
    }
    mismatches = {key: {"expected": value, "actual": payload.get(key)}
                  for key, value in fixed.items() if payload.get(key) != value}
    if mismatches:
        raise SequenceStop(f"SFT manifest terminal mismatch: {mismatches}")
    data = payload.get("data", {})
    if data.get("version") != "pathmmu_image_disjoint_v1":
        raise SequenceStop(f"unexpected SFT data version: {data.get('version')}")
    if data.get("dataset") != f"pathvlm_sft_n{expected_count:04d}":
        raise SequenceStop(f"unexpected SFT dataset: {data.get('dataset')}")
    if data.get("qa_count") != expected_count or data.get("all_image_paths_exist") is not True:
        raise SequenceStop(f"SFT data count/image gate failed: {data}")
    if "test" in str(data.get("adapter", "")).lower():
        raise SequenceStop("SFT adapter path contains a forbidden test reference")
    model = payload.get("model", {})
    if model.get("base_id") != "Qwen/Qwen2.5-VL-7B-Instruct":
        raise SequenceStop(f"unexpected SFT base model: {model.get('base_id')}")
    if model.get("base_revision") != BASE_REVISION:
        raise SequenceStop(f"unexpected SFT base revision: {model.get('base_revision')}")
    training = payload.get("training", {})
    expected_training = {
        "seed": 42,
        "epochs": 10,
        "learning_rate": 2e-5,
        "finetuning_type": "full",
        "language_model_trainable": True,
        "vision_tower_frozen": True,
        "multimodal_projector_frozen": True,
        "backend": "deepspeed_zero2_gpu_fused_adamw_gc",
    }
    training_mismatches = {
        key: {"expected": value, "actual": training.get(key)}
        for key, value in expected_training.items()
        if training.get(key) != value
    }
    if training_mismatches:
        raise SequenceStop(f"SFT training mismatch: {training_mismatches}")
    hardware = payload.get("hardware", {})
    if hardware.get("cuda_visible_devices") != FORMAL_GPU_IDS or hardware.get("gpu_count") != 8:
        raise SequenceStop(f"SFT hardware topology mismatch: {hardware}")
    require_true_gates(payload, SFT_GATES, "SFT")
    outputs = payload.get("outputs", {})
    checkpoint = Path(outputs.get("final_checkpoint", "")).resolve()
    expected_checkpoint = (path.parent / "output").resolve()
    if checkpoint != expected_checkpoint:
        raise SequenceStop(f"SFT final checkpoint mismatch: {checkpoint} != {expected_checkpoint}")
    if Path(outputs.get("run_dir", "")).resolve() != path.parent:
        raise SequenceStop("SFT output run directory does not match its manifest directory")
    for required_file in ("model.safetensors.index.json", "trainer_state.json", "config.json"):
        if not (checkpoint / required_file).is_file():
            raise SequenceStop(f"SFT final checkpoint is missing {required_file}: {checkpoint}")
    snapshot_root = Path(outputs.get("epoch_snapshots", "")).resolve()
    expected_snapshot_root = (path.parent / "epoch_snapshots").resolve()
    if snapshot_root != expected_snapshot_root:
        raise SequenceStop(
            f"SFT epoch snapshot root mismatch: {snapshot_root} != {expected_snapshot_root}"
        )
    snapshots = sorted(snapshot_root.glob("checkpoint-*"))
    if len(snapshots) != 10 or any(
        not (snapshot / "snapshot_manifest.json").is_file() for snapshot in snapshots
    ):
        raise SequenceStop(f"SFT epoch snapshots are incomplete: {snapshot_root}")
    return payload, checkpoint


def validate_validation_manifest(
    path: Path, install: Path, parent_manifest: Path, checkpoint: Path
) -> dict:
    path = path.resolve()
    parent_manifest = parent_manifest.resolve()
    checkpoint = checkpoint.resolve()
    expected_root = (parent_manifest.parent / "validation").resolve()
    require_within(path, expected_root, "validation manifest")
    if path.name != "run_manifest.yaml" or path.parent.parent != expected_root:
        raise SequenceStop(f"unexpected validation manifest layout: {path}")
    payload = read_yaml(path)
    fixed = {
        "status": "completed",
        "stage": "stage1_validation",
        "formal_result": True,
        "test_accessed": False,
    }
    mismatches = {key: {"expected": value, "actual": payload.get(key)}
                  for key, value in fixed.items() if payload.get(key) != value}
    if mismatches:
        raise SequenceStop(f"validation terminal mismatch: {mismatches}")
    parent = payload.get("parent", {})
    if Path(parent.get("manifest", "")).resolve() != parent_manifest:
        raise SequenceStop("validation parent manifest path mismatch")
    if parent.get("manifest_sha256") != sha256(parent_manifest):
        raise SequenceStop("validation parent manifest hash mismatch")
    if Path(parent.get("checkpoint", "")).resolve() != checkpoint:
        raise SequenceStop("validation checkpoint path mismatch")
    data = payload.get("data", {})
    if (
        data.get("version") != "pathmmu_image_disjoint_v1"
        or data.get("split") != "validation_0385"
        or data.get("count") != 385
    ):
        raise SequenceStop(f"validation data mismatch: {data}")
    if "test" in str(data.get("path", "")).lower():
        raise SequenceStop("validation data path contains a forbidden test reference")
    generation = payload.get("generation", {})
    if generation.get("do_sample") is not False or generation.get("max_new_tokens") != 192:
        raise SequenceStop(f"validation decoding mismatch: {generation}")
    require_true_gates(payload, VALIDATION_GATES, "validation")
    metrics = payload.get("metrics", {})
    if metrics.get("count") != 385 or metrics.get("do_sample") is not False:
        raise SequenceStop(f"validation metrics mismatch: {metrics}")
    if metrics.get("test_accessed") is not False:
        raise SequenceStop("validation metrics report test access")
    outputs = payload.get("outputs", {})
    metrics_path = Path(outputs.get("metrics", "")).resolve()
    predictions_path = Path(outputs.get("predictions", "")).resolve()
    require_within(metrics_path, path.parent, "validation metrics")
    require_within(predictions_path, path.parent, "validation predictions")
    if not metrics_path.is_file() or not predictions_path.is_file():
        raise SequenceStop("validation outputs are missing")
    with predictions_path.open(encoding="utf-8") as handle:
        prediction_count = sum(1 for line in handle if line.strip())
    if prediction_count != 385:
        raise SequenceStop(f"validation raw prediction count is {prediction_count}, expected 385")
    return payload


def acquire_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise SequenceStop(f"another scale supervisor holds the lock: {path}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} started_at={now_iso()}\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle


class Supervisor:
    def __init__(self, args: argparse.Namespace):
        self.repo = args.repo_root.resolve()
        self.install = args.install_root.resolve()
        self.current_manifest = args.current_n0500_manifest.resolve()
        self.poll_seconds = args.poll_seconds
        self.validation_gpu = args.validation_gpu
        self.control = self.install / "control" / "sft_scale_seed0042"
        self.state_path = self.control / "sequence_state.json"
        self.log_path = self.control / "supervisor.jsonl"
        self.command_logs = self.control / "command_logs"
        self.command_logs.mkdir(parents=True, exist_ok=True)
        self.lock_handle = acquire_lock(self.control / "supervisor.lock")
        self.expected_git_commit = self.git_commit()
        if self.state_path.is_file():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state.get("status") in {"completed", "stopped"}:
                raise SequenceStop(
                    f"existing sequence is terminal ({self.state.get('status')}): {self.state_path}"
                )
            recorded_commit = self.state.get("expected_git_commit")
            if recorded_commit != self.expected_git_commit:
                raise SequenceStop(
                    f"Git commit changed since sequence start: {recorded_commit} != {self.expected_git_commit}"
                )
            recorded_current = self.state.get("runs", {}).get("500", {}).get("sft_manifest")
            if not recorded_current or Path(recorded_current).resolve() != self.current_manifest:
                raise SequenceStop(
                    "resume current-manifest argument does not exactly match the recorded n=500 run"
                )
            if self.state.get("matrix") != list(COUNTS):
                raise SequenceStop(f"recorded sequence matrix changed: {self.state.get('matrix')}")
            if self.state.get("validation_gpu") != self.validation_gpu:
                raise SequenceStop(
                    f"validation GPU changed on resume: {self.state.get('validation_gpu')} "
                    f"!= {self.validation_gpu}"
                )
        else:
            sequence_id = "formal_sft_scale_seed0042_" + datetime.now().astimezone().strftime(
                "%Y%m%d_%H%M%S"
            )
            self.state = {
                "schema_version": 1,
                "sequence_id": sequence_id,
                "status": "running",
                "created_at": now_iso(),
                "updated_at": now_iso(),
                "supervisor_pid": os.getpid(),
                "host": socket.gethostname(),
                "expected_git_commit": self.expected_git_commit,
                "core_code_manifest": str(self.repo / "protocol" / CORE_MANIFEST_NAME),
                "sequence_code_manifest": str(
                    self.repo / "protocol" / SEQUENCE_MANIFEST_NAME
                ),
                "matrix": list(COUNTS),
                "validation_split": "validation_0385",
                "validation_gpu": self.validation_gpu,
                "runs": {
                    "500": {"sft_manifest": str(self.current_manifest), "source": "attached"}
                },
                "events": [],
                "test_accessed": False,
            }
            self.save_state()

    def log(self, event: str, **details) -> None:
        payload = {"timestamp": now_iso(), "event": event, **details}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def save_state(self, **updates) -> None:
        self.state.update(updates)
        self.state["updated_at"] = now_iso()
        self.state["supervisor_pid"] = os.getpid()
        atomic_write_json(self.state_path, self.state)

    def event(self, name: str, **details) -> None:
        entry = {"timestamp": now_iso(), "event": name, **details}
        self.state.setdefault("events", []).append(entry)
        self.save_state(last_event=entry)
        self.log(name, **details)

    def git_commit(self) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"],
            text=True, capture_output=True, check=False,
        )
        if result.returncode:
            raise SequenceStop(f"cannot read Git commit: {result.stderr.strip()}")
        return result.stdout.strip()

    def verify_integrity(self) -> None:
        status = subprocess.run(
            ["git", "-C", str(self.repo), "status", "--porcelain"],
            text=True, capture_output=True, check=False,
        )
        if status.returncode or status.stdout.strip():
            raise SequenceStop(f"repository is not clean: {status.stdout.strip()} {status.stderr.strip()}")
        if self.git_commit() != self.expected_git_commit:
            raise SequenceStop("Git commit changed while sequence was running")
        verifier = self.repo / "scripts" / "verify_code_hash_manifest.py"
        for manifest_name in (CORE_MANIFEST_NAME, SEQUENCE_MANIFEST_NAME):
            manifest = self.repo / "protocol" / manifest_name
            result = subprocess.run(
                [sys.executable, str(verifier), "--repo-root", str(self.repo),
                 "--manifest", str(manifest)],
                text=True, capture_output=True, check=False,
            )
            if result.returncode:
                raise SequenceStop(
                    f"code manifest verification failed for {manifest}: "
                    f"{result.stdout}{result.stderr}"
                )
        if shutil.disk_usage(self.install).free < MIN_FREE_DISK_BYTES:
            raise SequenceStop("less than 500 GiB free; automatic sequence stopped")

    def gpu_state(self) -> tuple[list[dict], list[dict]]:
        inventory = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=False,
        )
        if inventory.returncode:
            raise SequenceStop(f"nvidia-smi inventory failed: {inventory.stderr.strip()}")
        rows = []
        for line in inventory.stdout.splitlines():
            index, name, total, used, free, util = [item.strip() for item in line.split(",")]
            rows.append({
                "index": int(index), "name": name, "memory_total_mib": int(total),
                "memory_used_mib": int(used), "memory_free_mib": int(free),
                "utilization_percent": int(util),
            })
        if [row["index"] for row in rows] != FORMAL_GPU_IDS:
            raise SequenceStop(f"formal GPU inventory changed: {rows}")
        processes = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=False,
        )
        if processes.returncode:
            raise SequenceStop(f"nvidia-smi process query failed: {processes.stderr.strip()}")
        process_rows = []
        for line in processes.stdout.splitlines():
            if not line.strip():
                continue
            uuid, pid, name, used = [item.strip() for item in line.split(",")]
            process_rows.append({
                "gpu_uuid": uuid, "pid": int(pid), "process_name": name,
                "used_memory_mib": int(used),
            })
        return rows, process_rows

    def wait_for_idle_gpus(self, next_stage: str) -> None:
        self.event("waiting_for_idle_gpus", next_stage=next_stage)
        last_processes = None
        while True:
            self.verify_integrity()
            gpus, processes = self.gpu_state()
            if not processes:
                self.event("all_gpus_idle", next_stage=next_stage, gpus=gpus)
                return
            if processes != last_processes:
                self.log("gpu_wait_blocked", next_stage=next_stage, processes=processes)
                last_processes = processes
            self.save_state(
                phase="waiting_for_idle_gpus", next_stage=next_stage,
                blocking_gpu_processes=processes,
            )
            time.sleep(self.poll_seconds)

    def latest_progress(self, manifest_path: Path) -> dict | None:
        trainer_log = manifest_path.parent / "output" / "trainer_log.jsonl"
        if not trainer_log.is_file():
            return None
        last = None
        with trainer_log.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.strip():
                    try:
                        last = json.loads(line)
                    except json.JSONDecodeError:
                        continue
        return last

    def wait_sft_terminal(self, manifest_path: Path, count: int) -> tuple[dict, Path]:
        while True:
            payload = read_yaml(manifest_path)
            status = payload.get("status")
            if status == "completed":
                result = validate_sft_manifest(manifest_path, self.install, count)
                self.event("sft_completed", count=count, manifest=str(manifest_path))
                return result
            if status in TERMINAL_FAILURE_STATES or status not in {"running"}:
                raise SequenceStop(
                    f"SFT n={count} entered non-success terminal/unknown state {status}: {manifest_path}"
                )
            self.save_state(
                phase="waiting_for_sft", current_count=count,
                current_manifest=str(manifest_path), progress=self.latest_progress(manifest_path),
            )
            time.sleep(self.poll_seconds)

    def wait_validation_terminal(
        self, manifest_path: Path, parent_manifest: Path, checkpoint: Path, count: int
    ) -> dict:
        while True:
            payload = read_yaml(manifest_path)
            status = payload.get("status")
            if status == "completed":
                result = validate_validation_manifest(
                    manifest_path, self.install, parent_manifest, checkpoint
                )
                self.event(
                    "validation_completed", count=count, manifest=str(manifest_path),
                    metrics=result.get("metrics"),
                )
                return result
            if status in TERMINAL_FAILURE_STATES or status not in {"running"}:
                raise SequenceStop(
                    f"validation for n={count} entered non-success terminal/unknown state "
                    f"{status}: {manifest_path}"
                )
            self.save_state(
                phase="waiting_for_validation", current_count=count,
                current_manifest=str(manifest_path),
            )
            time.sleep(self.poll_seconds)

    def discover_sft_manifests(self, count: int) -> list[Path]:
        root = self.install / "runs" / "stage1_sft" / f"n{count:04d}_seed0042"
        if not root.is_dir():
            return []
        return sorted(root.glob(f"formal_sft_n{count:04d}_seed0042_*/run_manifest.yaml"))

    @staticmethod
    def discover_validation_manifests(parent_manifest: Path) -> list[Path]:
        root = parent_manifest.parent / "validation"
        if not root.is_dir():
            return []
        return sorted(root.glob("validation_0385_*/run_manifest.yaml"))

    def launch_and_discover(
        self, command: list[str], existing: set[Path], discover, label: str
    ) -> tuple[subprocess.Popen, Path]:
        log_path = self.command_logs / f"{label}.log"
        self.save_state(pending_launch={
            "label": label, "command": command, "declared_at": now_iso(),
            "rule": "never retry automatically if no unique manifest can be recovered",
        })
        self.event("launching", label=label, command=command, command_log=str(log_path))
        env = os.environ.copy()
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
            env.pop(key, None)
        with log_path.open("a", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command, cwd=self.repo, env=env, stdout=log_handle, stderr=subprocess.STDOUT,
                text=True, start_new_session=True,
            )
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            candidates = [path.resolve() for path in discover() if path.resolve() not in existing]
            if len(candidates) == 1:
                manifest_path = candidates[0]
                self.save_state(pending_launch=None)
                self.event(
                    "launched", label=label, pid=process.pid, manifest=str(manifest_path)
                )
                return process, manifest_path
            if len(candidates) > 1:
                self.terminate_owned_process_group(process)
                raise SequenceStop(f"multiple new manifests appeared during {label}: {candidates}")
            returncode = process.poll()
            if returncode is not None:
                raise SequenceStop(
                    f"{label} exited with {returncode} before creating exactly one manifest; "
                    f"see {log_path}"
                )
            time.sleep(2)
        self.terminate_owned_process_group(process)
        raise SequenceStop(f"{label} did not create a manifest within 300 seconds")

    def terminate_owned_process_group(self, process: subprocess.Popen) -> None:
        """Terminate only a child process group created by this supervisor."""
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return

    def monitor_child(self, process: subprocess.Popen, manifest: Path, label: str) -> None:
        while True:
            returncode = process.poll()
            if returncode is not None:
                if returncode != 0:
                    raise SequenceStop(f"{label} launcher exited with {returncode}")
                self.event("launcher_exited", label=label, returncode=returncode)
                return
            payload = read_yaml(manifest)
            self.save_state(
                phase=f"running_{label}", current_manifest=str(manifest),
                child_pid=process.pid, child_status=payload.get("status"),
                progress=self.latest_progress(manifest),
            )
            time.sleep(self.poll_seconds)

    def ensure_sft(self, count: int) -> tuple[Path, dict, Path]:
        record = self.state.setdefault("runs", {}).setdefault(str(count), {})
        recorded = record.get("sft_manifest")
        if recorded:
            manifest = Path(recorded).resolve()
            payload, checkpoint = self.wait_sft_terminal(manifest, count)
            return manifest, payload, checkpoint
        existing = self.discover_sft_manifests(count)
        if len(existing) > 1:
            raise SequenceStop(f"multiple SFT manifests already exist for n={count}: {existing}")
        if len(existing) == 1:
            manifest = existing[0].resolve()
            record.update({"sft_manifest": str(manifest), "source": "recovered_existing"})
            self.save_state(pending_launch=None)
            payload, checkpoint = self.wait_sft_terminal(manifest, count)
            return manifest, payload, checkpoint
        pending = self.state.get("pending_launch")
        if pending:
            raise SequenceStop(
                f"unresolved prior launch intent forbids automatic retry: {pending}"
            )
        self.verify_integrity()
        self.wait_for_idle_gpus(f"sft_n{count:04d}")
        config = self.install / "generated_configs" / "sft" / f"sft_n{count:04d}_seed0042.yaml"
        command = [
            "bash", str(self.repo / "scripts" / "launch_formal_sft_run.sh"),
            "--config", str(config),
        ]
        process, manifest = self.launch_and_discover(
            command, set(), lambda: self.discover_sft_manifests(count), f"sft_n{count:04d}"
        )
        record.update({"sft_manifest": str(manifest), "source": "supervisor_launched"})
        self.save_state()
        self.monitor_child(process, manifest, f"sft_n{count:04d}")
        payload, checkpoint = self.wait_sft_terminal(manifest, count)
        return manifest, payload, checkpoint

    def ensure_validation(
        self, count: int, parent_manifest: Path, checkpoint: Path
    ) -> tuple[Path, dict]:
        record = self.state.setdefault("runs", {}).setdefault(str(count), {})
        recorded = record.get("validation_manifest")
        if recorded:
            manifest = Path(recorded).resolve()
            payload = self.wait_validation_terminal(manifest, parent_manifest, checkpoint, count)
            return manifest, payload
        existing = self.discover_validation_manifests(parent_manifest)
        if len(existing) > 1:
            raise SequenceStop(
                f"multiple validation manifests already exist for n={count}: {existing}"
            )
        if len(existing) == 1:
            manifest = existing[0].resolve()
            record.update({"validation_manifest": str(manifest), "source": "recovered_existing"})
            self.save_state(pending_launch=None)
            payload = self.wait_validation_terminal(
                manifest, parent_manifest, checkpoint, count
            )
            return manifest, payload
        pending = self.state.get("pending_launch")
        if pending:
            raise SequenceStop(
                f"unresolved prior launch intent forbids automatic retry: {pending}"
            )
        self.verify_integrity()
        self.wait_for_idle_gpus(f"validation_n{count:04d}")
        command = [
            "bash", str(self.repo / "scripts" / "launch_formal_validation.sh"),
            "--checkpoint", str(checkpoint),
            "--parent-manifest", str(parent_manifest),
            "--gpu", str(self.validation_gpu),
        ]
        process, manifest = self.launch_and_discover(
            command, set(), lambda: self.discover_validation_manifests(parent_manifest),
            f"validation_n{count:04d}",
        )
        record.update({"validation_manifest": str(manifest), "source": "supervisor_launched"})
        self.save_state()
        self.monitor_child(process, manifest, f"validation_n{count:04d}")
        payload = self.wait_validation_terminal(manifest, parent_manifest, checkpoint, count)
        return manifest, payload

    def run(self) -> None:
        self.event("supervisor_started", current_manifest=str(self.current_manifest))
        self.verify_integrity()
        for count in COUNTS:
            self.save_state(current_count=count, phase="sft")
            sft_manifest, _, checkpoint = self.ensure_sft(count)
            self.save_state(current_count=count, phase="validation")
            validation_manifest, validation = self.ensure_validation(
                count, sft_manifest, checkpoint
            )
            record = self.state["runs"][str(count)]
            record.update({
                "sft_manifest": str(sft_manifest),
                "validation_manifest": str(validation_manifest),
                "validation_metrics": validation.get("metrics"),
                "completed": True,
            })
            self.save_state()
        self.event("sequence_completed", counts=list(COUNTS))
        self.save_state(
            status="completed", phase="completed", completed_at=now_iso(),
            test_accessed=False,
        )


def daemonize(args: argparse.Namespace) -> int:
    control = args.install_root.resolve() / "control" / "sft_scale_seed0042"
    control.mkdir(parents=True, exist_ok=True)
    daemon_log = control / "daemon.log"
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--repo-root", str(args.repo_root.resolve()),
        "--install-root", str(args.install_root.resolve()),
        "--current-n0500-manifest", str(args.current_n0500_manifest.resolve()),
        "--poll-seconds", str(args.poll_seconds),
        "--validation-gpu", str(args.validation_gpu),
    ]
    with daemon_log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=args.repo_root.resolve(), stdout=handle, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    time.sleep(2)
    returncode = process.poll()
    if returncode is not None:
        raise SequenceStop(
            f"supervisor daemon exited immediately with {returncode}; see {daemon_log}"
        )
    print(json.dumps({
        "supervisor_pid": process.pid,
        "state": str(control / "sequence_state.json"),
        "log": str(control / "supervisor.jsonl"),
        "daemon_log": str(daemon_log),
    }, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--install-root", required=True, type=Path)
    parser.add_argument("--current-n0500-manifest", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--validation-gpu", type=int, default=0)
    parser.add_argument("--daemon", action="store_true")
    args = parser.parse_args()
    if not 10 <= args.poll_seconds <= 300:
        parser.error("poll-seconds must be between 10 and 300")
    if args.validation_gpu not in FORMAL_GPU_IDS:
        parser.error("validation-gpu must be one of physical GPUs 0-7")
    return args


def main() -> None:
    args = parse_args()
    if args.daemon:
        raise SystemExit(daemonize(args))
    supervisor = None
    try:
        supervisor = Supervisor(args)
        supervisor.run()
    except Exception as exc:
        if supervisor is not None:
            supervisor.log(
                "sequence_stopped", error_type=type(exc).__name__, error=str(exc),
                traceback=traceback.format_exc(),
            )
            supervisor.save_state(
                status="stopped", phase="stopped", stopped_at=now_iso(),
                stop_reason={"type": type(exc).__name__, "message": str(exc)},
                test_accessed=False,
            )
        raise


if __name__ == "__main__":
    main()
