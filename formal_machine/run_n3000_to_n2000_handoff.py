#!/usr/bin/env python3
"""Fail-closed one-way handoff from the exact formal n=3000 run to fresh n=2000."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import yaml


PARENT_RUN_ID = "formal_sft_n3000_seed0042_20260717_014544"
TARGET_PREFIX = "formal_sft_n2000_seed0042_"
BASE_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
CORE_MANIFEST_NAME = "code_hash_manifest_20260717_010146.json"
HANDOFF_MANIFEST_NAME = "n3000_to_n2000_handoff_manifest_20260717_015614.json"
TARGET_CONFIG_SHA256 = "e495c64057a810e1c21bb44279e8b81f8159115eb5554aba7abc9d62cbccc8bd"
PROJECTED_START_FREE_BYTES = 1_157_493_686_272
FORMAL_GPU_IDS = list(range(8))
TERMINAL_FAILURE_STATES = {"failed", "failed_gate", "aborted", "cancelled"}
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


class HandoffStop(RuntimeError):
    """A condition that forbids automatic progression."""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_yaml(path: Path) -> dict:
    last_error: Exception | None = None
    for _ in range(3):
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("manifest is not a mapping")
            return payload
        except (OSError, ValueError, yaml.YAMLError) as exc:
            last_error = exc
            time.sleep(1)
    raise HandoffStop(f"cannot read stable manifest {path}: {last_error}")


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
    resolved = path.resolve()
    allowed = root.resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise HandoffStop(f"{label} escapes allowed root: {resolved} not under {allowed}")


def require_all_true_gates(payload: dict) -> None:
    gates = payload.get("gates")
    if not isinstance(gates, dict):
        raise HandoffStop("SFT gates are missing")
    missing = sorted(SFT_GATES.difference(gates))
    false = {key: gates.get(key) for key in sorted(gates) if gates.get(key) is not True}
    if missing or false:
        raise HandoffStop(f"SFT gates are not all true: missing={missing}, false={false}")


def validate_completed_sft_manifest(
    manifest_path: Path, install: Path, count: int, expected_run_id: str | None = None
) -> dict:
    manifest_path = manifest_path.resolve()
    root = (install / "runs" / "stage1_sft" / f"n{count:04d}_seed0042").resolve()
    require_within(manifest_path, root, "SFT manifest")
    if manifest_path.name != "run_manifest.yaml" or manifest_path.parent.parent != root:
        raise HandoffStop(f"unexpected SFT manifest layout: {manifest_path}")
    payload = read_yaml(manifest_path)
    run_id = str(payload.get("run_id", ""))
    if expected_run_id is not None:
        if run_id != expected_run_id:
            raise HandoffStop(f"parent run id changed: {run_id} != {expected_run_id}")
    elif not run_id.startswith(f"formal_sft_n{count:04d}_seed0042_"):
        raise HandoffStop(f"unexpected target run id: {run_id}")
    fixed = {
        "status": "completed",
        "stage": "stage1_sft",
        "formal_result": True,
        "test_accessed": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in fixed.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise HandoffStop(f"SFT terminal state mismatch: {mismatches}")
    data = payload.get("data", {})
    expected_data = {
        "version": "pathmmu_image_disjoint_v1",
        "dataset": f"pathvlm_sft_n{count:04d}",
        "qa_count": count,
        "image_reference_count": count,
        "all_image_paths_exist": True,
    }
    data_mismatches = {
        key: {"expected": expected, "actual": data.get(key)}
        for key, expected in expected_data.items()
        if data.get(key) != expected
    }
    if data_mismatches or "test" in str(data.get("adapter", "")).lower():
        raise HandoffStop(f"SFT data mismatch or forbidden path: {data_mismatches}")
    model = payload.get("model", {})
    if (
        model.get("base_id") != "Qwen/Qwen2.5-VL-7B-Instruct"
        or model.get("base_revision") != BASE_REVISION
    ):
        raise HandoffStop(f"SFT base model mismatch: {model}")
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
        key: {"expected": expected, "actual": training.get(key)}
        for key, expected in expected_training.items()
        if training.get(key) != expected
    }
    policy = training.get("checkpoint_policy", {})
    expected_policy = {
        "save_strategy": "epoch",
        "model_only_epoch_snapshots": 10,
        "full_resume_checkpoints_retained": 2,
        "final_model_retained": True,
        "automatic_scientific_snapshot_pruning": False,
    }
    policy_mismatches = {
        key: {"expected": expected, "actual": policy.get(key)}
        for key, expected in expected_policy.items()
        if policy.get(key) != expected
    }
    if training_mismatches or policy_mismatches:
        raise HandoffStop(
            f"SFT training/checkpoint policy mismatch: {training_mismatches} {policy_mismatches}"
        )
    hardware = payload.get("hardware", {})
    if hardware.get("cuda_visible_devices") != FORMAL_GPU_IDS or hardware.get("gpu_count") != 8:
        raise HandoffStop(f"SFT hardware topology mismatch: {hardware}")
    require_all_true_gates(payload)
    outputs = payload.get("outputs", {})
    final_checkpoint = Path(outputs.get("final_checkpoint", "")).resolve()
    if final_checkpoint != (manifest_path.parent / "output").resolve():
        raise HandoffStop(f"SFT final checkpoint mismatch: {final_checkpoint}")
    for name in ("model.safetensors.index.json", "trainer_state.json", "config.json"):
        if not (final_checkpoint / name).is_file():
            raise HandoffStop(f"SFT final checkpoint is missing {name}")
    snapshot_root = Path(outputs.get("epoch_snapshots", "")).resolve()
    if snapshot_root != (manifest_path.parent / "epoch_snapshots").resolve():
        raise HandoffStop(f"SFT snapshot root mismatch: {snapshot_root}")
    expected_stride = (count + 7) // 8
    expected_steps = [expected_stride * epoch for epoch in range(1, 11)]
    retention = payload.get("retention", {})
    retained_steps = [entry.get("global_step") for entry in retention.get("snapshots", [])]
    if retained_steps != expected_steps:
        raise HandoffStop(f"SFT retained snapshot steps mismatch: {retained_steps}")
    snapshots = sorted(
        snapshot_root.glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1])
    )
    if [int(path.name.split("-")[-1]) for path in snapshots] != expected_steps:
        raise HandoffStop(f"SFT snapshot directories are incomplete: {snapshot_root}")
    if any(not (path / "snapshot_manifest.json").is_file() for path in snapshots):
        raise HandoffStop("SFT snapshot manifest is missing")
    expected_config_hash = (
        "d15428f642fdc522254e7b3322c977d5bb985e6516df0350c816d988638ab273"
        if count == 3000 else TARGET_CONFIG_SHA256
    )
    if payload.get("provenance", {}).get("source_config_sha256") != expected_config_hash:
        raise HandoffStop("SFT source config hash mismatch")
    return payload


def acquire_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise HandoffStop(f"another handoff watcher holds the lock: {path}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} started_at={now_iso()}\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def select_unique_new_manifest(before: set[Path], after: list[Path]) -> Path | None:
    candidates = [path.resolve() for path in after if path.resolve() not in before]
    if len(candidates) > 1:
        raise HandoffStop(f"multiple new n=2000 manifests appeared: {candidates}")
    return candidates[0] if candidates else None


def require_start_disk(free_bytes: int) -> None:
    if free_bytes < PROJECTED_START_FREE_BYTES:
        raise HandoffStop(
            f"disk below formal start gate: free={free_bytes}, "
            f"required={PROJECTED_START_FREE_BYTES}"
        )


def gpus_are_idle(processes: list[dict]) -> bool:
    return not processes


class Handoff:
    def __init__(self, args: argparse.Namespace):
        self.repo = args.repo_root.resolve()
        self.install = args.install_root.resolve()
        self.parent_manifest = args.parent_manifest.resolve()
        self.poll_seconds = args.poll_seconds
        expected_parent = (
            self.install / "runs" / "stage1_sft" / "n3000_seed0042"
            / PARENT_RUN_ID / "run_manifest.yaml"
        ).resolve()
        if self.parent_manifest != expected_parent:
            raise HandoffStop(f"watcher is bound to {expected_parent}, not {self.parent_manifest}")
        self.control = self.install / "control" / "n3000_to_n2000_seed0042_20260717"
        self.state_path = self.control / "handoff_state.json"
        self.log_path = self.control / "handoff.jsonl"
        self.command_log = self.control / "n2000_launcher.log"
        self.control.mkdir(parents=True, exist_ok=True)
        self.lock_handle = acquire_lock(self.control / "handoff.lock")
        current_commit = self.git_commit()
        if self.state_path.is_file():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            if self.state.get("status") in {"completed", "stopped"}:
                raise HandoffStop(f"existing handoff is terminal: {self.state_path}")
            if self.state.get("expected_git_commit") != current_commit:
                raise HandoffStop("Git commit changed since handoff watcher start")
            if Path(self.state.get("parent_manifest", "")).resolve() != self.parent_manifest:
                raise HandoffStop("recorded parent manifest changed")
        else:
            self.state = {
                "schema_version": 1,
                "handoff_id": "n3000_to_n2000_seed0042_20260717",
                "status": "running",
                "phase": "waiting_for_n3000",
                "created_at": now_iso(),
                "expected_git_commit": current_commit,
                "parent_manifest": str(self.parent_manifest),
                "target_count": 2000,
                "target_seed": 42,
                "test_accessed": False,
                "events": [],
                "pending_launch": None,
            }
            self.save()

    def git_commit(self) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"],
            text=True, capture_output=True, check=False,
        )
        if result.returncode:
            raise HandoffStop(f"cannot read Git commit: {result.stderr.strip()}")
        return result.stdout.strip()

    def save(self, **updates) -> None:
        self.state.update(updates)
        self.state["updated_at"] = now_iso()
        self.state["watcher_pid"] = os.getpid()
        atomic_write_json(self.state_path, self.state)

    def event(self, name: str, **details) -> None:
        entry = {"timestamp": now_iso(), "event": name, **details}
        self.state.setdefault("events", []).append(entry)
        self.save(last_event=entry)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def verify_integrity(self) -> None:
        status = subprocess.run(
            ["git", "-C", str(self.repo), "status", "--porcelain"],
            text=True, capture_output=True, check=False,
        )
        if status.returncode or status.stdout.strip():
            raise HandoffStop(f"repository is not clean: {status.stdout.strip()}")
        if self.git_commit() != self.state["expected_git_commit"]:
            raise HandoffStop("Git commit changed while handoff watcher was running")
        verifier = self.repo / "scripts" / "verify_code_hash_manifest.py"
        for name in (CORE_MANIFEST_NAME, HANDOFF_MANIFEST_NAME):
            manifest = self.repo / "protocol" / name
            result = subprocess.run(
                [sys.executable, str(verifier), "--repo-root", str(self.repo),
                 "--manifest", str(manifest)],
                text=True, capture_output=True, check=False,
            )
            if result.returncode:
                raise HandoffStop(
                    f"code manifest verification failed for {manifest}: "
                    f"{result.stdout}{result.stderr}"
                )
        config = self.install / "generated_configs" / "sft" / "sft_n2000_seed0042.yaml"
        if not config.is_file() or sha256(config) != TARGET_CONFIG_SHA256:
            raise HandoffStop(f"frozen n=2000 config hash mismatch: {config}")
        require_start_disk(shutil.disk_usage(self.install).free)

    def gpu_state(self) -> tuple[list[dict], list[dict]]:
        inventory = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False,
        )
        if inventory.returncode:
            raise HandoffStop(f"nvidia-smi inventory failed: {inventory.stderr.strip()}")
        rows = []
        for line in inventory.stdout.splitlines():
            index, name, total, used, free, util = [item.strip() for item in line.split(",")]
            rows.append({"index": int(index), "name": name, "memory_total_mib": int(total),
                         "memory_used_mib": int(used), "memory_free_mib": int(free),
                         "utilization_percent": int(util)})
        if [row["index"] for row in rows] != FORMAL_GPU_IDS:
            raise HandoffStop(f"GPU inventory changed: {rows}")
        processes = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
             "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False,
        )
        if processes.returncode:
            raise HandoffStop(f"nvidia-smi process query failed: {processes.stderr.strip()}")
        process_rows = []
        for line in processes.stdout.splitlines():
            if line.strip():
                uuid, pid, name, used = [item.strip() for item in line.split(",")]
                process_rows.append({"gpu_uuid": uuid, "pid": int(pid), "process_name": name,
                                     "used_memory_mib": int(used)})
        return rows, process_rows

    def wait_parent(self) -> None:
        while True:
            self.verify_integrity()
            payload = read_yaml(self.parent_manifest)
            status = payload.get("status")
            if status == "completed":
                validate_completed_sft_manifest(
                    self.parent_manifest, self.install, 3000, PARENT_RUN_ID
                )
                self.event("n3000_completed_and_validated", manifest=str(self.parent_manifest))
                return
            if status in TERMINAL_FAILURE_STATES or status != "running":
                raise HandoffStop(f"n=3000 entered non-success state: {status}")
            self.save(phase="waiting_for_n3000", parent_status=status)
            time.sleep(self.poll_seconds)

    def wait_idle_gpus(self) -> None:
        self.event("waiting_for_idle_gpus")
        last_processes = None
        while True:
            self.verify_integrity()
            gpus, processes = self.gpu_state()
            if gpus_are_idle(processes):
                self.event("all_eight_gpus_idle", gpus=gpus)
                return
            if processes != last_processes:
                self.event("gpu_wait_blocked", processes=processes)
                last_processes = processes
            self.save(phase="waiting_for_idle_gpus", blocking_gpu_processes=processes)
            time.sleep(self.poll_seconds)

    def discover_target_manifests(self) -> list[Path]:
        root = self.install / "runs" / "stage1_sft" / "n2000_seed0042"
        return sorted(root.glob(f"{TARGET_PREFIX}*/run_manifest.yaml")) if root.is_dir() else []

    def launch_target(self) -> Path:
        recorded = self.state.get("target_manifest")
        if recorded:
            return Path(recorded).resolve()
        if self.state.get("pending_launch"):
            raise HandoffStop(
                f"unresolved prior launch intent forbids retry: {self.state['pending_launch']}"
            )
        self.verify_integrity()
        self.wait_idle_gpus()
        before = {path.resolve() for path in self.discover_target_manifests()}
        config = self.install / "generated_configs" / "sft" / "sft_n2000_seed0042.yaml"
        command = ["bash", str(self.repo / "scripts" / "launch_formal_sft_run.sh"),
                   "--config", str(config)]
        pending = {"declared_at": now_iso(), "command": command,
                   "rule": "never retry automatically without recovering exactly one manifest"}
        self.save(phase="launching_n2000", pending_launch=pending)
        self.event("launching_n2000", command=command, command_log=str(self.command_log))
        env = os.environ.copy()
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
            env.pop(key, None)
        with self.command_log.open("a", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command, cwd=self.repo, env=env, stdin=subprocess.DEVNULL,
                stdout=log_handle, stderr=subprocess.STDOUT, text=True, start_new_session=True,
            )
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            manifest = select_unique_new_manifest(before, self.discover_target_manifests())
            if manifest is not None:
                self.save(pending_launch=None, target_manifest=str(manifest), target_pid=process.pid,
                          phase="running_n2000")
                self.event("n2000_launched", manifest=str(manifest), pid=process.pid)
                return manifest
            returncode = process.poll()
            if returncode is not None:
                raise HandoffStop(
                    f"n=2000 launcher exited with {returncode} before manifest creation"
                )
            time.sleep(2)
        raise HandoffStop("n=2000 launcher did not create one manifest within 300 seconds")

    def wait_target(self, manifest: Path) -> None:
        while True:
            payload = read_yaml(manifest)
            status = payload.get("status")
            if status == "completed":
                validate_completed_sft_manifest(manifest, self.install, 2000)
                self.event("n2000_completed_and_validated", manifest=str(manifest))
                self.save(status="completed", phase="completed", completed_at=now_iso(),
                          test_accessed=False)
                return
            if status in TERMINAL_FAILURE_STATES or status != "running":
                raise HandoffStop(f"n=2000 entered non-success state: {status}")
            self.save(phase="running_n2000", target_manifest=str(manifest), target_status=status)
            time.sleep(self.poll_seconds)

    def run(self) -> None:
        self.event("handoff_watcher_started", parent_manifest=str(self.parent_manifest))
        self.wait_parent()
        target = self.launch_target()
        self.wait_target(target)


def daemonize(args: argparse.Namespace) -> int:
    control = args.install_root.resolve() / "control" / "n3000_to_n2000_seed0042_20260717"
    control.mkdir(parents=True, exist_ok=True)
    daemon_log = control / "daemon.log"
    command = [sys.executable, str(Path(__file__).resolve()),
               "--repo-root", str(args.repo_root.resolve()),
               "--install-root", str(args.install_root.resolve()),
               "--parent-manifest", str(args.parent_manifest.resolve()),
               "--poll-seconds", str(args.poll_seconds)]
    with daemon_log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, cwd=args.repo_root.resolve(), stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT, text=True, start_new_session=True,
        )
    time.sleep(2)
    if process.poll() is not None:
        raise HandoffStop(f"handoff daemon exited immediately; see {daemon_log}")
    print(json.dumps({"watcher_pid": process.pid,
                      "state": str(control / "handoff_state.json"),
                      "events": str(control / "handoff.jsonl"),
                      "daemon_log": str(daemon_log)}, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--install-root", required=True, type=Path)
    parser.add_argument("--parent-manifest", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--daemon", action="store_true")
    args = parser.parse_args()
    if not 10 <= args.poll_seconds <= 300:
        parser.error("poll-seconds must be between 10 and 300")
    return args


def main() -> None:
    args = parse_args()
    if args.daemon:
        raise SystemExit(daemonize(args))
    handoff = None
    try:
        handoff = Handoff(args)
        handoff.run()
    except Exception as exc:
        if handoff is not None:
            handoff.event("handoff_stopped", error_type=type(exc).__name__, error=str(exc),
                          traceback=traceback.format_exc())
            handoff.save(status="stopped", phase="stopped", stopped_at=now_iso(),
                         stop_reason={"type": type(exc).__name__, "message": str(exc)},
                         test_accessed=False)
        raise


if __name__ == "__main__":
    main()
