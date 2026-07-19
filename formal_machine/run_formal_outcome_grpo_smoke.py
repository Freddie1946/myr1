#!/usr/bin/env python3
"""Fail-closed one-step Outcome-GRPO gate on the selected formal SFT parent."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


BASE_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
DATA_VERSION = "pathmmu_image_disjoint_v2"
GATE_PROTOCOL = "outcome_grpo_gate_manifest_20260720_013054.json"
CODE_MANIFEST = "outcome_grpo_gate_code_manifest_20260720_013054.json"
CURVE_RUN_ID = "sft_base_n2000_n3000_seed0042_20260719_184313"
CURVE_MANIFEST_SHA256 = "e7170be9ddd3050b8421a02c3a12a3e59d9762b6296a26f246abe445ea06ff24"
SFT_RUN_ID = "formal_sft_n3000_seed0042_20260717_014544"
SFT_MANIFEST_SHA256 = "088959edf5d1fa025938e1ad46331fab8e3e5007b4fa3a3c9125bc29b329e4b5"
SNAPSHOT_MANIFEST_SHA256 = "6c8708dd677f5ab1563e77317d86cd1b53cb9b969ebda09e4a312581a2b9f1f1"
SMOKE_YAML_SHA256 = "4c6fb9b6c0d1016320e9eeb614588573ae49f826cc14013c80e71f9b9d18cfd6"
SMOKE_JSON_SHA256 = "7f7e9fbb3205085c03bfb7a10678120701dbfc0cd571a2d1c0a771a5527f95c0"
FROZEN_SPLIT_REPORT_SHA256 = "f75e74884b9b08dcb16276dbfafa89d94ae28f31deef07c54d64b6a6d5de32aa"
FORMAL_DATA_MANIFEST_SHA256 = "6dc08a6317dcac8932df9c7f0b345b80bbf7f63d9312908add6b1dcdb8e534ba"
GPU_IDS = list(range(8))
MASTER_PORT = 29800
DISK_RESERVE_BYTES = 550 * 1024**3
EXPECTED_EVENTS = 16


class GateStop(RuntimeError):
    """A fail-closed gate condition."""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise GateStop(f"missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise GateStop(f"missing YAML file: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GateStop(f"YAML is not a mapping: {path}")
    return payload


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def require_hash(path: Path, expected: str) -> dict[str, Any]:
    actual = sha256(path)
    if actual != expected:
        raise GateStop(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    return {"path": str(path), "sha256": actual}


def verify_git_clean(repo: Path) -> dict[str, Any]:
    branch = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo, text=True, capture_output=True, check=True
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, text=True, capture_output=True, check=True
    ).stdout
    if status:
        raise GateStop("repository must be clean before the gate:\n" + status)
    return {"branch": branch, "commit": commit, "worktree_clean": True}


def verify_code_manifest(repo: Path, python: Path) -> dict[str, Any]:
    manifest = repo / "protocol" / CODE_MANIFEST
    result = subprocess.run(
        [str(python), str(repo / "scripts/verify_code_hash_manifest.py"),
         "--repo-root", str(repo), "--manifest", str(manifest)],
        text=True, capture_output=True, check=False,
    )
    if result.returncode:
        raise GateStop("gate code manifest failed:\n" + result.stdout + result.stderr)
    return {"path": str(manifest), "sha256": sha256(manifest), "verified": True}


def verify_protocol(repo: Path) -> dict[str, Any]:
    path = repo / "protocol" / GATE_PROTOCOL
    payload = read_json(path)
    expected = {
        "status": "frozen_before_execution",
        "formal_result": False,
        "data_version": DATA_VERSION,
        "parent_sample_count": 3000,
        "parent_epoch": 3,
        "parent_global_step": 1125,
        "seed": 42,
        "max_steps": 1,
        "num_generations": 4,
        "rl_smoke_count": 8,
        "test_accessed": False,
        "long_grpo_authorized": False,
    }
    mismatch = {key: {"expected": value, "actual": payload.get(key)}
                for key, value in expected.items() if payload.get(key) != value}
    if mismatch or payload.get("picked_json_used") is not False:
        raise GateStop(f"gate protocol mismatch: {mismatch}")
    return {"path": str(path), "sha256": sha256(path), "verified_fields": expected}


def verify_snapshot(snapshot: Path) -> dict[str, Any]:
    manifest_path = snapshot / "snapshot_manifest.json"
    require_hash(manifest_path, SNAPSHOT_MANIFEST_SHA256)
    manifest = read_json(manifest_path)
    if (
        manifest.get("epoch") != 3.0
        or manifest.get("global_step") != 1125
        or manifest.get("model_only") is not True
        or manifest.get("resumable") is not False
    ):
        raise GateStop("selected snapshot metadata mismatch")
    required = {
        "chat_template.json", "config.json", "model.safetensors.index.json",
        "preprocessor_config.json", "tokenizer_config.json",
    }
    entries = manifest.get("files", [])
    if not required.issubset({item.get("name") for item in entries}):
        raise GateStop("selected snapshot lacks required model/processor files")
    for item in entries:
        file_path = snapshot / str(item.get("name"))
        if not file_path.is_file() or file_path.stat().st_size != item.get("size_bytes"):
            raise GateStop(f"snapshot file size mismatch: {file_path}")
        if sha256(file_path) != item.get("sha256"):
            raise GateStop(f"snapshot file hash mismatch: {file_path}")
    config = read_json(snapshot / "config.json")
    if config.get("model_type") != "qwen2_5_vl":
        raise GateStop(f"selected parent is not Qwen2.5-VL: {config.get('model_type')}")
    return {
        "path": str(snapshot),
        "manifest": str(manifest_path),
        "manifest_sha256": SNAPSHOT_MANIFEST_SHA256,
        "file_count": len(entries),
        "total_file_bytes": sum(int(item["size_bytes"]) for item in entries),
        "all_files_verified": True,
    }


def verify_parent(install: Path) -> tuple[Path, dict[str, Any]]:
    curve_path = install / "runs" / "stage1_validation_curves" / CURVE_RUN_ID / "run_manifest.yaml"
    require_hash(curve_path, CURVE_MANIFEST_SHA256)
    curve = read_yaml(curve_path)
    if (
        curve.get("run_id") != CURVE_RUN_ID
        or curve.get("status") != "completed"
        or curve.get("formal_result") is not True
        or curve.get("data", {}).get("version") != DATA_VERSION
        or curve.get("gates", {}).get("test_not_accessed") is not True
        or any(value is not True for value in curve.get("gates", {}).values())
    ):
        raise GateStop("validation-curve parent-selection evidence is not fully passing")
    selection = curve.get("selections", {}).get("3000", {})
    if {key: selection.get(key) for key in ("epoch", "global_step")} != {
        "epoch": 3, "global_step": 1125
    }:
        raise GateStop(f"n=3000 validation selection mismatch: {selection}")

    sft_path = install / "runs" / "stage1_sft" / "n3000_seed0042" / SFT_RUN_ID / "run_manifest.yaml"
    require_hash(sft_path, SFT_MANIFEST_SHA256)
    sft = read_yaml(sft_path)
    if (
        sft.get("status") != "completed"
        or sft.get("formal_result") is not True
        or sft.get("test_accessed") is not False
        or sft.get("model", {}).get("base_revision") != BASE_REVISION
        or sft.get("training", {}).get("seed") != 42
        or sft.get("training", {}).get("finetuning_type") != "full"
        or sft.get("training", {}).get("language_model_trainable") is not True
        or sft.get("training", {}).get("vision_tower_frozen") is not True
        or sft.get("training", {}).get("multimodal_projector_frozen") is not True
        or any(value is not True for value in sft.get("gates", {}).values())
    ):
        raise GateStop("formal SFT parent gates are not fully passing")
    recorded_parent = next(
        (item for item in curve.get("parents", []) if item.get("sample_count") == 3000), None
    )
    if not recorded_parent or recorded_parent.get("manifest_sha256") != SFT_MANIFEST_SHA256:
        raise GateStop("validation curve does not bind the expected SFT parent manifest")
    snapshot = sft_path.parent / "epoch_snapshots" / "checkpoint-1125"
    evidence = verify_snapshot(snapshot)
    selected_job = next((job for job in curve.get("jobs", []) if job.get("label") == "n3000_epoch03"), None)
    if (
        not selected_job
        or Path(selected_job.get("checkpoint", "")).resolve() != snapshot.resolve()
        or selected_job.get("checkpoint_evidence", {}).get("manifest_sha256")
        != SNAPSHOT_MANIFEST_SHA256
    ):
        raise GateStop("curve job does not bind the selected snapshot")
    evidence.update({
        "sft_manifest": str(sft_path),
        "sft_manifest_sha256": SFT_MANIFEST_SHA256,
        "curve_manifest": str(curve_path),
        "curve_manifest_sha256": CURVE_MANIFEST_SHA256,
        "validation_accuracy": selected_job.get("metrics", {}).get("mean_accuracy_reward"),
        "validation_format": selected_job.get("metrics", {}).get("mean_format_reward"),
    })
    return snapshot.resolve(), evidence


def verify_data(repo: Path, install: Path) -> dict[str, Any]:
    local_manifest = read_json(repo / "data" / DATA_VERSION / "manifest.json")
    if (
        local_manifest.get("version") != DATA_VERSION
        or local_manifest.get("picked_json_used") is not False
        or any(int(value) != 0 for value in local_manifest.get("exact_content_overlap_counts", {}).values())
        or local_manifest.get("primary_splits", {}).get("test", {}).get("qa_count") != 999
    ):
        raise GateStop("local v2 split manifest invariants failed")
    split_report = install / "reports" / "frozen_split_verification_v2.json"
    require_hash(split_report, FROZEN_SPLIT_REPORT_SHA256)
    split_payload = read_json(split_report)
    if split_payload.get("passed") is not True or split_payload.get("gates", {}).get("picked_json_absent") is not True:
        raise GateStop("frozen v2 split verification is not passing")
    formal_manifest = install / "data" / DATA_VERSION / "formal_data_manifest.json"
    require_hash(formal_manifest, FORMAL_DATA_MANIFEST_SHA256)
    formal = read_json(formal_manifest)
    if (
        formal.get("data_version") != DATA_VERSION
        or formal.get("picked_json_used") is not False
        or any(int(value) != 0 for value in formal.get("pairwise_image_overlaps", {}).values())
        or formal.get("test_policy") != "evaluation_only_never_training_or_selection"
    ):
        raise GateStop("formal data manifest invariants failed")
    yaml_path = install / "data" / DATA_VERSION / "grpo" / "pathvlm_rl_smoke_n0008.yaml"
    json_path = install / "data" / DATA_VERSION / "grpo" / "pathvlm_rl_smoke_n0008.json"
    require_hash(yaml_path, SMOKE_YAML_SHA256)
    require_hash(json_path, SMOKE_JSON_SHA256)
    adapter = read_json(json_path)
    if not isinstance(adapter, list) or len(adapter) != 8:
        raise GateStop(f"RL smoke adapter must contain exactly 8 rows, got {len(adapter)}")
    frozen_rl = read_json(repo / "data" / DATA_VERSION / "subsets" / "rl" / "rl_1000_with_cot.json")
    frozen_triplets = {
        (Path(row["image"]).name, row["problem"], row["solution"]) for row in frozen_rl
    }
    image_hashes: list[str] = []
    for index, row in enumerate(adapter):
        image = Path(row.get("image", ""))
        if not image.is_file():
            raise GateStop(f"RL smoke image missing at row {index}: {image}")
        basename = image.name
        if (basename, row.get("problem"), row.get("solution")) not in frozen_triplets:
            raise GateStop(f"RL smoke row {index} is not an exact frozen RL record")
        image_hashes.append(sha256(image))
    return {
        "version": DATA_VERSION,
        "split": "rl_smoke_n0008",
        "count": 8,
        "yaml": str(yaml_path),
        "yaml_sha256": SMOKE_YAML_SHA256,
        "adapter": str(json_path),
        "adapter_sha256": SMOKE_JSON_SHA256,
        "unique_image_content_count": len(set(image_hashes)),
        "all_rows_in_frozen_rl": True,
        "all_image_paths_exist": True,
        "pairwise_primary_split_image_overlaps_zero": True,
        "picked_json_used": False,
        "test_accessed": False,
    }


def gpu_inventory(selected: list[int]) -> dict[str, Any]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False,
    )
    if result.returncode:
        raise GateStop("nvidia-smi inventory failed: " + result.stderr)
    rows: list[dict[str, Any]] = []
    uuids: set[str] = set()
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        index = int(parts[0])
        if index in selected:
            row = {
                "index": index, "uuid": parts[1], "name": parts[2],
                "memory_total_mib": int(parts[3]), "memory_used_mib": int(parts[4]),
                "memory_free_mib": int(parts[5]), "utilization_percent": int(parts[6]),
            }
            rows.append(row)
            uuids.add(parts[1])
    rows.sort(key=lambda item: item["index"])
    if [item["index"] for item in rows] != selected:
        raise GateStop(f"selected GPU inventory mismatch: {rows}")
    process_result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False,
    )
    if process_result.returncode:
        raise GateStop("nvidia-smi process query failed: " + process_result.stderr)
    conflicts = []
    for line in process_result.stdout.splitlines():
        if not line.strip():
            continue
        uuid, pid, process_name, used = [item.strip() for item in line.split(",")]
        if uuid in uuids:
            conflicts.append({"gpu_uuid": uuid, "pid": int(pid), "process_name": process_name,
                              "used_memory_mib": int(used)})
    if conflicts:
        raise GateStop(f"selected GPUs already have compute processes: {conflicts}")
    return {"gpus": rows, "compute_processes": [], "all_selected_idle": True}


def system_inventory(install: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(install)
    if disk.free < DISK_RESERVE_BYTES:
        raise GateStop(
            f"free disk {disk.free / 1024**3:.1f} GiB is below reserve "
            f"{DISK_RESERVE_BYTES / 1024**3:.1f} GiB"
        )
    meminfo: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        meminfo[key] = int(value.strip().split()[0]) * 1024
    return {
        "disk_total_bytes": disk.total, "disk_used_bytes": disk.used,
        "disk_free_bytes": disk.free, "disk_reserve_bytes": DISK_RESERVE_BYTES,
        "memory_total_bytes": meminfo.get("MemTotal"),
        "memory_available_bytes": meminfo.get("MemAvailable"),
        "swap_total_bytes": meminfo.get("SwapTotal"),
        "swap_free_bytes": meminfo.get("SwapFree"),
    }


def require_free_port(port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as exc:
        raise GateStop(f"master port {port} is unavailable: {exc}") from exc
    finally:
        sock.close()


def run_checked(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    evidence = {"command": command, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    if result.returncode:
        raise GateStop(f"command failed: {command}\n{result.stdout}{result.stderr}")
    return evidence


def build_command(repo: Path, install: Path, python: Path, parent_alias: Path, output: Path) -> list[str]:
    return [
        str(python), "-m", "torch.distributed.run", "--nproc_per_node=8",
        f"--master_port={MASTER_PORT}", str(repo / "scripts/grpo_pathmmu.py"),
        "--deepspeed", str(repo / "configs/deepspeed/ds_z3_optimizer_offload_torch_adamw.json"),
        "--output_dir", str(output),
        "--model_name_or_path", str(parent_alias),
        "--dataset_name", str(install / "data" / DATA_VERSION / "grpo" / "pathvlm_rl_smoke_n0008.yaml"),
        "--image_root", "/",
        "--reward_funcs", "accuracy", "format",
        "--freeze_vision_modules", "true",
        "--max_pixels", "65536", "--min_pixels", "3136",
        "--num_generations", "4", "--max_completion_length", "192",
        "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "1",
        "--learning_rate", "1.0e-6", "--logging_steps", "1",
        "--bf16", "true", "--torch_dtype", "bfloat16",
        "--gradient_checkpointing", "true", "--attn_implementation", "sdpa",
        "--beta", "0.04", "--num_iterations", "1",
        "--save_strategy", "no", "--report_to", "none",
        "--seed", "42", "--data_seed", "42", "--max_steps", "1",
    ]


def run_training(command: list[str], repo: Path, install: Path, run_dir: Path, env: dict[str, str]) -> int:
    log_path = run_dir / "train.log"
    monitor_path = run_dir / "resource_monitor.jsonl"
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(command, cwd=repo / "vendor/open-r1-multimodal", env=env,
                                   stdout=log_handle, stderr=subprocess.STDOUT)
        while process.poll() is None:
            sample: dict[str, Any] = {"timestamp": now_iso(), "pid": process.pid}
            try:
                query = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False,
                )
                sample["gpus"] = query.stdout.strip().splitlines()
                sample["disk_free_bytes"] = shutil.disk_usage(install).free
                if sample["disk_free_bytes"] < DISK_RESERVE_BYTES:
                    process.terminate()
                    sample["stop_reason"] = "disk_reserve_breached"
            except Exception as exc:  # monitoring failure is recorded; trainer status remains authoritative
                sample["monitor_error"] = repr(exc)
            with monitor_path.open("a", encoding="utf-8") as monitor:
                monitor.write(json.dumps(sample) + "\n")
            print(f"[outcome-grpo-gate] pid={process.pid} still running; see {log_path}", flush=True)
            time.sleep(15)
        return int(process.returncode)


def wrapped(text: str):
    return [[{"role": "assistant", "content": text}]]


def audit_rewards(repo: Path, reward_dir: Path) -> dict[str, Any]:
    sys.path.insert(0, str(repo / "scripts"))
    from pathmmu_rewards import accuracy_reward, format_reward  # noqa: PLC0415

    files = sorted(reward_dir.glob("rank_*.jsonl"))
    if [path.name for path in files] != [f"rank_{rank:02d}.jsonl" for rank in GPU_IDS]:
        raise GateStop(f"expected one reward log per rank, found {[path.name for path in files]}")
    events: list[dict[str, Any]] = []
    for path in files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GateStop(f"invalid reward JSONL {path}:{line_number}: {exc}") from exc
            events.append(event)
    if len(events) != EXPECTED_EVENTS:
        raise GateStop(f"expected {EXPECTED_EVENTS} reward events, found {len(events)}")
    pairs: list[dict[str, Any]] = []
    for rank in GPU_IDS:
        rank_events = [event for event in events if event.get("rank") == rank]
        if len(rank_events) != 2 or {event.get("reward_type") for event in rank_events} != {"accuracy", "format"}:
            raise GateStop(f"rank {rank} reward event mismatch: {rank_events}")
        by_type = {event["reward_type"]: event for event in rank_events}
        accuracy_event, format_event = by_type["accuracy"], by_type["format"]
        for key in ("completion", "solution", "call_index", "item_index"):
            if accuracy_event.get(key) != format_event.get(key):
                raise GateStop(f"rank {rank} reward pair mismatch in {key}")
        completion = str(accuracy_event["completion"])
        solution = str(accuracy_event["solution"])
        offline_accuracy = accuracy_reward(wrapped(completion), [solution])[0]
        offline_format = format_reward(wrapped(completion))[0]
        if (
            float(accuracy_event["reward"]) != offline_accuracy
            or float(format_event["reward"]) != offline_format
        ):
            raise GateStop(f"rank {rank} online/offline reward parser mismatch")
        pairs.append({
            "rank": rank, "completion": completion, "solution": solution,
            "accuracy_reward": offline_accuracy, "format_reward": offline_format,
            "total_reward": offline_accuracy + offline_format,
        })
    groups: dict[str, list[float]] = {}
    for pair in pairs:
        groups.setdefault(pair["solution"], []).append(pair["total_reward"])
    if sorted(len(values) for values in groups.values()) != [4, 4]:
        raise GateStop(f"expected two four-generation prompt groups, got {[len(v) for v in groups.values()]}")
    group_std = [statistics.stdev(values) for values in groups.values()]
    mean_std = statistics.mean(group_std)
    return {
        "event_files": [str(path) for path in files],
        "event_count": len(events), "completion_count": len(pairs),
        "parser_consistency": True, "pairs": pairs,
        "group_total_rewards": list(groups.values()),
        "group_sample_std": group_std, "mean_group_sample_std": mean_std,
        "positive_reward_variance": mean_std > 0.0,
    }


def audit_training_state(output: Path) -> dict[str, Any]:
    path = output / "pathvlm_train_state_audit.json"
    payload = read_json(path)
    history = payload.get("log_history", [])
    step_rows = [row for row in history if "loss" in row and "grad_norm" in row]
    if len(step_rows) != 1:
        raise GateStop(f"expected exactly one logged optimizer step, found {len(step_rows)}")
    row = step_rows[0]
    numeric_keys = ("loss", "grad_norm", "reward", "reward_std")
    if any(key not in row or not math.isfinite(float(row[key])) for key in numeric_keys):
        raise GateStop(f"non-finite or missing training metric: {row}")
    trainability = payload.get("trainability", {})
    return {
        "path": str(path), "global_step": payload.get("global_step"),
        "step_metrics": row, "trainability": trainability,
        "loss_finite": math.isfinite(float(row["loss"])),
        "gradient_finite_nonzero": math.isfinite(float(row["grad_norm"])) and float(row["grad_norm"]) > 0,
        "trainer_reward_std_positive": float(row["reward_std"]) > 0,
        "trainability_gate_passed": trainability.get("passed") is True,
    }


def audit_output_loadability(output: Path, python: Path, repo: Path) -> dict[str, Any]:
    required = {"config.json", "model.safetensors.index.json", "preprocessor_config.json",
                "tokenizer_config.json", "pathvlm_train_state_audit.json"}
    missing = sorted(name for name in required if not (output / name).is_file())
    if missing:
        raise GateStop(f"saved gate checkpoint lacks files: {missing}")
    code = (
        "from transformers import AutoConfig, AutoProcessor; import sys; "
        "p=sys.argv[1]; c=AutoConfig.from_pretrained(p, local_files_only=True); "
        "AutoProcessor.from_pretrained(p, local_files_only=True); "
        "assert c.model_type=='qwen2_5_vl'; print(c.model_type)"
    )
    evidence = run_checked([str(python), "-c", code, str(output)], cwd=repo)
    evidence.update({"required_files_present": True, "model_type": "qwen2_5_vl"})
    return evidence


def compare_tensors(parent: Path, child: Path, run_dir: Path, python: Path, repo: Path) -> dict[str, Any]:
    output = run_dir / "tensor_comparison.json"
    run_checked(
        [str(python), str(repo / "scripts/compare_checkpoint_tensors.py"),
         "--parent", str(parent), "--child", str(child), "--output", str(output)],
        cwd=repo,
    )
    payload = read_json(output)
    language = payload.get("tensors", {}).get("language", {})
    visual = payload.get("tensors", {}).get("visual", {})
    payload["gates"] = {
        "representative_language_tensor_changed": (
            language.get("exactly_equal") is False and int(language.get("changed_elements", 0)) > 0
        ),
        "representative_visual_tensor_exactly_equal": visual.get("exactly_equal") is True,
    }
    atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--install-root", type=Path, default=Path("/home/wjy/pathvlm_r1_v1_formal"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    repo, install = args.repo_root.resolve(), args.install_root.resolve()
    python = Path(sys.executable).resolve()
    if python != (install / "envs/grpo/bin/python").resolve():
        raise GateStop(f"must run with pinned GRPO Python, got {python}")

    preflight = {
        "created_at": now_iso(), "git": verify_git_clean(repo),
        "protocol": verify_protocol(repo), "code": verify_code_manifest(repo, python),
    }
    parent, preflight["parent"] = verify_parent(install)
    preflight["data"] = verify_data(repo, install)
    preflight["parser_regression"] = run_checked(
        [str(python), str(repo / "scripts/test_pathmmu_rewards.py")], cwd=repo
    )
    preflight["wrapper_regression"] = run_checked(
        [str(python), str(repo / "scripts/test_grpo_pathmmu_audit.py")], cwd=repo
    )
    preflight["system"] = system_inventory(install)
    preflight["passed"] = True
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return

    preflight["hardware"] = gpu_inventory(GPU_IDS)
    require_free_port(MASTER_PORT)
    run_id = f"outcome_grpo_gate_n0008_seed0042_{timestamp()}"
    run_dir = install / "runs" / "stage2_outcome_grpo" / "gate_n0008_seed0042" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    parent_alias = run_dir / "parent_Qwen2.5-VL-7B-Instruct"
    parent_alias.symlink_to(parent, target_is_directory=True)
    if parent_alias.resolve() != parent:
        raise GateStop("run-local parent alias does not resolve to the selected snapshot")
    output = run_dir / "output"
    reward_dir = run_dir / "online_reward_events"
    command = build_command(repo, install, python, parent_alias, output)
    manifest: dict[str, Any] = {
        "schema_version": 1, "run_id": run_id, "status": "running",
        "stage": "stage2_outcome_grpo_one_step_gate", "formal_result": False,
        "gate_passed": False, "created_at": now_iso(), "preflight": preflight,
        "parent": {**preflight["parent"], "alias": str(parent_alias),
                   "alias_realpath": str(parent_alias.resolve())},
        "training": {
            "seed": 42, "max_steps": 1, "num_generations": 4,
            "learning_rate": 1.0e-6, "beta": 0.04, "max_completion_length": 192,
            "per_device_train_batch_size": 1, "gradient_accumulation_steps": 1,
            "deepspeed": "zero3_optimizer_cpu_offload", "language_model_trainable": True,
            "vision_tower_frozen": True, "multimodal_projector_frozen": True,
        },
        "command": command, "test_accessed": False, "picked_json_used": False,
        "long_grpo_authorized": False,
        "outputs": {"run_dir": str(run_dir), "model": str(output),
                    "reward_events": str(reward_dir), "train_log": str(run_dir / "train.log")},
    }
    manifest_path = run_dir / "run_manifest.json"
    atomic_json(manifest_path, manifest)

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, GPU_IDS)),
        "NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "DEBUG_MODE": "false", "PATHVLM_REWARD_LOG_DIR": str(reward_dir),
        "TOKENIZERS_PARALLELISM": "false",
    })
    for key in list(env):
        if key.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
            env.pop(key, None)

    try:
        returncode = run_training(command, repo, install, run_dir, env)
        manifest["training_returncode"] = returncode
        if returncode:
            raise GateStop(f"distributed trainer exited with return code {returncode}")
        rewards = audit_rewards(repo, reward_dir)
        training_state = audit_training_state(output)
        loadability = audit_output_loadability(output, python, repo)
        tensors = compare_tensors(parent, output, run_dir, python, repo)
        gates = {
            "trainer_completed": True,
            "exact_one_optimizer_step": training_state["global_step"] == 1,
            "online_reward_jsonl_complete": rewards["event_count"] == EXPECTED_EVENTS,
            "online_offline_parser_consistency": rewards["parser_consistency"],
            "positive_offline_reward_variance": rewards["positive_reward_variance"],
            "positive_trainer_reward_std": training_state["trainer_reward_std_positive"],
            "finite_loss": training_state["loss_finite"],
            "finite_nonzero_gradient": training_state["gradient_finite_nonzero"],
            "language_and_freeze_trainability": training_state["trainability_gate_passed"],
            "saved_checkpoint_loadable": loadability["required_files_present"],
            **tensors["gates"],
            "test_not_accessed": True,
            "picked_json_unused": True,
        }
        if not all(gates.values()):
            raise GateStop(f"post-training gate failed: {gates}")
        manifest.update({
            "status": "completed", "gate_passed": True, "completed_at": now_iso(),
            "reward_audit": rewards, "training_audit": training_state,
            "loadability_audit": loadability, "tensor_audit": tensors, "gates": gates,
        })
    except Exception as exc:
        manifest.update({"status": "failed", "gate_passed": False, "failed_at": now_iso(),
                         "failure": {"type": type(exc).__name__, "message": str(exc)}})
        atomic_json(manifest_path, manifest)
        raise
    atomic_json(manifest_path, manifest)
    print(f"[outcome-grpo-gate] PASS: {manifest_path}")


if __name__ == "__main__":
    main()
