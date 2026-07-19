#!/usr/bin/env python3
"""Run audited deterministic inference on the frozen 385-QA validation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def require_idle_gpu(index: int) -> dict:
    inventory = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    if inventory.returncode:
        raise RuntimeError(inventory.stderr.strip())
    selected = None
    uuid = None
    for line in inventory.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if int(fields[0]) == index:
            uuid = fields[1]
            selected = {
                "index": index, "uuid": uuid, "name": fields[2],
                "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4]),
                "memory_free_mib": int(fields[5]), "utilization_percent": int(fields[6]),
            }
    if selected is None or uuid is None:
        raise ValueError(f"GPU {index} is not visible")
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    if processes.returncode:
        raise RuntimeError(processes.stderr.strip())
    conflicts = []
    for line in processes.stdout.splitlines():
        if not line.strip():
            continue
        process_uuid, pid, name, used = [item.strip() for item in line.split(",")]
        if process_uuid == uuid:
            conflicts.append({"pid": int(pid), "process_name": name, "used_memory_mib": int(used)})
    if conflicts:
        raise RuntimeError(f"GPU {index} has compute processes: {conflicts}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--install-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--parent-manifest", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    install = args.install_root.resolve()
    checkpoint = args.checkpoint.resolve()
    parent_manifest_path = args.parent_manifest.resolve()
    formal_runs_root = (install / "runs/stage1_sft").resolve()
    if formal_runs_root not in parent_manifest_path.parents:
        raise ValueError(f"parent manifest is outside formal SFT runs: {parent_manifest_path}")
    if not parent_manifest_path.is_file():
        raise FileNotFoundError(parent_manifest_path)
    parent = yaml.safe_load(parent_manifest_path.read_text(encoding="utf-8"))
    if parent.get("stage") != "stage1_sft" or parent.get("status") != "completed":
        raise ValueError("validation parent is not a completed formal SFT run")
    if parent.get("formal_result") is not True or parent.get("test_accessed") is not False:
        raise ValueError("validation parent failed formal-result/test-isolation gate")
    recorded_checkpoint = Path(parent.get("outputs", {}).get("final_checkpoint", "")).resolve()
    if checkpoint != recorded_checkpoint:
        raise ValueError(f"checkpoint does not match parent manifest: {checkpoint}")
    if not (checkpoint / "model.safetensors.index.json").is_file():
        raise FileNotFoundError("checkpoint model index is missing")
    if "test" in str(checkpoint).lower():
        raise ValueError("test reference is forbidden in validation execution")

    data = install / "data/pathmmu_image_disjoint_v2/rewritten_records/validation_0385.json"
    records = json.loads(data.read_text(encoding="utf-8"))
    if len(records) != 385:
        raise ValueError(f"expected 385 validation records, found {len(records)}")
    hardware = require_idle_gpu(args.gpu)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = parent_manifest_path.parent / "validation" / f"validation_0385_{timestamp}"
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    results = run_dir / "results"
    command = [
        sys.executable, str(repo / "scripts/infer_and_score_pathmmu.py"),
        "--model", str(checkpoint), "--data", str(data),
        "--output-dir", str(results), "--max-new-tokens", "192",
    ]
    (run_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1, "run_id": run_dir.name, "status": "running",
        "stage": "stage1_validation", "formal_result": False,
        "created_at": datetime.now().astimezone().isoformat(),
        "parent": {"manifest": str(parent_manifest_path), "manifest_sha256": sha256(parent_manifest_path),
                   "checkpoint": str(checkpoint)},
        "data": {"version": "pathmmu_image_disjoint_v2", "split": "validation_0385",
                 "count": 385, "path": str(data), "sha256": sha256(data)},
        "generation": {"do_sample": False, "max_new_tokens": 192},
        "hardware": {"host": socket.gethostname(), "gpu": hardware},
        "provenance": {"repo_code_manifest": str(repo / "protocol/code_hash_manifest_20260719_172547.json")},
        "test_accessed": False,
    }
    manifest_path = run_dir / "run_manifest.yaml"
    write_yaml(manifest_path, manifest)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(args.gpu), "HF_HUB_OFFLINE": "1", "WANDB_MODE": "disabled"})
    env["PYTHONPATH"] = str(repo / "scripts") + os.pathsep + env.get("PYTHONPATH", "")
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        env.pop(key, None)
    try:
        with (run_dir / "inference.log").open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(
                command, cwd=repo, env=env, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            returncode = process.wait()
        if returncode:
            raise RuntimeError(f"validation inference failed with exit {returncode}")
        metrics_path = results / "metrics.json"
        predictions_path = results / "predictions.jsonl"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        prediction_count = sum(1 for line in predictions_path.open(encoding="utf-8") if line.strip())
        gates = {
            "inference_completed": True,
            "exact_validation_count": metrics.get("count") == 385 and prediction_count == 385,
            "deterministic_decoding": metrics.get("do_sample") is False,
            "test_not_accessed": metrics.get("test_accessed") is False,
            "raw_predictions_saved": predictions_path.is_file(),
        }
        manifest.update({
            "metrics": metrics, "gates": gates,
            "outputs": {"metrics": str(metrics_path), "predictions": str(predictions_path)},
            "status": "completed" if all(gates.values()) else "failed_gate",
            "formal_result": all(gates.values()),
            "completed_at": datetime.now().astimezone().isoformat(),
        })
        write_yaml(manifest_path, manifest)
        if not all(gates.values()):
            raise RuntimeError(f"validation gates failed: {gates}")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["formal_result"] = False
        manifest["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        manifest["completed_at"] = datetime.now().astimezone().isoformat()
        write_yaml(manifest_path, manifest)
        raise
    print(json.dumps({"run_dir": str(run_dir), "metrics": manifest["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
