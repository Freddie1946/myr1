#!/usr/bin/env python3
"""Legacy helper for data-ratio model uploads.

Automatic weight backup for small data-ratio ablations was retired on 2026-08-09.
Metrics, logs and manifests remain backed up, while weights stay local unless a
specific final checkpoint is explicitly selected for a later manual backup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


MODEL_NAMES = {
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "model.safetensors.index.json",
    "pathvlm_ratio_train_state_audit.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "train_results.json",
    "trainer_state.json",
    "training_args.bin",
    "vocab.json",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_files(root: Path) -> list[Path]:
    files = sorted(
        path
        for path in root.iterdir()
        if path.is_file()
        and (
            path.name in MODEL_NAMES
            or (path.name.startswith("model-") and path.name.endswith(".safetensors"))
        )
    )
    names = {path.name for path in files}
    required = {
        "config.json",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
    }
    if not required.issubset(names):
        raise RuntimeError(f"incomplete model at {root}: {sorted(required - names)}")
    if not any(path.name.endswith(".safetensors") for path in files):
        raise RuntimeError(f"no model weights at {root}")
    return files


def lfs_sha(item: Any) -> str | None:
    lfs = getattr(item, "lfs", None)
    if isinstance(lfs, dict):
        return lfs.get("sha256")
    return getattr(lfs, "sha256", None)


def verify_remote(api: HfApi, repo_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    info = api.repo_info(repo_id, repo_type="model", files_metadata=True)
    remote = {item.rfilename: item for item in (info.siblings or [])}
    errors: list[str] = []
    for entry in manifest["files"]:
        item = remote.get(entry["name"])
        if item is None:
            errors.append(f"missing:{entry['name']}")
            continue
        size = getattr(item, "size", None)
        if size is not None and int(size) != entry["size_bytes"]:
            errors.append(f"size:{entry['name']}:{size}!={entry['size_bytes']}")
        if entry["name"].endswith(".safetensors"):
            digest = lfs_sha(item)
            if digest and digest != entry["sha256"]:
                errors.append(f"sha256:{entry['name']}:{digest}!={entry['sha256']}")
    if "snapshot_manifest.json" not in remote:
        errors.append("missing:snapshot_manifest.json")
    if errors:
        raise RuntimeError(f"remote verification failed for {repo_id}: {errors}")
    return {
        "revision": info.sha,
        "private": bool(info.private),
        "remote_file_count": len(remote),
        "verified_file_count": len(manifest["files"]),
        "verified_total_bytes": manifest["total_bytes"],
    }


def build_manifest(task: dict[str, Any], files: list[Path]) -> dict[str, Any]:
    entries = []
    for path in files:
        entries.append(
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    serialization = "".join(
        f"{entry['name']}\t{entry['size_bytes']}\t{entry['sha256']}\n"
        for entry in entries
    )
    return {
        "schema_version": 1,
        "created_at": now_iso(),
        "task_id": task["task_id"],
        "source": str(task["source"]),
        "repo_id": task["repo_id"],
        "model_only": True,
        "resumable": False,
        "formal_training_complete": True,
        "files": entries,
        "file_count": len(entries),
        "total_bytes": sum(entry["size_bytes"] for entry in entries),
        "aggregate_manifest_sha256": hashlib.sha256(serialization.encode()).hexdigest(),
    }


def upload_one(api: HfApi, hf_bin: str, task: dict[str, Any], state_root: Path) -> dict[str, Any]:
    root = task["source"]
    files = selected_files(root)
    manifest_path = state_root / "manifests" / f"{task['task_id']}.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = build_manifest(task, files)
        atomic_json(manifest_path, manifest)

    api.create_repo(task["repo_id"], repo_type="model", private=True, exist_ok=True)
    command = [
        hf_bin,
        "upload-large-folder",
        task["repo_id"],
        str(root),
        "--repo-type",
        "model",
        "--private",
        "--num-workers",
        "2",
        "--no-bars",
    ]
    for path in files:
        command.extend(["--include", path.name])
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise RuntimeError(f"HF upload exited {completed.returncode} for {task['repo_id']}")
    api.upload_file(
        path_or_fileobj=str(manifest_path),
        path_in_repo="snapshot_manifest.json",
        repo_id=task["repo_id"],
        repo_type="model",
        commit_message=f"Add verified model-only manifest for {task['task_id']}",
    )
    return verify_remote(api, task["repo_id"], manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--hf-bin", default="/usr/local/bin/hf")
    args = parser.parse_args()
    formal_root = args.formal_root.resolve()
    state_root = args.state_root.resolve()
    state_path = state_root / "state.json"
    atomic_json(
        state_path,
        {
            "schema_version": 1,
            "status": "retired_results_only_policy",
            "updated_at": now_iso(),
            "pid": os.getpid(),
            "formal_root": str(formal_root),
            "weights_uploaded": False,
            "reason": (
                "Small data-ratio ablation weights remain local; only results and logs "
                "are backed up. A final critical checkpoint must be selected explicitly."
            ),
        },
    )
    print("automatic small-ablation model backup is retired; no weights uploaded")
    return

    tasks = [
        {
            "task_id": "sft0750_rl0250_sft",
            "repo_id": "Freddie1946/PathVLM-R1-Ratio-SFT0750-seed42",
        },
        {
            "task_id": "sft0750_rl0250_rule_rl",
            "repo_id": "Freddie1946/PathVLM-R1-Ratio-SFT0750-RuleRL0250-seed42",
        },
        {
            "task_id": "sft0500_rl0500_sft",
            "repo_id": "Freddie1946/PathVLM-R1-Ratio-SFT0500-seed42",
        },
        {
            "task_id": "sft0500_rl0500_rule_rl",
            "repo_id": "Freddie1946/PathVLM-R1-Ratio-SFT0500-RuleRL0500-seed42",
        },
        {
            "task_id": "sft0250_rl0750_sft",
            "repo_id": "Freddie1946/PathVLM-R1-Ratio-SFT0250-seed42",
        },
    ]
    for task in tasks:
        task["source"] = formal_root / "tasks" / task["task_id"] / "output"

    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {"schema_version": 1, "status": "running", "tasks": {}, "created_at": now_iso()}
    )
    state.update({"status": "running", "pid": os.getpid(), "updated_at": now_iso()})
    atomic_json(state_path, state)
    api = HfApi()
    for task in tasks:
        previous = state["tasks"].get(task["task_id"], {})
        if previous.get("status") == "completed":
            continue
        state["current_task"] = task["task_id"]
        state["tasks"][task["task_id"]] = {
            "status": "running",
            "repo_id": task["repo_id"],
            "started_at": now_iso(),
        }
        atomic_json(state_path, state)
        try:
            verification = upload_one(api, args.hf_bin, task, state_root)
        except Exception as exc:
            state["status"] = "failed"
            state["tasks"][task["task_id"]].update(
                {
                    "status": "failed",
                    "failed_at": now_iso(),
                    "failure": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            state["updated_at"] = now_iso()
            atomic_json(state_path, state)
            raise
        state["tasks"][task["task_id"]].update(
            {"status": "completed", "completed_at": now_iso(), "verification": verification}
        )
        state["updated_at"] = now_iso()
        atomic_json(state_path, state)
    state.update({"status": "completed", "completed_at": now_iso(), "current_task": None})
    atomic_json(state_path, state)
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"backup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
