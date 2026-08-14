#!/usr/bin/env python3
"""Sequential, resumable public/manual-gated backup of critical model-only snapshots.

The queue waits for two already-running large uploads before starting. It never
deletes local or remote data and explicitly excludes optimizer/scheduler/RNG state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


WORK = Path("/home/dataset-assist-0/czy/wjy")
TASKS = (
    {
        "id": "historical_full_sft3000",
        "repo": "Freddie1946/PathVLM-R1-SFT-n3000-seed42-epoch3-GatedArchive",
        "source": WORK / "pathvlm_r1_v1_a100/transferred_checkpoints/sft_n3000_seed42_epoch03_step1125",
    },
    {
        "id": "full_rule_rl_n8_step1000",
        "repo": "Freddie1946/PathVLM-R1-FullRuleRL-n8-step1000-seed42-GatedArchive",
        "source": WORK / "pathvlm_r1_v1_a100/runs/full_language_rule_rl_clean_n4_n8_step1000_20260812/n8_fresh_step1000/model_snapshots/checkpoint-1000",
    },
    {
        "id": "full_rule_rl_n4_step1000",
        "repo": "Freddie1946/PathVLM-R1-FullRuleRL-n4-step1000-seed42-GatedArchive",
        "source": WORK / "pathvlm_r1_v1_a100/runs/full_language_rule_rl_clean_n4_n8_step1000_20260812/n4_fresh_step1000/model_snapshots/checkpoint-1000",
    },
    {
        "id": "selected_lora_sft3000_step80",
        "repo": "Freddie1946/PathVLM-R1-LoRA-SFT3000-step80-seed42-GatedArchive",
        "source": WORK / "pathvlm_r1_v1_a100/runs/formal_selected_sft3000_20260811/output/checkpoint-80",
    },
    {
        "id": "lora_sft4000_control",
        "repo": "Freddie1946/PathVLM-R1-LoRA-SFT4000-Control-seed42-GatedArchive",
        "source": WORK / "pathvlm_r1_v1_a100/runs/lora_sft4000_control_20260814/formal_8gpu_gbs96/output",
    },
    {
        "id": "gpt4o_stage3_n8_parent_500_1000_1500",
        "repo": "Freddie1946/PathVLM-R1-Stage3-GPT4o-n8-parent-seed42-GatedArchive",
        "source": WORK / "pathvlm_r1_v1_a100/runs/stage3_process_grpo/gpt4o_n8_parent_full_20260813/epoch_model_snapshots",
    },
    {
        "id": "grok43_stage3_500_1000_1500",
        "repo": "Freddie1946/PathVLM-R1-Stage3-Grok43-seed42-GatedArchive",
        "source": WORK / "pathvlm_r1_v1_a100/runs/stage3_process_grpo/grok43_full3epoch_seed42_20260805_attempt02/epoch_model_snapshots",
    },
    {
        "id": "historical_outcome_grpo",
        "repo": "Freddie1946/PathVLM-R1-Outcome-GRPO-n1000-seed42-epoch2-GatedArchive",
        "source": WORK / "pathvlm_r1_v1_a100/transferred_checkpoints/outcome_grpo_n1000_seed42_epoch02_step1000",
    },
    {
        "id": "historical_full_sft4000_control",
        "repo": "Freddie1946/PathVLM-R1-Full-SFT4000-Control-seed42-GatedArchive",
        "source": WORK / "pathvlm_r1_v1_a100/runs/stage2_control_sft4000/n1000_seed0042/sft4000_control_rl1000_seed0042_epoch02_20260729_212746/epoch_snapshots/checkpoint-250",
    },
)

EXCLUDED_NAMES = {"scheduler.pt", "zero_to_fp32.py"}
EXCLUDED_PREFIXES = ("rng_state",)
EXCLUDED_SUFFIXES = ("optim_states.pt", "model_states.pt")


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def excluded(relative: Path) -> bool:
    if ".cache" in relative.parts or any(part.startswith("global_step") for part in relative.parts):
        return True
    name = relative.name
    return (
        name in EXCLUDED_NAMES
        or name.startswith(EXCLUDED_PREFIXES)
        or name.endswith(EXCLUDED_SUFFIXES)
    )


def selected_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise RuntimeError(f"missing source directory: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file() and not excluded(path.relative_to(root)))
    names = {path.name for path in files}
    if not ({"model.safetensors", "model.safetensors.index.json", "adapter_model.safetensors"} & names):
        raise RuntimeError(f"no loadable model or adapter weights under {root}")
    return files


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(task: dict[str, Any], files: list[Path]) -> dict[str, Any]:
    root = task["source"]
    entries = []
    for path in files:
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": 1,
        "created_at": now(),
        "task_id": task["id"],
        "source": str(root),
        "repo_id": task["repo"],
        "visibility": "public_manual_gated",
        "model_only": True,
        "optimizer_scheduler_rng_included": False,
        "file_count": len(entries),
        "total_bytes": sum(item["size_bytes"] for item in entries),
        "files": entries,
    }


def wait_for_pids(pids: list[int], poll_seconds: int, state: dict[str, Any], state_path: Path) -> None:
    while True:
        alive = []
        for pid in pids:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            except PermissionError:
                alive.append(pid)
            else:
                alive.append(pid)
        state["waiting_for_existing_upload_pids"] = alive
        state["updated_at"] = now()
        atomic_json(state_path, state)
        if not alive:
            return
        time.sleep(poll_seconds)


def prepare_repo(api: HfApi, repo: str) -> None:
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    api.update_repo_settings(repo_id=repo, repo_type="model", private=False, gated="manual")
    info = api.model_info(repo)
    if info.private or getattr(info, "gated", False) != "manual":
        raise RuntimeError(f"public/manual-gated policy mismatch for {repo}")


def remote_complete(api: HfApi, task: dict[str, Any], manifest: dict[str, Any]) -> tuple[bool, str | None]:
    try:
        info = api.model_info(task["repo"], files_metadata=True)
    except Exception:
        return False, None
    if info.private or getattr(info, "gated", False) != "manual":
        return False, info.sha
    remote = {item.rfilename: item for item in info.siblings or []}
    for item in manifest["files"]:
        found = remote.get(item["path"])
        if found is None:
            return False, info.sha
        size = getattr(found, "size", None)
        if size is not None and int(size) != item["size_bytes"]:
            return False, info.sha
    return "snapshot_manifest.json" in remote, info.sha


def upload_with_retry(task: dict[str, Any], state: dict[str, Any], state_path: Path, state_root: Path, retry_seconds: int) -> dict[str, Any]:
    api = HfApi()
    files = selected_files(task["source"])
    manifest_path = state_root / "manifests" / f"{task['id']}.json"
    manifest = build_manifest(task, files)
    atomic_json(manifest_path, manifest)
    prepare_repo(api, task["repo"])
    complete, revision = remote_complete(api, task, manifest)
    attempt = 0
    while not complete:
        attempt += 1
        state["tasks"][task["id"]].update({"status": "uploading", "attempt": attempt, "updated_at": now()})
        atomic_json(state_path, state)
        command = [
            "/usr/local/bin/hf", "upload-large-folder", task["repo"], str(task["source"]),
            "--repo-type", "model", "--num-workers", "1", "--no-bars",
            "--exclude", ".cache/**", "--exclude", "global_step*/**",
            "--exclude", "*optim_states.pt", "--exclude", "*model_states.pt",
            "--exclude", "scheduler.pt", "--exclude", "rng_state*.pth",
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode == 0:
            api.upload_file(
                path_or_fileobj=str(manifest_path), path_in_repo="snapshot_manifest.json",
                repo_id=task["repo"], repo_type="model",
                commit_message=f"Add verified model-only manifest for {task['id']}",
            )
            complete, revision = remote_complete(api, task, manifest)
            if complete:
                break
        state["tasks"][task["id"]].update(
            {"status": "retry_wait", "last_returncode": completed.returncode, "updated_at": now()}
        )
        atomic_json(state_path, state)
        time.sleep(retry_seconds)
    return {"revision": revision, "file_count": manifest["file_count"], "total_bytes": manifest["total_bytes"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--wait-pid", action="append", type=int, default=[])
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--retry-seconds", type=int, default=60)
    args = parser.parse_args()
    state_root = args.state_root.resolve()
    state_path = state_root / "state.json"
    state = {
        "schema_version": 1,
        "status": "waiting_for_existing_uploads",
        "started_at": now(),
        "pid": os.getpid(),
        "policy": "critical_model_only_public_manual_gated_no_optimizer_no_delete",
        "tasks": {},
    }
    for task in TASKS:
        state["tasks"][task["id"]] = {
            "status": "queued", "repo_id": task["repo"], "source": str(task["source"])
        }
    atomic_json(state_path, state)
    wait_for_pids(args.wait_pid, args.poll_seconds, state, state_path)
    state.pop("waiting_for_existing_upload_pids", None)
    for task in TASKS:
        state["status"] = "running"
        state["current_task"] = task["id"]
        state["tasks"][task["id"]].update({"status": "hashing_or_verifying", "started_at": now()})
        atomic_json(state_path, state)
        verification = upload_with_retry(
            task, state, state_path, state_root, args.retry_seconds
        )
        state["tasks"][task["id"]].update(
            {"status": "completed", "completed_at": now(), "verification": verification}
        )
        atomic_json(state_path, state)
    state.update({"status": "completed", "current_task": None, "completed_at": now()})
    atomic_json(state_path, state)


if __name__ == "__main__":
    main()
