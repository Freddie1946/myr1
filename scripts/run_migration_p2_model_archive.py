#!/usr/bin/env python3
"""Back up all remaining loadable experiment snapshots to one gated P2 archive.

The inventory is content-deduplicated by model-weight and configuration hashes.
Only directly loadable files are uploaded; optimizer, scheduler, RNG and nested
checkpoint state are excluded.  A stable state file makes the queue resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


WORK = Path("/home/dataset-assist-0/czy/wjy")
RUNS = WORK / "pathvlm_r1_v1_a100/runs"
REPO_ID = "Freddie1946/PathVLM-R1-P2-Model-Snapshot-Archive-20260815"
LOADABLE_MARKERS = {"model.safetensors.index.json", "adapter_model.safetensors"}
EXCLUDED_NAMES = {"README.md", "scheduler.pt", "zero_to_fp32.py"}
EXCLUDED_SUFFIXES = ("optim_states.pt", "model_states.pt")
EXCLUDED_PREFIXES = ("rng_state",)

# These model versions already have a complete remote P1/private archive.
REMOTE_COVERED = (
    "stage3_process_grpo/gpt4o_n8_parent_full_20260813/epoch_model_snapshots/checkpoint-500",
    "stage3_process_grpo/gpt4o_n8_parent_full_20260813/epoch_model_snapshots/checkpoint-1000",
    "stage3_process_grpo/gpt4o_n8_parent_full_20260813/epoch_model_snapshots/checkpoint-1500",
    "formal_selected_sft3000_20260811/output/checkpoint-80",
    "lora_sft4000_control_20260814/formal_8gpu_gbs96/output",
    "lora_sft4000_control_20260814/formal_8gpu_gbs96/output/checkpoint-11",
    "lora_sft4000_control_20260814/formal_8gpu_gbs96/output/checkpoint-22",
    "full_language_rule_rl_clean_n4_n8_step1000_20260812/n4_fresh_step1000/model_snapshots/checkpoint-1000",
    "full_language_rule_rl_clean_n4_n8_step1000_20260812/n8_fresh_step1000/model_snapshots/checkpoint-1000",
    "stage2_control_sft4000/n1000_seed0042/sft4000_control_rl1000_seed0042_epoch02_20260729_212746/epoch_snapshots/checkpoint-250",
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def excluded(path: Path) -> bool:
    name = path.name
    return (
        name in EXCLUDED_NAMES
        or name.startswith(EXCLUDED_PREFIXES)
        or name.endswith(EXCLUDED_SUFFIXES)
        or name.startswith("optimizer")
    )


def direct_files(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_file() and not excluded(path))


def loadable(root: Path) -> bool:
    names = {path.name for path in root.iterdir() if path.is_file()}
    return bool(LOADABLE_MARKERS & names) or "model.safetensors" in names


def weight_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file()
        and (
            path.name == "adapter_model.safetensors"
            or path.name == "model.safetensors"
            or (path.name.startswith("model-") and path.suffix == ".safetensors")
            or path.name.startswith("pytorch_model")
        )
    )


def sha256(path: Path, inode_cache: dict[tuple[int, int, int], str]) -> str:
    stat = path.stat()
    key = (stat.st_dev, stat.st_ino, stat.st_size)
    if key in inode_cache:
        return inode_cache[key]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    inode_cache[key] = value
    return value


def model_fingerprint(
    root: Path, inode_cache: dict[tuple[int, int, int], str]
) -> str | None:
    weights = weight_files(root)
    if not weights:
        return None
    parts = [
        (path.name, path.stat().st_size, sha256(path, inode_cache))
        for path in weights
    ]
    for name in ("config.json", "adapter_config.json"):
        path = root / name
        if path.is_file():
            parts.append((name, path.stat().st_size, sha256(path, inode_cache)))
    return hashlib.sha256(
        json.dumps(parts, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def discover_inventory(state_root: Path) -> dict[str, Any]:
    inode_cache: dict[tuple[int, int, int], str] = {}
    roots = sorted(
        root
        for root, directories, files in os.walk(RUNS)
        if loadable(Path(root))
    )
    covered = {(RUNS / item).resolve() for item in REMOTE_COVERED}
    covered_fingerprints = {
        fingerprint
        for root in covered
        if root.is_dir() and (fingerprint := model_fingerprint(root, inode_cache))
    }
    by_fingerprint: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    for raw in roots:
        root = Path(raw).resolve()
        relative = root.relative_to(RUNS).as_posix()
        fingerprint = model_fingerprint(root, inode_cache)
        if fingerprint is None:
            continue
        if fingerprint in covered_fingerprints:
            aliases[relative] = "REMOTE_COVERED"
            continue
        if fingerprint in by_fingerprint:
            aliases[relative] = by_fingerprint[fingerprint]["id"]
            by_fingerprint[fingerprint]["aliases"].append(relative)
            continue
        files = direct_files(root)
        item_id = f"snapshot_{len(by_fingerprint) + 1:04d}"
        item = {
            "id": item_id,
            "fingerprint": fingerprint,
            "source": str(root),
            "relative_source": relative,
            "aliases": [relative],
            "file_count": len(files),
            "total_bytes": sum(path.stat().st_size for path in files),
            "files": [
                {
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path, inode_cache),
                }
                for path in files
            ],
        }
        by_fingerprint[fingerprint] = item
        aliases[relative] = item_id
        atomic_json(
            state_root / "inventory_progress.json",
            {
                "status": "inventorying",
                "updated_at": now(),
                "discovered_roots": len(roots),
                "processed_roots": len(aliases),
                "unique_snapshots": len(by_fingerprint),
            },
        )
    snapshots = list(by_fingerprint.values())
    return {
        "schema_version": 1,
        "created_at": now(),
        "repo_id": REPO_ID,
        "policy": "model_only_content_deduplicated_no_optimizer_scheduler_rng",
        "discovered_loadable_roots": len(roots),
        "remote_covered_roots": sum(value == "REMOTE_COVERED" for value in aliases.values()),
        "unique_snapshot_count": len(snapshots),
        "total_unique_bytes": sum(item["total_bytes"] for item in snapshots),
        "aliases": aliases,
        "snapshots": snapshots,
    }


def prepare_repo(api: HfApi) -> None:
    api.create_repo(REPO_ID, repo_type="model", private=False, exist_ok=True)
    api.update_repo_settings(
        repo_id=REPO_ID, repo_type="model", private=False, gated="manual"
    )
    info = api.model_info(REPO_ID)
    if info.private or getattr(info, "gated", False) != "manual":
        raise RuntimeError("P2 repository visibility/gate mismatch")


def stage_files(item: dict[str, Any], state_root: Path) -> Path:
    staging = Path(tempfile.mkdtemp(prefix="p2-stage-", dir=state_root))
    source = Path(item["source"])
    for entry in item["files"]:
        src = source / entry["name"]
        dst = staging / entry["name"]
        try:
            os.link(src, dst)
        except OSError:
            os.symlink(src, dst)
    return staging


def remote_complete(api: HfApi, item: dict[str, Any]) -> tuple[bool, str]:
    info = api.model_info(REPO_ID, files_metadata=True)
    remote = {entry.rfilename: entry for entry in info.siblings or []}
    prefix = f"snapshots/{item['id']}"
    for entry in item["files"]:
        found = remote.get(f"{prefix}/{entry['name']}")
        if found is None:
            return False, info.sha
        size = getattr(found, "size", None)
        if size is not None and int(size) != entry["size_bytes"]:
            return False, info.sha
    return f"{prefix}/archive_manifest.json" in remote, info.sha


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--retry-seconds", type=int, default=60)
    args = parser.parse_args()
    state_root = args.state_root.resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    state_path = state_root / "state.json"
    inventory_path = state_root / "p2_inventory.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.is_file()
        else {"schema_version": 1, "started_at": now(), "snapshots": {}}
    )
    state.update({"status": "inventorying", "pid": os.getpid(), "updated_at": now()})
    atomic_json(state_path, state)
    inventory = (
        json.loads(inventory_path.read_text(encoding="utf-8"))
        if inventory_path.is_file()
        else discover_inventory(state_root)
    )
    atomic_json(inventory_path, inventory)
    state.update(
        {
            "status": "uploading",
            "unique_snapshot_count": inventory["unique_snapshot_count"],
            "total_unique_bytes": inventory["total_unique_bytes"],
            "updated_at": now(),
        }
    )
    atomic_json(state_path, state)
    api = HfApi()
    prepare_repo(api)
    api.upload_file(
        path_or_fileobj=str(inventory_path),
        path_in_repo="p2_inventory.json",
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Add P2 deduplicated snapshot inventory",
    )
    for item in inventory["snapshots"]:
        task_state = state["snapshots"].setdefault(item["id"], {})
        if task_state.get("status") == "completed":
            continue
        complete, revision = remote_complete(api, item)
        if complete:
            task_state.update({"status": "completed", "revision": revision, "completed_at": now()})
            atomic_json(state_path, state)
            continue
        attempt = int(task_state.get("attempt", 0))
        while True:
            attempt += 1
            state["current_snapshot"] = item["id"]
            task_state.update(
                {
                    "status": "uploading",
                    "attempt": attempt,
                    "source": item["source"],
                    "total_bytes": item["total_bytes"],
                    "updated_at": now(),
                }
            )
            atomic_json(state_path, state)
            staging: Path | None = None
            try:
                staging = stage_files(item, state_root)
                api.upload_folder(
                    folder_path=str(staging),
                    path_in_repo=f"snapshots/{item['id']}",
                    repo_id=REPO_ID,
                    repo_type="model",
                    commit_message=f"Archive P2 snapshot {item['id']}",
                )
                item_manifest = state_root / "manifests" / f"{item['id']}.json"
                atomic_json(item_manifest, item)
                api.upload_file(
                    path_or_fileobj=str(item_manifest),
                    path_in_repo=f"snapshots/{item['id']}/archive_manifest.json",
                    repo_id=REPO_ID,
                    repo_type="model",
                    commit_message=f"Verify P2 snapshot {item['id']}",
                )
                complete, revision = remote_complete(api, item)
                if not complete:
                    raise RuntimeError("remote file/size verification failed")
            except Exception as error:
                task_state.update(
                    {"status": "retry_wait", "last_error": repr(error), "updated_at": now()}
                )
                atomic_json(state_path, state)
                time.sleep(args.retry_seconds)
                continue
            finally:
                if staging is not None:
                    shutil.rmtree(staging, ignore_errors=True)
            task_state.update({"status": "completed", "revision": revision, "completed_at": now()})
            atomic_json(state_path, state)
            break
    state.update({"status": "completed", "current_snapshot": None, "completed_at": now()})
    atomic_json(state_path, state)


if __name__ == "__main__":
    main()
