#!/usr/bin/env python3
"""Wait for the active evaluation wave and upload an immutable gated increment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


WORK = Path("/home/dataset-assist-0/czy/wjy")
REPO_ID = "Freddie1946/PathVLM-R1-Migration-Archive-20260815"
PREFIX = "increments/20260815_gpu_evaluations_v1"
STAGE2 = WORK / "pathvlm_revision_eval_a100/runs/stage2_continued_rule_rl1000_core_eval_20260815"
REPEATED = WORK / "pathvlm_revision_eval_a100/runs/repeated_inference_pathmmu_test999_20260815"
GPU_STATE_ROOT = WORK / "pathvlm_r1_v1_a100/reports/gpu_eval_continuation_20260815"
RULE4000 = WORK / "pathvlm_revision_eval_a100/runs/base_rule_rl4000_checkpoint2500_uniform_eval_20260815"
LLAVA = WORK / "pathvlm_revision_eval_a100/runs/llava_med_formal_20260814"
SOURCES = {
    "stage2_continued_rule_rl1000_core_eval": STAGE2,
    "repeated_pathmmu_test999": REPEATED,
    "gpu_queue_state": GPU_STATE_ROOT,
    "rule_rl4000_checkpoint2500_uniform_eval": RULE4000,
    "llava_med_formal": LLAVA,
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def complete() -> bool:
    stage2_metrics = [
        STAGE2 / "pathmmu_val385/metrics.json",
        STAGE2 / "pathmmu_test999/metrics.json",
        STAGE2 / "pathvqa_test3362_yesno/metrics.json",
        STAGE2 / "pathvqa_test3362_ab/metrics.json",
    ]
    if not all(path.is_file() for path in stage2_metrics):
        return False
    base_summary = REPEATED / "base_qwen2_5_vl_7b/summary.json"
    queue_state = GPU_STATE_ROOT / "state.json"
    if not base_summary.is_file() or not queue_state.is_file():
        return False
    try:
        return json.loads(queue_state.read_text(encoding="utf-8")).get("status") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def files_under(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--retry-seconds", type=int, default=60)
    args = parser.parse_args()
    state_root = args.state_root.resolve()
    state_path = state_root / "state.json"
    state: dict[str, Any] = {
        "schema_version": 1, "status": "waiting_for_evaluations", "started_at": now(),
        "repo_id": REPO_ID, "path_in_repo": PREFIX,
    }
    write_json(state_path, state)
    while not complete():
        state["updated_at"] = now()
        write_json(state_path, state)
        time.sleep(args.poll_seconds)

    manifest: dict[str, Any] = {
        "schema_version": 1, "created_at": now(), "repo_id": REPO_ID,
        "path_in_repo": PREFIX, "sources": {}, "files": [],
    }
    for label, root in SOURCES.items():
        if not root.is_dir():
            continue
        selected = files_under(root)
        manifest["sources"][label] = str(root)
        for path in selected:
            manifest["files"].append({
                "path": f"{label}/{path.relative_to(root).as_posix()}",
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    manifest["file_count"] = len(manifest["files"])
    manifest["total_bytes"] = sum(item["size_bytes"] for item in manifest["files"])
    manifest_path = state_root / "increment_manifest.json"
    write_json(manifest_path, manifest)

    api = HfApi()
    api.create_repo(REPO_ID, repo_type="dataset", private=False, exist_ok=True)
    api.update_repo_settings(REPO_ID, repo_type="dataset", private=False, gated="manual")
    for label, root in SOURCES.items():
        if not root.is_dir():
            continue
        attempt = 0
        while True:
            attempt += 1
            state.update({"status": "uploading", "current_source": label, "attempt": attempt, "updated_at": now()})
            write_json(state_path, state)
            try:
                api.upload_folder(
                    folder_path=str(root), repo_id=REPO_ID, repo_type="dataset",
                    path_in_repo=f"{PREFIX}/{label}",
                    ignore_patterns=[".cache/**"],
                    commit_message=f"Add immutable evaluation increment: {label}",
                )
                break
            except Exception as error:
                state.update({"status": "retry_wait", "last_error": repr(error), "updated_at": now()})
                write_json(state_path, state)
                time.sleep(args.retry_seconds)
    api.upload_file(
        path_or_fileobj=str(manifest_path), path_in_repo=f"{PREFIX}/increment_manifest.json",
        repo_id=REPO_ID, repo_type="dataset", commit_message="Add evaluation increment manifest",
    )
    info = api.dataset_info(REPO_ID, files_metadata=True, token=True)
    remote = {item.rfilename: item for item in info.siblings or []}
    missing = []
    for item in manifest["files"]:
        remote_path = f"{PREFIX}/{item['path']}"
        found = remote.get(remote_path)
        if found is None or (found.size is not None and int(found.size) != item["size_bytes"]):
            missing.append(remote_path)
    if missing:
        raise RuntimeError(f"remote verification failed for {len(missing)} files; first={missing[:3]}")
    state.update({
        "status": "completed", "completed_at": now(), "revision": info.sha,
        "file_count": manifest["file_count"], "total_bytes": manifest["total_bytes"],
    })
    write_json(state_path, state)


if __name__ == "__main__":
    main()
