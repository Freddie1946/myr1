#!/usr/bin/env python3
"""Back up a fresh private Codex snapshot and results, then start P2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


WORK = Path("/home/dataset-assist-0/czy/wjy")
REPO = WORK / "myr1"
P1_STATE = WORK / "pathvlm_r1_v1_a100/reports/migration_critical_model_backup_deduplicated_20260815/state.json"
CODEX_REPO = "Freddie1946/PathVLM-R1-Codex-Private-Snapshots"
RESULT_REPO = "Freddie1946/PathVLM-R1-Migration-Archive-20260815"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retry(state: dict, state_path: Path, label: str, seconds: int, operation) -> None:
    attempt = 0
    while True:
        attempt += 1
        state.update({"status": label, "attempt": attempt, "updated_at": now()})
        atomic_json(state_path, state)
        try:
            operation()
            return
        except Exception as error:
            state.update({"status": f"{label}_retry_wait", "last_error": repr(error), "updated_at": now()})
            atomic_json(state_path, state)
            time.sleep(seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--retry-seconds", type=int, default=60)
    args = parser.parse_args()
    state_root = args.state_root.resolve()
    state_path = state_root / "state.json"
    state = {"schema_version": 1, "started_at": now(), "pid": os.getpid()}
    atomic_json(state_path, state)
    p1 = json.loads(P1_STATE.read_text(encoding="utf-8"))
    if p1.get("status") != "completed":
        raise RuntimeError("P1 must be remotely verified before the post-P1 sequence")
    api = HfApi()

    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = WORK / "backup_archives" / f"codex_online_snapshot_{tag}"
    subprocess.run(
        [
            "python3", str(REPO / "scripts/create_codex_online_snapshot.py"),
            "--codex-home", str(WORK / ".codex-wjy"), "--output-dir", str(snapshot_dir),
        ],
        check=True,
    )
    tar_dir = WORK / "backup_archives/private_codex_snapshots"
    tar_dir.mkdir(parents=True, exist_ok=True)
    tar_path = tar_dir / f"codex_online_snapshot_{tag}.tar.gz"
    with tarfile.open(tar_path, "w:gz", compresslevel=6) as archive:
        archive.add(snapshot_dir, arcname=snapshot_dir.name, recursive=True)
    codex_prefix = f"snapshots/{tag}"

    def upload_codex() -> None:
        api.create_repo(CODEX_REPO, repo_type="dataset", private=True, exist_ok=True)
        api.update_repo_settings(CODEX_REPO, repo_type="dataset", private=True)
        api.upload_file(
            path_or_fileobj=str(tar_path), path_in_repo=f"{codex_prefix}/{tar_path.name}",
            repo_id=CODEX_REPO, repo_type="dataset", commit_message=f"Add Codex snapshot {tag}",
        )
        api.upload_file(
            path_or_fileobj=str(snapshot_dir / "manifest.json"),
            path_in_repo=f"{codex_prefix}/manifest.json", repo_id=CODEX_REPO,
            repo_type="dataset", commit_message=f"Add Codex snapshot manifest {tag}",
        )
        info = api.dataset_info(CODEX_REPO, files_metadata=True)
        remote = {item.rfilename: item for item in info.siblings or []}
        expected = {
            f"{codex_prefix}/{tar_path.name}": tar_path.stat().st_size,
            f"{codex_prefix}/manifest.json": (snapshot_dir / "manifest.json").stat().st_size,
        }
        for name, size in expected.items():
            found = remote.get(name)
            if found is None or (found.size is not None and int(found.size) != size):
                raise RuntimeError(f"Codex remote verification failed: {name}")
        if not info.private:
            raise RuntimeError("Codex snapshot repository is not private")
        state["codex_backup"] = {
            "tag": tag, "archive": str(tar_path), "bytes": tar_path.stat().st_size,
            "sha256": sha256(tar_path), "revision": info.sha,
        }

    retry(state, state_path, "uploading_codex_private_snapshot", args.retry_seconds, upload_codex)
    atomic_json(state_path, state)

    result_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_dir = WORK / "backup_archives" / f"evaluation_results_post_p1_{result_tag}"
    subprocess.run(
        [
            "python3", str(REPO / "scripts/build_evaluation_results_archive.py"),
            "--workspace", str(WORK), "--output-dir", str(result_dir),
        ],
        check=True,
    )
    result_prefix = f"evaluation_snapshots/post_p1_{result_tag}"

    def upload_results() -> None:
        api.create_repo(RESULT_REPO, repo_type="dataset", private=False, exist_ok=True)
        api.update_repo_settings(
            RESULT_REPO, repo_type="dataset", private=False, gated="manual"
        )
        api.upload_folder(
            folder_path=str(result_dir), path_in_repo=result_prefix,
            repo_id=RESULT_REPO, repo_type="dataset",
            commit_message=f"Add post-P1 evaluation snapshot {result_tag}",
        )
        info = api.dataset_info(RESULT_REPO, files_metadata=True)
        remote = {item.rfilename: item for item in info.siblings or []}
        for local in result_dir.iterdir():
            if not local.is_file():
                continue
            name = f"{result_prefix}/{local.name}"
            found = remote.get(name)
            if found is None or (found.size is not None and int(found.size) != local.stat().st_size):
                raise RuntimeError(f"evaluation remote verification failed: {name}")
        if info.private or getattr(info, "gated", False) != "manual":
            raise RuntimeError("evaluation archive gate/visibility mismatch")
        verification = json.loads((result_dir / "ARCHIVE_VERIFICATION.json").read_text())
        state["evaluation_backup"] = {
            "tag": result_tag, "prefix": result_prefix, "revision": info.sha, **verification,
        }

    retry(state, state_path, "uploading_post_p1_evaluation_snapshot", args.retry_seconds, upload_results)
    atomic_json(state_path, state)

    state.update({"status": "running_p2", "p2_started_at": now(), "updated_at": now()})
    atomic_json(state_path, state)
    p2_root = WORK / "pathvlm_r1_v1_a100/reports/migration_p2_model_archive_20260815"
    result = subprocess.run(
        [
            "python3", str(REPO / "scripts/run_migration_p2_model_archive.py"),
            "--state-root", str(p2_root), "--retry-seconds", str(args.retry_seconds),
        ]
    )
    if result.returncode:
        state.update({"status": "p2_failed", "p2_returncode": result.returncode, "updated_at": now()})
        atomic_json(state_path, state)
        raise SystemExit(result.returncode)
    state.update({"status": "completed", "completed_at": now(), "updated_at": now()})
    atomic_json(state_path, state)


if __name__ == "__main__":
    main()
