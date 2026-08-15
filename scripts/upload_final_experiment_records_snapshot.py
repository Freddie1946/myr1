#!/usr/bin/env python3
"""Upload the final experiment-records snapshot and verify its immutable HF revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download


REPO_ID = "Freddie1946/PathVLM-R1-Migration-Archive-20260815"
PREFIX = "increments/20260815_final_experiment_records_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retry(label: str, operation, attempts: int = 5):
    last = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as error:  # network/provider failures are retried uniformly
            last = error
            if attempt == attempts:
                break
            time.sleep(min(120, 10 * 2 ** (attempt - 1)))
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last!r}") from last


def lfs_sha256(item) -> str | None:
    value = getattr(item, "lfs", None)
    if isinstance(value, dict):
        result = value.get("sha256") or value.get("oid")
    else:
        result = getattr(value, "sha256", None) or getattr(value, "oid", None)
    return result.removeprefix("sha256:") if isinstance(result, str) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.snapshot_dir.resolve()
    verification = json.loads(
        (root / "FINAL_EXPERIMENT_RECORDS_VERIFICATION.json").read_text(encoding="utf-8")
    )
    names = [
        verification["archive"],
        "FINAL_EXPERIMENT_RECORDS_MANIFEST.json",
        "FINAL_EXPERIMENT_RECORDS_AUDIT.json",
        "FINAL_EXPERIMENT_RECORDS_VERIFICATION.json",
        "FINAL_EXPERIMENT_RECORDS_INDEPENDENT_VERIFICATION.json",
    ]
    files = {f"{PREFIX}/{name}": root / name for name in names}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    expected = {
        remote: {"bytes": local.stat().st_size, "sha256": sha256(local)}
        for remote, local in files.items()
    }

    api = HfApi()
    api.create_repo(REPO_ID, repo_type="dataset", private=False, exist_ok=True)
    api.update_repo_settings(REPO_ID, repo_type="dataset", private=False, gated="manual")
    operations = [
        CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
        for remote, local in files.items()
    ]
    commit = retry(
        "upload final experiment-records snapshot",
        lambda: api.create_commit(
            repo_id=REPO_ID,
            repo_type="dataset",
            operations=operations,
            commit_message="Add final experiment-records snapshot and verification",
        ),
    )
    revision = commit.oid
    info = retry(
        "fetch immutable revision metadata",
        lambda: api.dataset_info(REPO_ID, revision=revision, files_metadata=True),
    )
    remote_items = {item.rfilename: item for item in info.siblings or []}
    failures: list[dict[str, str]] = []
    for remote, target in expected.items():
        item = remote_items.get(remote)
        if item is None:
            failures.append({"path": remote, "reason": "missing"})
            continue
        if item.size is not None and int(item.size) != target["bytes"]:
            failures.append({"path": remote, "reason": "remote size mismatch"})
        remote_sha = lfs_sha256(item)
        if remote_sha is not None and remote_sha != target["sha256"]:
            failures.append({"path": remote, "reason": "remote LFS SHA-256 mismatch"})

    downloaded: dict[str, str] = {}
    for name in names[1:]:
        remote = f"{PREFIX}/{name}"
        path = Path(retry(
            f"download verification for {name}",
            lambda remote=remote: hf_hub_download(
                REPO_ID, remote, repo_type="dataset", revision=revision
            ),
        ))
        downloaded[remote] = sha256(path)
        if downloaded[remote] != expected[remote]["sha256"]:
            failures.append({"path": remote, "reason": "downloaded SHA-256 mismatch"})

    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "verified" if not failures else "failed",
        "repository": REPO_ID,
        "repo_type": "dataset",
        "private": info.private,
        "gated": getattr(info, "gated", None),
        "revision": revision,
        "prefix": PREFIX,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in expected.values()),
        "files": expected,
        "download_verified_sha256": downloaded,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        key: result[key] for key in ("status", "revision", "prefix", "file_count", "total_bytes")
    }, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
