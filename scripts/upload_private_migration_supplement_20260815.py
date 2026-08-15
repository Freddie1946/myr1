#!/usr/bin/env python3
"""Upload and verify the private migration supplement on Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "Freddie1946/PathVLM-R1-Private-Migration-Supplement-20260815"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retry(operation, attempts: int = 5):
    last_error = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as error:  # network/provider errors vary by client version
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(min(120, 10 * 2**attempt))
    raise RuntimeError(f"operation failed after {attempts} attempts: {last_error!r}") from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
    api.update_repo_settings(REPO_ID, repo_type="dataset", private=True)
    prefix = f"snapshots/{args.tag}"
    local_files = [args.archive.resolve(), args.manifest.resolve(), args.verification.resolve()]
    files = {f"{prefix}/{path.name}": path for path in local_files}
    for remote_path, local_path in files.items():
        retry(lambda remote_path=remote_path, local_path=local_path: api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"Add private migration supplement {args.tag}",
        ))

    info = retry(lambda: api.dataset_info(REPO_ID, files_metadata=True))
    remote_files = {item.rfilename: item for item in info.siblings or []}
    failures: list[str] = []
    expected: dict[str, dict] = {}
    for remote_path, local_path in files.items():
        local_hash = sha256(local_path)
        expected[remote_path] = {"bytes": local_path.stat().st_size, "sha256": local_hash}
        item = remote_files.get(remote_path)
        if item is None:
            failures.append(f"{remote_path}:missing")
            continue
        if item.size is not None and int(item.size) != local_path.stat().st_size:
            failures.append(f"{remote_path}:size")
        lfs = getattr(item, "lfs", None)
        lfs_hash = getattr(lfs, "sha256", None) if lfs is not None else None
        if lfs_hash is not None and lfs_hash != local_hash:
            failures.append(f"{remote_path}:lfs_sha256")

    for metadata_name in (args.manifest.name, args.verification.name):
        remote_path = f"{prefix}/{metadata_name}"
        downloaded = Path(retry(lambda remote_path=remote_path: hf_hub_download(
            REPO_ID, remote_path, repo_type="dataset", revision=info.sha
        )))
        if sha256(downloaded) != expected[remote_path]["sha256"]:
            failures.append(f"{remote_path}:download_sha256")

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "verified" if not failures and info.private else "failed",
        "repository": REPO_ID,
        "repo_type": "dataset",
        "private": info.private,
        "revision": info.sha,
        "prefix": prefix,
        "files": expected,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] != "verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
