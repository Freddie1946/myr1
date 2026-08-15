#!/usr/bin/env python3
"""Upload one secret-free Codex online snapshot to the private HF archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "Freddie1946/PathVLM-R1-Codex-Private-Snapshots-Clean-20260815"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retry(operation, attempts: int = 5):
    last = None
    for number in range(attempts):
        try:
            return operation()
        except Exception as error:
            last = error
            if number + 1 < attempts:
                time.sleep(min(120, 10 * 2**number))
    raise RuntimeError(f"operation failed after {attempts} attempts: {last!r}") from last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    api = HfApi()
    repo_id = args.repo_id
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.update_repo_settings(repo_id, repo_type="dataset", private=True)
    prefix = f"snapshots/{args.tag}"
    files = {
        f"{prefix}/{args.archive.name}": args.archive.resolve(),
        f"{prefix}/manifest.json": args.manifest.resolve(),
    }
    for remote, local in files.items():
        retry(lambda remote=remote, local=local: api.upload_file(
            path_or_fileobj=str(local), path_in_repo=remote, repo_id=repo_id,
            repo_type="dataset", commit_message=f"Add secret-free Codex snapshot {args.tag}",
        ))
    info = retry(lambda: api.dataset_info(repo_id, files_metadata=True))
    if not info.private:
        raise RuntimeError("Codex snapshot repository is not private")
    remote = {item.rfilename: item for item in info.siblings or []}
    failures = []
    expected = {}
    for name, path in files.items():
        expected[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        item = remote.get(name)
        if item is None or (item.size is not None and int(item.size) != path.stat().st_size):
            failures.append(name)
            continue
        lfs = getattr(item, "lfs", None)
        lfs_hash = getattr(lfs, "sha256", None) if lfs is not None else None
        if lfs_hash is not None and lfs_hash != expected[name]["sha256"]:
            failures.append(f"{name}:lfs_sha256")
    downloaded = Path(retry(lambda: hf_hub_download(
        repo_id, f"{prefix}/manifest.json", repo_type="dataset", revision=info.sha
    )))
    if sha256(downloaded) != expected[f"{prefix}/manifest.json"]["sha256"]:
        failures.append(f"{prefix}/manifest.json:sha256")
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "verified" if not failures else "failed",
        "repository": repo_id,
        "private": info.private,
        "revision": info.sha,
        "tag": args.tag,
        "files": expected,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "revision": info.sha, "tag": args.tag}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
