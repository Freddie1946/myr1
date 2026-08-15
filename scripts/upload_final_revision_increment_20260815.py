#!/usr/bin/env python3
"""Upload and remotely verify the final revision/result/human-review increment."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


WORK = Path("/home/dataset-assist-0/czy/wjy")
REPO = WORK / "myr1"
HF_REPO = "Freddie1946/PathVLM-R1-Migration-Archive-20260815"
PREFIX = "increments/20260815_final_revision_closure_v1"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    api = HfApi()
    api.create_repo(HF_REPO, repo_type="dataset", private=False, exist_ok=True)
    api.update_repo_settings(HF_REPO, repo_type="dataset", private=False, gated="manual")

    files = {
        f"{PREFIX}/documents/reviewer_response_manuscript_revision_and_complete_tables.md": REPO / "docs/20260815_reviewer_response_manuscript_revision_and_complete_tables.md",
        f"{PREFIX}/documents/human_review_packet_instructions.md": REPO / "docs/20260815_human_review_packet_instructions.md",
        f"{PREFIX}/audit/final_revision_assets_audit_20260815.json": REPO / "protocol/final_revision_assets_audit_20260815.json",
        f"{PREFIX}/human_review/archive_manifest.json": WORK / "backup_archives/human_review_distribution_20260815/archive_manifest.json",
        f"{PREFIX}/human_review/pathvlm_human_review_owner_complete_20260815.tar.gz": WORK / "backup_archives/human_review_distribution_20260815/pathvlm_human_review_owner_complete_20260815.tar.gz",
        f"{PREFIX}/human_review/pathvlm_human_review_reviewer_blinded_20260815.tar.gz": WORK / "backup_archives/human_review_distribution_20260815/pathvlm_human_review_reviewer_blinded_20260815.tar.gz",
    }
    for remote, local in files.items():
        if not local.is_file():
            raise FileNotFoundError(local)
        retry(
            f"upload {remote}",
            lambda remote=remote, local=local: api.upload_file(
                path_or_fileobj=str(local), path_in_repo=remote,
                repo_id=HF_REPO, repo_type="dataset",
                commit_message=f"Add final revision closure asset {local.name}",
            ),
        )

    folders = {
        f"{PREFIX}/evaluation/missing_paper_evaluations_20260815": WORK / "pathvlm_revision_eval_a100/runs/missing_paper_evaluations_20260815",
        f"{PREFIX}/result_catalog": REPO / "docs/result_catalog_20260815",
    }
    for remote, local in folders.items():
        retry(
            f"upload folder {remote}",
            lambda remote=remote, local=local: api.upload_folder(
                folder_path=str(local), path_in_repo=remote,
                repo_id=HF_REPO, repo_type="dataset",
                commit_message=f"Add {Path(remote).name}",
                ignore_patterns=["**/.cache/**", "**/__pycache__/**", "**/*.pyc"],
            ),
        )

    info = retry("fetch remote metadata", lambda: api.dataset_info(HF_REPO, files_metadata=True))
    if info.private or getattr(info, "gated", False) != "manual":
        raise RuntimeError(f"visibility mismatch: private={info.private}, gated={getattr(info, 'gated', None)}")
    remote = {item.rfilename: item for item in info.siblings or []}
    expected: dict[str, dict] = {}
    for name, path in files.items():
        expected[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    for prefix, folder in folders.items():
        for path in folder.rglob("*"):
            if not path.is_file() or ".cache" in path.parts or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            name = f"{prefix}/{path.relative_to(folder).as_posix()}"
            expected[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}

    failures = []
    for name, target in expected.items():
        item = remote.get(name)
        if item is None:
            failures.append({"path": name, "reason": "missing"})
            continue
        if item.size is not None and int(item.size) != target["bytes"]:
            failures.append({"path": name, "reason": f"size={item.size}, expected={target['bytes']}"})
            continue
        lfs = getattr(item, "lfs", None)
        lfs_sha = None
        if isinstance(lfs, dict):
            lfs_sha = lfs.get("sha256") or lfs.get("oid")
        elif lfs is not None:
            lfs_sha = getattr(lfs, "sha256", None) or getattr(lfs, "oid", None)
        if isinstance(lfs_sha, str):
            lfs_sha = lfs_sha.removeprefix("sha256:")
            if lfs_sha != target["sha256"]:
                failures.append({"path": name, "reason": "remote LFS SHA-256 mismatch"})

    # Independently download the small audit and combined manuscript document.
    downloaded = {}
    for name in (
        f"{PREFIX}/documents/reviewer_response_manuscript_revision_and_complete_tables.md",
        f"{PREFIX}/audit/final_revision_assets_audit_20260815.json",
        f"{PREFIX}/human_review/archive_manifest.json",
    ):
        path = Path(retry(
            f"download verify {name}",
            lambda name=name: hf_hub_download(HF_REPO, name, repo_type="dataset", revision=info.sha),
        ))
        actual = sha256(path)
        downloaded[name] = actual
        if actual != expected[name]["sha256"]:
            failures.append({"path": name, "reason": "downloaded SHA-256 mismatch"})

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "verified" if not failures else "failed",
        "repository": HF_REPO,
        "repo_type": "dataset",
        "private": info.private,
        "gated": getattr(info, "gated", None),
        "revision": info.sha,
        "prefix": PREFIX,
        "file_count": len(expected),
        "total_bytes": sum(item["bytes"] for item in expected.values()),
        "download_verified_sha256": downloaded,
        "failures": failures,
        "files": expected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("status", "revision", "file_count", "total_bytes")}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
