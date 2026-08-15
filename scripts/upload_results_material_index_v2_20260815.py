#!/usr/bin/env python3
"""Upload and verify the corrected tables, material index, and self-contained human packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


WORK = Path("/home/dataset-assist-0/czy/wjy")
REPO = WORK / "myr1"
ARCHIVE = WORK / "backup_archives/human_review_distribution_20260815_v2"
HF_REPO = "Freddie1946/PathVLM-R1-Migration-Archive-20260815"
PREFIX = "increments/20260815_results_material_index_v2"


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
        except Exception as error:
            last = error
            if attempt == attempts:
                break
            time.sleep(min(120, 10 * 2 ** (attempt - 1)))
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last!r}") from last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = {
        f"{PREFIX}/documents/reviewer_response_manuscript_revision_and_complete_tables.md": REPO / "docs/20260815_reviewer_response_manuscript_revision_and_complete_tables.md",
        f"{PREFIX}/documents/human_review_packet_instructions.md": REPO / "docs/20260815_human_review_packet_instructions.md",
        f"{PREFIX}/result_catalog/paper_tables.md": REPO / "docs/result_catalog_20260815/paper_tables.md",
        f"{PREFIX}/result_catalog/result_lineage.json": REPO / "docs/result_catalog_20260815/result_lineage.json",
        f"{PREFIX}/human_review/archive_manifest.json": ARCHIVE / "archive_manifest.json",
        f"{PREFIX}/human_review/pathvlm_human_review_owner_complete_20260815.tar.gz": ARCHIVE / "pathvlm_human_review_owner_complete_20260815.tar.gz",
        f"{PREFIX}/human_review/pathvlm_human_review_reviewer_blinded_20260815.tar.gz": ARCHIVE / "pathvlm_human_review_reviewer_blinded_20260815.tar.gz",
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    api = HfApi()
    api.update_repo_settings(HF_REPO, repo_type="dataset", private=False, gated="manual")
    for remote, local in files.items():
        retry(
            f"upload {remote}",
            lambda remote=remote, local=local: api.upload_file(
                path_or_fileobj=str(local), path_in_repo=remote, repo_id=HF_REPO,
                repo_type="dataset", commit_message=f"Add corrected result/material asset {local.name}",
            ),
        )

    info = retry("fetch remote metadata", lambda: api.dataset_info(HF_REPO, files_metadata=True))
    remote = {item.rfilename: item for item in info.siblings or []}
    failures = []
    expected = {}
    for name, path in files.items():
        record = {"bytes": path.stat().st_size, "sha256": sha256(path)}
        expected[name] = record
        item = remote.get(name)
        if item is None:
            failures.append({"path": name, "reason": "missing"})
            continue
        if item.size is not None and int(item.size) != record["bytes"]:
            failures.append({"path": name, "reason": f"size={item.size}, expected={record['bytes']}"})
        lfs = getattr(item, "lfs", None)
        oid = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if isinstance(oid, str) and oid.removeprefix("sha256:") != record["sha256"]:
            failures.append({"path": name, "reason": "remote LFS SHA-256 mismatch"})

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
        "failures": failures,
        "files": expected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "revision", "file_count", "total_bytes")}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
