#!/usr/bin/env python3
"""Upload and verify external-review references and PathVQA matching results."""

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
HF_REPO = "Freddie1946/PathVLM-R1-Migration-Archive-20260815"
PREFIX = "increments/20260815_external_review_and_pathvqa_matching_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
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
    matching = WORK / "pathvlm_revision_eval_a100/runs/pathvqa_statement_matching_20260815"
    reward = WORK / "pathvlm_revision_eval_a100/human_review/external_reference_reward_review_20260815"
    files: dict[str, Path] = {
        f"{PREFIX}/documents/pathvqa_plip_conch_statement_matching.md": REPO / "docs/20260815_pathvqa_plip_conch_statement_matching.md",
        f"{PREFIX}/documents/human_review_packet_instructions.md": REPO / "docs/20260815_human_review_packet_instructions.md",
        f"{PREFIX}/documents/reviewer_response_manuscript_revision_and_complete_tables.md": REPO / "docs/20260815_reviewer_response_manuscript_revision_and_complete_tables.md",
        f"{PREFIX}/documents/paper_tables.md": REPO / "docs/result_catalog_20260815/paper_tables.md",
        f"{PREFIX}/documents/LATEST.md": REPO / "docs/LATEST.md",
        f"{PREFIX}/human_review/pathvlm_external_reference_second_pass_20260815.tar.gz": WORK / "backup_archives/human_review_distribution_20260815_v3/pathvlm_external_reference_second_pass_20260815.tar.gz",
        f"{PREFIX}/human_review/pathvlm_human_review_assisted_navigation_20260815.tar.gz": WORK / "backup_archives/human_review_distribution_20260815_v3/pathvlm_human_review_assisted_navigation_20260815.tar.gz",
        f"{PREFIX}/human_review/reward_external_reference/references.jsonl": reward / "references.jsonl",
        f"{PREFIX}/human_review/reward_external_reference/manifest.json": reward / "manifest.json",
    }
    for backend in ("plip", "conch"):
        local = matching / f"{backend}_full_yesno3362"
        for name in ("run_config.json", "metrics.json", "predictions.jsonl"):
            files[f"{PREFIX}/pathvqa_statement_matching/{backend}/{name}"] = local / name
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    api = HfApi()
    api.update_repo_settings(HF_REPO, repo_type="dataset", private=False, gated="manual")
    for remote_name, local_path in files.items():
        retry(
            f"upload {remote_name}",
            lambda remote_name=remote_name, local_path=local_path: api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_name,
                repo_id=HF_REPO,
                repo_type="dataset",
                commit_message=f"Add external-review/PathVQA asset {local_path.name}",
            ),
        )

    info = retry("fetch metadata", lambda: api.dataset_info(HF_REPO, files_metadata=True))
    remote = {item.rfilename: item for item in info.siblings or []}
    failures = []
    records = {}
    for remote_name, local_path in files.items():
        record = {"bytes": local_path.stat().st_size, "sha256": sha256(local_path)}
        records[remote_name] = record
        item = remote.get(remote_name)
        if item is None:
            failures.append({"path": remote_name, "reason": "missing"})
            continue
        if item.size is not None and int(item.size) != record["bytes"]:
            failures.append({"path": remote_name, "reason": "size mismatch"})
        lfs = getattr(item, "lfs", None)
        oid = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if isinstance(oid, str) and oid.removeprefix("sha256:") != record["sha256"]:
            failures.append({"path": remote_name, "reason": "LFS SHA-256 mismatch"})
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "verified" if not failures else "failed",
        "repository": HF_REPO,
        "private": info.private,
        "gated": getattr(info, "gated", None),
        "revision": info.sha,
        "prefix": PREFIX,
        "file_count": len(records),
        "total_bytes": sum(value["bytes"] for value in records.values()),
        "failures": failures,
        "files": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "revision", "file_count", "total_bytes")}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
