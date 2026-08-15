#!/usr/bin/env python3
"""Upload and remotely verify the bilingual expert-review supplement."""

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
PREFIX = "increments/20260815_bilingual_expert_review_v1"


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
    translations = WORK / "pathvlm_revision_eval_a100/human_review/bilingual_translations_20260815"
    agreement = WORK / "pathvlm_revision_eval_a100/human_review/external_reference_reward_review_20260815/gpt4o_claude_agreement.json"
    archive = WORK / "backup_archives/human_review_distribution_20260815_v3/pathvlm_human_review_bilingual_assisted_20260815.tar.gz"
    files: dict[str, Path] = {
        f"{PREFIX}/human_review/pathvlm_human_review_bilingual_assisted_20260815.tar.gz": archive,
        f"{PREFIX}/human_review/translations/manifest.json": translations / "manifest.json",
        f"{PREFIX}/human_review/translations/reward.jsonl": translations / "reward.jsonl",
        f"{PREFIX}/human_review/translations/roi.jsonl": translations / "roi.jsonl",
        f"{PREFIX}/human_review/translations/generation.jsonl": translations / "generation.jsonl",
        f"{PREFIX}/human_review/gpt4o_claude_agreement.json": agreement,
        f"{PREFIX}/documents/gpt4o_claude_reward_reference_agreement.md": REPO / "docs/20260815_gpt4o_claude_reward_reference_agreement.md",
        f"{PREFIX}/documents/human_review_packet_instructions.md": REPO / "docs/20260815_human_review_packet_instructions.md",
        f"{PREFIX}/documents/final_migration_and_human_review_closure.md": REPO / "docs/20260815_FINAL_MIGRATION_AND_HUMAN_REVIEW_CLOSURE.md",
        f"{PREFIX}/documents/quick_new_machine_codex_resume.md": REPO / "docs/QUICK_NEW_MACHINE_CODEX_RESUME_20260815.md",
        f"{PREFIX}/documents/reviewer_response_manuscript_revision_and_complete_tables.md": REPO / "docs/20260815_reviewer_response_manuscript_revision_and_complete_tables.md",
        f"{PREFIX}/audit/final_revision_assets_audit.json": REPO / "protocol/final_revision_assets_audit_20260815.json",
        f"{PREFIX}/tools/import_expert_reward_exports.py": REPO / "scripts/import_expert_reward_exports.py",
        f"{PREFIX}/tools/analyze_human_reward_score_differences.py": REPO / "scripts/analyze_human_reward_score_differences.py",
        f"{PREFIX}/documents/LATEST.md": REPO / "docs/LATEST.md",
    }
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
                commit_message=f"Add bilingual expert-review asset {local_path.name}",
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
        "total_bytes": sum(record["bytes"] for record in records.values()),
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
