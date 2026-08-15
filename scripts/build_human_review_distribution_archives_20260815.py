#!/usr/bin/env python3
"""Build owner-complete and reviewer-blinded human-review archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path


WORK = Path("/home/dataset-assist-0/czy/wjy")
REPO = WORK / "myr1"
BROWSE = WORK / "pathvlm_revision_eval_a100/human_review/expert_review_browse_packets_20260815"
REWARD_SOURCE = WORK / "pathvlm_revision_eval_a100/human_review/reward_agreement_60case_20260814"
PAIRWISE = WORK / "pathvlm_revision_eval_a100/runs/stage2_stage3_human_review_packet_20260815"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, destination, copy_function=shutil.copy2)


def tree_manifest(root: Path) -> dict:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        records.append({
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return {
        "file_count": len(records),
        "total_bytes": sum(item["bytes"] for item in records),
        "files": records,
    }


def archive_tree(root: Path, destination: Path) -> None:
    with tarfile.open(destination, "w:gz", compresslevel=6) as archive:
        archive.add(root, arcname=root.name, recursive=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)

    owner = out / "owner_complete"
    reviewer = out / "reviewer_blinded"
    owner.mkdir()
    reviewer.mkdir()

    copy_tree(BROWSE, owner / "expert_review_browse_packets")
    copy_tree(REWARD_SOURCE, owner / "reward_agreement_frozen_source")
    copy_tree(PAIRWISE, owner / "stage2_stage3_human_review")
    shutil.copy2(REPO / "docs/20260815_human_review_packet_instructions.md", owner / "README.md")
    shutil.copy2(REPO / "docs/20260815_reviewer_response_manuscript_revision_and_complete_tables.md", owner / "reviewer_response_manuscript_revision_and_complete_tables.md")
    shutil.copy2(REPO / "protocol/final_revision_assets_audit_20260815.json", owner / "final_revision_assets_audit.json")

    copy_tree(BROWSE / "reward_agreement_blinded", reviewer / "reward_agreement_blinded")
    copy_tree(BROWSE / "interpretability_roi_blinded", reviewer / "interpretability_roi_blinded")
    copy_tree(PAIRWISE / "blind_pairwise_multijudge_panel100/review_packet", reviewer / "stage2_stage3_blind_pairwise_100")
    shutil.copy2(REPO / "docs/20260815_human_review_packet_instructions.md", reviewer / "README.md")

    forbidden = ("INTERNAL_DO_NOT_SEND", "answer_key", "source_mapping", "adjudication_key")
    leaked = [
        str(path.relative_to(reviewer))
        for path in reviewer.rglob("*")
        if any(token.lower() in str(path.relative_to(reviewer)).lower() for token in forbidden)
    ]
    if leaked:
        raise RuntimeError(f"reviewer archive contains blinded-key material: {leaked}")

    owner_manifest = tree_manifest(owner)
    reviewer_manifest = tree_manifest(reviewer)
    (owner / "PACKAGE_MANIFEST.json").write_text(json.dumps(owner_manifest, ensure_ascii=False, indent=2) + "\n")
    (reviewer / "PACKAGE_MANIFEST.json").write_text(json.dumps(reviewer_manifest, ensure_ascii=False, indent=2) + "\n")

    owner_tar = out / "pathvlm_human_review_owner_complete_20260815.tar.gz"
    reviewer_tar = out / "pathvlm_human_review_reviewer_blinded_20260815.tar.gz"
    archive_tree(owner, owner_tar)
    archive_tree(reviewer, reviewer_tar)

    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "owner_archive": {
            "path": str(owner_tar), "bytes": owner_tar.stat().st_size,
            "sha256": sha256(owner_tar), **owner_manifest,
        },
        "reviewer_archive": {
            "path": str(reviewer_tar), "bytes": reviewer_tar.stat().st_size,
            "sha256": sha256(reviewer_tar), **reviewer_manifest,
        },
        "reviewer_forbidden_material_check": "passed",
        "human_ratings": "not included; pending expert work",
    }
    (out / "archive_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "owner_bytes": result["owner_archive"]["bytes"],
        "reviewer_bytes": result["reviewer_archive"]["bytes"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
