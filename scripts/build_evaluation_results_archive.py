#!/usr/bin/env python3
"""Build a secret-free, hash-addressed archive of completed evaluation results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path


SIBLINGS = {
    "metrics.json", "predictions.jsonl", "run_config.json",
    "full_integrity_verified.json", "complete_summary.json",
    "summary.json", "manifest.json",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    runs = workspace / "pathvlm_revision_eval_a100" / "runs"
    repo = workspace / "myr1"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    selected: set[Path] = set()
    for metrics in runs.rglob("metrics.json"):
        relative = metrics.relative_to(runs)
        if any("smoke" in part.lower() for part in relative.parts):
            continue
        selected.add(metrics)
        for name in SIBLINGS - {"metrics.json"}:
            candidate = metrics.parent / name
            if candidate.is_file():
                selected.add(candidate)
    # Some completed analyses (notably fixed-seed inference repeats) place their
    # aggregate summary one directory above per-run metrics. Include these
    # standalone summaries explicitly; unfinished prediction streams still have
    # no summary and therefore remain excluded.
    for name in ("summary.json", "complete_summary.json", "full_integrity_verified.json"):
        for candidate in runs.rglob(name):
            relative = candidate.relative_to(runs)
            if any("smoke" in part.lower() for part in relative.parts):
                continue
            selected.add(candidate)
    for pattern in ("docs/2026081*.md", "protocol/*2026081*.json"):
        selected.update(path for path in repo.glob(pattern) if path.is_file())

    entries = []
    for path in sorted(selected):
        if path.is_relative_to(runs):
            archive_name = Path("evaluation_runs") / path.relative_to(runs)
        else:
            archive_name = Path("repository") / path.relative_to(repo)
        entries.append({
            "source": str(path), "archive_path": archive_name.as_posix(),
            "bytes": path.stat().st_size, "sha256": digest(path),
        })

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "secret-free evaluation results and migration backup",
        "contains_model_weights": False,
        "contains_credentials": False,
        "selection": (
            "completed non-smoke metrics and sibling predictions/config/integrity files; "
            "large RISE case tensors and unfinished runs excluded"
        ),
        "file_count": len(entries),
        "total_uncompressed_bytes": sum(item["bytes"] for item in entries),
        "files": entries,
        "conversation_history_boundary": (
            "The platform raw transcript is not a workspace file and cannot be exported. "
            "Repository handoff documents preserve decisions, protocols and resume state."
        ),
    }
    manifest_path = output / "ARCHIVE_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    archive_path = output / "evaluation_results_20260814.tar.gz"
    with tarfile.open(archive_path, "w:gz", compresslevel=6) as archive:
        for item in entries:
            archive.add(item["source"], arcname=item["archive_path"], recursive=False)
        archive.add(manifest_path, arcname="ARCHIVE_MANIFEST.json", recursive=False)
    verification = {
        "archive": archive_path.name,
        "bytes": archive_path.stat().st_size,
        "sha256": digest(archive_path),
        "manifest_sha256": digest(manifest_path),
        "file_count": len(entries),
        "total_uncompressed_bytes": manifest["total_uncompressed_bytes"],
    }
    (output / "ARCHIVE_VERIFICATION.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
