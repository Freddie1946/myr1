#!/usr/bin/env python3
"""Build a private, secret-scanned supplement for migration-only source material."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path("/home/dataset-assist-0/czy/wjy")
SOURCE_ROOTS = {
    "rewrite_docs": WORKSPACE / "rewrite_docs",
}
TEXT_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".toml"}
SECRET_PATTERNS = {
    "huggingface_token": re.compile(rb"hf_[A-Za-z0-9]{24,}"),
    "openai_style_key": re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    "bearer_token": re.compile(rb"(?i)authorization\s*[:=]\s*bearer\s+[A-Za-z0-9._~+/-]{16,}"),
    "assigned_api_key": re.compile(
        rb"(?i)(?:api[_-]?key|token)\s*[\"']?\s*[:=]\s*[\"'][A-Za-z0-9._~+/-]{16,}[\"']"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def secret_findings(path: Path) -> list[str]:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return []
    payload = path.read_bytes()
    return sorted(label for label, pattern in SECRET_PATTERNS.items() if pattern.search(payload))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    entries: list[dict] = []
    failures: list[dict] = []
    for category, root in SOURCE_ROOTS.items():
        if not root.is_dir():
            failures.append({"source": str(root), "reason": "missing_source_root"})
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file() and not item.is_symlink()):
            findings = secret_findings(path)
            if findings:
                failures.append({"source": str(path), "reason": "secret_pattern", "patterns": findings})
                continue
            entries.append({
                "category": category,
                "source": str(path),
                "archive_path": f"private_materials/{category}/{path.relative_to(root).as_posix()}",
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })

    if failures:
        raise RuntimeError(json.dumps(failures, ensure_ascii=False, indent=2))

    created_at = datetime.now(timezone.utc).astimezone().isoformat()
    manifest = {
        "schema_version": 1,
        "created_at": created_at,
        "tag": args.tag,
        "purpose": "private migration supplement containing original manuscript and reviewer source material",
        "visibility_required": "private",
        "credentials_included": False,
        "historical_stage3_draft_included": False,
        "file_count": len(entries),
        "total_bytes": sum(item["bytes"] for item in entries),
        "files": entries,
    }
    manifest_path = output / "PRIVATE_MIGRATION_SUPPLEMENT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    archive_path = output / f"pathvlm_private_migration_supplement_{args.tag}.tar.gz"
    with tarfile.open(archive_path, "w:gz", compresslevel=6) as archive:
        for item in entries:
            archive.add(item["source"], arcname=item["archive_path"], recursive=False)
        archive.add(manifest_path, arcname=manifest_path.name, recursive=False)

    verification = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "archive": archive_path.name,
        "bytes": archive_path.stat().st_size,
        "sha256": sha256(archive_path),
        "manifest_sha256": sha256(manifest_path),
        "payload_file_count": len(entries),
        "payload_bytes": manifest["total_bytes"],
    }
    verification_path = output / "PRIVATE_MIGRATION_SUPPLEMENT_VERIFICATION.json"
    verification_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": manifest, "verification": verification}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
