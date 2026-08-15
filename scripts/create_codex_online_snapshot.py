#!/usr/bin/env python3
"""Create a secret-free online snapshot of a personal Codex home.

SQLite databases are copied through SQLite's online backup API so their main
database and WAL state are consistent.  Session/history JSONL files are copied
as immutable point-in-time prefixes.  Authentication and caches are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


EXCLUDED_ROOT_NAMES = {
    "auth.json",
    "cache",
    "packages",
    "plugins",
    ".tmp",
    "tmp",
}
COPY_ROOT_FILES = {"config.toml", "history.jsonl"}
COPY_ROOT_DIRS = {"sessions", "rules", "skills", "memories"}
COPY_SQLITE_FILES = {"state_5.sqlite", "goals_1.sqlite", "memories_1.sqlite"}
SANITIZED_TEXT_SUFFIXES = {".jsonl", ".json", ".toml", ".txt", ".md", ".sh"}
SECRET_PATTERNS = {
    "huggingface_token": re.compile(rb"hf_[A-Za-z0-9]{24,}"),
    "openai_style_key": re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    "github_pat": re.compile(rb"(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,})"),
    "bearer_token": re.compile(rb"(?i)authorization\s*[:=]\s*bearer\s+[A-Za-z0-9._~+/-]{16,}"),
    "assigned_api_key": re.compile(
        rb"(?i)(?:api[_-]?key|token)\s*[\"']?\s*[:=]\s*[\"'][A-Za-z0-9._~+/-]{16,}[\"']"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path, redactions: list[dict]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() not in SANITIZED_TEXT_SUFFIXES:
        shutil.copy2(source, destination)
        return
    payload = source.read_bytes()
    labels: dict[str, int] = {}
    for label, pattern in SECRET_PATTERNS.items():
        payload, count = pattern.subn(f"[REDACTED:{label}]".encode(), payload)
        if count:
            labels[label] = count
    destination.write_bytes(payload)
    shutil.copystat(source, destination)
    if labels:
        redactions.append({"source": str(source), "patterns": labels})


def copy_tree(source: Path, destination: Path, redactions: list[dict]) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file() and not path.is_symlink():
            copy_file(path, target, redactions)


def sqlite_backup(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src:
        with sqlite3.connect(destination) as dst:
            src.backup(dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codex-home", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.codex_home.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    snapshot_root = output / "codex-home-snapshot"
    snapshot_root.mkdir()
    started = datetime.now(timezone.utc).isoformat()
    redactions: list[dict] = []

    for path in sorted(source.iterdir()):
        if path.name in EXCLUDED_ROOT_NAMES:
            continue
        if path.is_file() and path.name in COPY_SQLITE_FILES:
            sqlite_backup(path, snapshot_root / path.name)
        elif path.is_file() and path.name in COPY_ROOT_FILES:
            copy_file(path, snapshot_root / path.name, redactions)
        elif path.is_dir() and path.name in COPY_ROOT_DIRS:
            copy_tree(path, snapshot_root / path.name, redactions)

    unresolved: list[dict] = []
    for path in sorted(item for item in snapshot_root.rglob("*") if item.is_file()):
        payload = path.read_bytes()
        labels = sorted(label for label, pattern in SECRET_PATTERNS.items() if pattern.search(payload))
        if labels:
            unresolved.append({"path": str(path.relative_to(snapshot_root)), "patterns": labels})
    if unresolved:
        raise RuntimeError(f"unresolved token-shaped content after sanitization: {unresolved}")

    records = []
    for path in sorted(p for p in snapshot_root.rglob("*") if p.is_file()):
        records.append({
            "path": str(path.relative_to(snapshot_root)),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    completed = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1,
        "snapshot_mode": "online_consistent_sqlite_plus_point_in_time_file_prefixes",
        "source_codex_home": str(source),
        "started_at": started,
        "completed_at": completed,
        "file_count": len(records),
        "total_bytes": sum(item["bytes"] for item in records),
        "excluded": sorted(EXCLUDED_ROOT_NAMES),
        "excluded_additional": ["logs_2.sqlite", "shell_snapshots", "all SQLite files not in the explicit allowlist"],
        "sqlite_allowlist": sorted(COPY_SQLITE_FILES),
        "redactions": redactions,
        "credential_policy": (
            "auth.json, credential/cache material, logs database and shell snapshots are excluded; "
            "token-shaped strings in copied text/session files are redacted; reauthenticate on destination"
        ),
        "continuation_note": "The source Codex session may continue after this snapshot. Create a later incremental or sealed snapshot to capture subsequent turns.",
        "files": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(output),
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "started_at": started,
        "completed_at": completed,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
