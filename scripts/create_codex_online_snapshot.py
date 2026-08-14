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
COPY_ROOT_DIRS = {"sessions", "rules", "skills", "shell_snapshots", "memories"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


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

    for path in sorted(source.iterdir()):
        if path.name in EXCLUDED_ROOT_NAMES:
            continue
        if path.is_file() and path.suffix == ".sqlite":
            sqlite_backup(path, snapshot_root / path.name)
        elif path.is_file() and path.name in COPY_ROOT_FILES:
            copy_file(path, snapshot_root / path.name)
        elif path.is_dir() and path.name in COPY_ROOT_DIRS:
            shutil.copytree(path, snapshot_root / path.name, copy_function=shutil.copy2)

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
        "credential_policy": "auth.json and credential/cache material intentionally excluded; reauthenticate on destination",
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
