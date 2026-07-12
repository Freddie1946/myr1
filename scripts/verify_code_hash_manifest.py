#!/usr/bin/env python3
"""Verify repository files against a timestamped SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    expected = json.loads(args.manifest.read_text(encoding="utf-8"))["sha256"]
    failures = []
    for relative, expected_hash in expected.items():
        path = args.repo_root / relative
        if not path.is_file():
            failures.append({"path": relative, "error": "missing"})
            continue
        actual = sha256(path)
        if actual != expected_hash:
            failures.append({"path": relative, "expected": expected_hash, "actual": actual})
    result = {"checked": len(expected), "failures": failures, "passed": not failures}
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

