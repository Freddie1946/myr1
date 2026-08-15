#!/usr/bin/env python3
"""Independently verify a final experiment-records archive and every payload member."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_dir", type=Path)
    args = parser.parse_args()
    root = args.snapshot_dir.resolve()
    manifest_path = root / "FINAL_EXPERIMENT_RECORDS_MANIFEST.json"
    audit_path = root / "FINAL_EXPERIMENT_RECORDS_AUDIT.json"
    verification_path = root / "FINAL_EXPERIMENT_RECORDS_VERIFICATION.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    archive_path = root / verification["archive"]

    outer_checks = {
        "archive_bytes": archive_path.stat().st_size == verification["bytes"],
        "archive_sha256": sha256_file(archive_path) == verification["sha256"],
        "manifest_sha256": sha256_file(manifest_path) == verification["manifest_sha256"],
        "audit_sha256": sha256_file(audit_path) == verification["audit_sha256"],
        "file_count": len(manifest["files"]) == verification["file_count"],
        "uncompressed_bytes": sum(item["bytes"] for item in manifest["files"])
        == verification["total_uncompressed_bytes"],
    }
    if not all(outer_checks.values()):
        raise SystemExit(f"outer verification failed: {outer_checks}")

    expected = {item["archive_path"]: item for item in manifest["files"]}
    seen: set[str] = set()
    metadata_members = {
        "FINAL_EXPERIMENT_RECORDS_MANIFEST.json",
        "FINAL_EXPERIMENT_RECORDS_AUDIT.json",
    }
    unexpected: list[str] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() and not member.islnk():
                unexpected.append(member.name)
                continue
            if member.name in metadata_members:
                continue
            item = expected.get(member.name)
            if item is None or member.name in seen:
                unexpected.append(member.name)
                continue
            if member.islnk():
                target = expected.get(member.linkname)
                if target is None:
                    raise SystemExit(
                        f"hard-link target is outside the declared payload: "
                        f"{member.name} -> {member.linkname}"
                    )
                if target["bytes"] != item["bytes"] or target["sha256"] != item["sha256"]:
                    raise SystemExit(
                        f"hard-link manifest mismatch: {member.name} -> {member.linkname}"
                    )
                # tarfile resolves a hard-link by seeking back through the compressed stream,
                # which is prohibitively slow for a multi-GiB archive. The builder only emits
                # links to an earlier member: require that target to have already passed its
                # content hash, then validate the declared link against the same manifest digest.
                if member.linkname not in seen:
                    raise SystemExit(
                        f"hard-link target was not verified first: {member.name} -> {member.linkname}"
                    )
                seen.add(member.name)
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise SystemExit(f"could not read member: {member.name}")
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
            if size != item["bytes"] or digest.hexdigest() != item["sha256"]:
                raise SystemExit(
                    f"member mismatch: {member.name}; size={size}/{item['bytes']}; "
                    f"sha256={digest.hexdigest()}/{item['sha256']}"
                )
            seen.add(member.name)

    missing = sorted(set(expected) - seen)
    result = {
        "status": "verified",
        "archive": str(archive_path),
        "archive_sha256": verification["sha256"],
        "verified_payload_files": len(seen),
        "verified_payload_bytes": sum(expected[name]["bytes"] for name in seen),
        "missing": missing,
        "unexpected": unexpected,
        "outer_checks": outer_checks,
    }
    if missing or unexpected:
        raise SystemExit(json.dumps(result, ensure_ascii=False, indent=2))
    output = root / "FINAL_EXPERIMENT_RECORDS_INDEPENDENT_VERIFICATION.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
