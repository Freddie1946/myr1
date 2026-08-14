#!/usr/bin/env python3
"""Merge paid external-ROI shards, preferring validated repair rows.

The merger verifies panel coverage, image identity, and exact served-model
identity.  It never silently retains a failed request when a validated repair
exists and refuses conflicting duplicate validated rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_rows(paths: list[Path]) -> tuple[dict[int, dict[str, Any]], dict[str, str]]:
    candidates: dict[int, list[dict[str, Any]]] = {}
    hashes = {}
    for path in paths:
        hashes[str(path.resolve())] = sha256_file(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            candidates.setdefault(int(row["panel_index"]), []).append(row)
    selected = {}
    for index, rows in candidates.items():
        valid = [row for row in rows if row.get("status") == "validated"]
        if len(valid) > 1:
            signatures = {
                json.dumps(row.get("annotation"), sort_keys=True, ensure_ascii=False)
                for row in valid
            }
            if len(signatures) > 1:
                raise ValueError(f"conflicting validated annotations for panel {index}")
        selected[index] = valid[-1] if valid else rows[-1]
    return selected, hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--allow-missing-panel-index", type=int, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise FileExistsError("refusing to overwrite merged paid outputs")
    panel_raw = args.panel.read_bytes()
    panel = json.loads(panel_raw)
    cases = {int(case["panel_index"]): case for case in panel["cases"]}
    rows, source_hashes = choose_rows(args.input)
    allowed_missing = set(args.allow_missing_panel_index)
    if not allowed_missing <= set(cases):
        raise ValueError("allowed-missing index is absent from panel")
    for index in allowed_missing:
        if index in rows and rows[index].get("status") != "validated":
            del rows[index]
    if set(rows) != set(cases) - allowed_missing:
        missing = sorted(set(cases) - set(rows)); extra = sorted(set(rows) - set(cases))
        raise ValueError(f"coverage mismatch missing={missing} extra={extra}")
    for index, row in rows.items():
        if row.get("status") != "validated":
            raise ValueError(f"panel {index} has no validated annotation")
        if row.get("served_model") != args.expected_model:
            raise ValueError(f"panel {index} served by {row.get('served_model')!r}")
        if row.get("image_sha256") != cases[index]["image_sha256"]:
            raise ValueError(f"panel {index} image identity mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index in sorted(rows):
            handle.write(json.dumps(rows[index], ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "expected_model": args.expected_model,
        "case_count": len(rows),
        "allowed_missing_panel_indices": sorted(allowed_missing),
        "actual_missing_panel_indices": sorted(set(cases) - set(rows)),
        "panel": str(args.panel.resolve()),
        "panel_sha256": hashlib.sha256(panel_raw).hexdigest(),
        "input_sha256": source_hashes,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "case_count": len(rows), "output_sha256": manifest["output_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
