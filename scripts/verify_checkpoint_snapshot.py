#!/usr/bin/env python3
"""Fail-closed integrity verification for a PathVLM model-only snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse JSON file {path}: {exc}") from exc


def verify_snapshot(
    snapshot: Path,
    *,
    expected_step: int | None = None,
    expected_epoch: float | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    snapshot = snapshot.resolve()
    manifest_path = snapshot / "snapshot_manifest.json"
    if not snapshot.is_dir() or not manifest_path.is_file():
        raise ValueError(f"snapshot or snapshot_manifest.json is missing: {snapshot}")

    manifest_sha256 = sha256_file(manifest_path)
    if expected_manifest_sha256 and manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "snapshot manifest SHA-256 mismatch: "
            f"expected {expected_manifest_sha256}, found {manifest_sha256}"
        )
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("snapshot manifest schema_version must be 1")
    if manifest.get("model_only") is not True or manifest.get("resumable") is not False:
        raise ValueError("snapshot must be model_only=true and resumable=false")

    step = manifest.get("global_step")
    epoch = manifest.get("epoch")
    if not isinstance(step, int) or step <= 0:
        raise ValueError(f"invalid global_step: {step!r}")
    if not isinstance(epoch, (int, float)) or float(epoch) <= 0:
        raise ValueError(f"invalid epoch: {epoch!r}")
    if expected_step is not None and step != expected_step:
        raise ValueError(f"global_step mismatch: expected {expected_step}, found {step}")
    if expected_epoch is not None and abs(float(epoch) - expected_epoch) > 1e-9:
        raise ValueError(f"epoch mismatch: expected {expected_epoch}, found {epoch}")

    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise ValueError("snapshot manifest files must be a nonempty list")
    declared: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"manifest files[{index}] is not an object")
        name = row.get("name")
        size = row.get("size_bytes")
        digest = row.get("sha256")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or Path(name).parts != (name,)
        ):
            raise ValueError(f"unsafe or nested manifest filename at row {index}: {name!r}")
        if name in declared:
            raise ValueError(f"duplicate manifest filename: {name}")
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid size for {name}: {size!r}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid SHA-256 for {name}: {digest!r}")
        declared[name] = row

    actual = {
        path.name
        for path in snapshot.iterdir()
        if path.is_file() and path.name != manifest_path.name
    }
    if actual != set(declared):
        raise ValueError(
            f"snapshot file set mismatch: missing={sorted(set(declared) - actual)}, "
            f"unexpected={sorted(actual - set(declared))}"
        )

    total_bytes = 0
    for name, row in declared.items():
        path = snapshot / name
        size = path.stat().st_size
        if size != row["size_bytes"]:
            raise ValueError(
                f"size mismatch for {name}: expected {row['size_bytes']}, found {size}"
            )
        digest = sha256_file(path)
        if digest != row["sha256"]:
            raise ValueError(
                f"SHA-256 mismatch for {name}: expected {row['sha256']}, found {digest}"
            )
        total_bytes += size

    required = {
        "config.json",
        "chat_template.json",
        "model.safetensors.index.json",
        "trainer_state.json",
    }
    missing_required = required - set(declared)
    if missing_required:
        raise ValueError(f"required snapshot files are missing: {sorted(missing_required)}")

    index = load_json(snapshot / "model.safetensors.index.json")
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model index weight_map must be a nonempty object")
    shards = set(weight_map.values())
    if not all(isinstance(name, str) and name in declared for name in shards):
        raise ValueError("model index references a shard absent from the snapshot manifest")

    trainer_state = load_json(snapshot / "trainer_state.json")
    if trainer_state.get("global_step") != step:
        raise ValueError(
            "trainer_state global_step does not match snapshot manifest: "
            f"{trainer_state.get('global_step')} != {step}"
        )
    trainer_epoch = trainer_state.get("epoch")
    if not isinstance(trainer_epoch, (int, float)) or abs(float(trainer_epoch) - float(epoch)) > 1e-9:
        raise ValueError(
            "trainer_state epoch does not match snapshot manifest: "
            f"{trainer_epoch!r} != {epoch!r}"
        )

    return {
        "status": "passed",
        "snapshot": str(snapshot),
        "global_step": step,
        "epoch": float(epoch),
        "manifest_sha256": manifest_sha256,
        "verified_file_count": len(declared),
        "verified_total_bytes": total_bytes,
        "model_shard_count": len(shards),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--expected-epoch", type=float)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = verify_snapshot(
            args.snapshot,
            expected_step=args.expected_step,
            expected_epoch=args.expected_epoch,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"snapshot verification failed: {exc}") from exc
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()

