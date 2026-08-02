#!/usr/bin/env python3
"""Verify the fixed local ModelScope Mllama snapshot structure without loading weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


FROZEN_MODELS = {
    "11b": {"shards": 5, "weight_bytes": 21_340_560_870},
    "90b": {"shards": 37, "weight_bytes": 177_186_907_758},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root: Path, size: str) -> dict[str, Any]:
    root = root.resolve()
    frozen = FROZEN_MODELS[size]
    required = [
        "LICENSE.txt",
        "USE_POLICY.md",
        "chat_template.json",
        "config.json",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError(f"required files are missing: {missing}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("architectures") != ["MllamaForConditionalGeneration"]:
        raise ValueError(f"unexpected architecture: {config.get('architectures')!r}")
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model index has no weight_map")
    shards = sorted(set(weight_map.values()))
    if len(shards) != frozen["shards"]:
        raise ValueError(
            f"shard-count mismatch: expected {frozen['shards']}, found {len(shards)}"
        )
    missing_shards = [name for name in shards if not (root / name).is_file()]
    if missing_shards:
        raise ValueError(f"weight shards are missing: {missing_shards}")
    weight_bytes = sum((root / name).stat().st_size for name in shards)
    if weight_bytes != frozen["weight_bytes"]:
        raise ValueError(
            f"weight-byte mismatch: expected {frozen['weight_bytes']}, found {weight_bytes}"
        )
    return {
        "status": "passed",
        "model_size": size,
        "snapshot": str(root),
        "architecture": "MllamaForConditionalGeneration",
        "shard_count": len(shards),
        "weight_bytes": weight_bytes,
        "config_sha256": sha256_file(root / "config.json"),
        "model_index_sha256": sha256_file(index_path),
        "chat_template_sha256": sha256_file(root / "chat_template.json"),
        "license_sha256": sha256_file(root / "LICENSE.txt"),
        "use_policy_sha256": sha256_file(root / "USE_POLICY.md"),
        "full_weight_sha256_not_computed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--size", choices=sorted(FROZEN_MODELS), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.snapshot, args.size)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Mllama snapshot verification failed: {exc}") from exc
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()

