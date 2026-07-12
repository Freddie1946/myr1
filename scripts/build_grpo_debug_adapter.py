#!/usr/bin/env python3
"""Build deterministic PathMMU GRPO debug adapters from a frozen split."""

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
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--yaml-json-path", help="Runtime-visible JSON path to write into YAML")
    parser.add_argument("--prefix", default="pathvlm_rl_smoke")
    args = parser.parse_args()

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if args.offset < 0 or args.offset + args.count > len(records):
        raise ValueError(
            f"requested offset={args.offset}, count={args.count} from {len(records)} records"
        )

    selected = records[args.offset : args.offset + args.count]
    required = {"image", "problem", "solution"}
    for index, record in enumerate(selected):
        missing = required.difference(record)
        if missing:
            raise ValueError(f"record {index} missing fields: {sorted(missing)}")
        if not str(record["image"]).strip() or not str(record["problem"]).strip() or not str(record["solution"]).strip():
            raise ValueError(f"record {index} contains an empty required field")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.output_dir / f"{args.prefix}_n{args.count:04d}.json"
    yaml_path = args.output_dir / f"{args.prefix}_n{args.count:04d}.yaml"
    manifest_path = args.output_dir / "adapter_manifest.json"

    data_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    runtime_json_path = args.yaml_json_path or data_path.as_posix()
    yaml_path.write_text(
        "datasets:\n"
        f"  - json_path: {runtime_json_path}\n"
        "    sampling_strategy: all\n",
        encoding="utf-8",
    )
    manifest = {
        "source": str(args.input),
        "source_sha256": sha256(args.input),
        "selection": f"slice:{args.offset}:{args.offset + args.count}",
        "count": len(selected),
        "data_file": str(data_path),
        "data_sha256": sha256(data_path),
        "yaml_file": str(yaml_path),
        "runtime_json_path": runtime_json_path,
        "test_accessed": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
