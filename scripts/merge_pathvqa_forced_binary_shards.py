#!/usr/bin/env python3
"""Merge complete, disjoint PathVQA forced-binary evaluation shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    shard_dirs = sorted(path.parent for path in args.shard_root.glob("shard_*/metrics.json"))
    if not shard_dirs:
        raise FileNotFoundError(f"no completed shards under {args.shard_root}")
    metrics = [load_json(path / "metrics.json") for path in shard_dirs]
    shard_count = int(metrics[0]["shard_count"])
    if len(metrics) != shard_count:
        raise ValueError(f"expected {shard_count} shards, found {len(metrics)}")

    invariant_keys = (
        "data_sha256", "source_count", "shard_count", "model_config_sha256",
        "adapter_config_sha256", "adapter_model_sha256", "image_mode",
        "result_role", "decision_rule", "prompt_template",
    )
    expected = {key: metrics[0].get(key) for key in invariant_keys}
    seen_shards: set[int] = set()
    rows: list[dict[str, Any]] = []
    shard_provenance = []
    for shard_dir, metric in zip(shard_dirs, metrics):
        if metric.get("status") != "completed":
            raise ValueError(f"incomplete shard: {shard_dir}")
        if any(metric.get(key) != value for key, value in expected.items()):
            raise ValueError(f"incompatible shard metrics: {shard_dir}")
        shard_index = int(metric["shard_index"])
        if shard_index in seen_shards:
            raise ValueError(f"duplicate shard index {shard_index}")
        seen_shards.add(shard_index)
        predictions = shard_dir / "predictions.jsonl"
        shard_rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
        if len(shard_rows) != int(metric["selected_count"]):
            raise ValueError(f"row count mismatch: {shard_dir}")
        rows.extend(shard_rows)
        shard_provenance.append({
            "shard_index": shard_index,
            "metrics": str((shard_dir / "metrics.json").resolve()),
            "metrics_sha256": sha256_file(shard_dir / "metrics.json"),
            "predictions_sha256": sha256_file(predictions),
        })
    if seen_shards != set(range(shard_count)):
        raise ValueError(f"missing shard indices: {set(range(shard_count)) - seen_shards}")

    rows.sort(key=lambda row: int(row["index"]))
    expected_indices = list(range(int(expected["source_count"])))
    indices = [int(row["index"]) for row in rows]
    if indices != expected_indices:
        raise ValueError("merged rows are not an exact, unique cover of source indices")

    args.output_dir.mkdir(parents=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    correct = sum(bool(row["forced_binary_correct"]) for row in rows)
    merged = dict(metrics[0])
    merged.update({
        "status": "completed_merged",
        "shard_index": None,
        "selected_count": len(rows),
        "count": len(rows),
        "coverage": 1.0,
        "correct": correct,
        "accuracy": correct / len(rows),
        "mean_absolute_margin": sum(abs(float(row["yes_minus_no_margin"])) for row in rows) / len(rows),
        "predictions": str(predictions_path.resolve()),
        "predictions_sha256": sha256_file(predictions_path),
        "shards": sorted(shard_provenance, key=lambda value: value["shard_index"]),
    })
    (args.output_dir / "metrics.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
