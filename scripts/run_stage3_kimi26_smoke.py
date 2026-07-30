#!/usr/bin/env python3
"""One paid, synthetic, non-training compatibility smoke for the Kimi 2.6 Judge."""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import os
import struct
import zlib
from pathlib import Path

from stage3_openrouter_judge import (
    OpenRouterJudge,
    atomic_json,
    make_cache_key,
    normalize_provider,
)


MODEL = "moonshotai/kimi-k2.6"
PROVIDER = "inceptron"


def png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = binascii.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)


def synthetic_png() -> bytes:
    width = height = 64
    scanlines = b"".join(
        b"\x00" + b"\x1e\x5a\xdc" * 32 + b"\xdc\x3c\x32" * 32
        for _ in range(height)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
        )
        + png_chunk(b"IDAT", zlib.compress(scanlines, 9))
        + png_chunk(b"IEND", b"")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/"
            "runs/stage3_process_grpo/kimi26_50step_seed42_20260730"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    marker = run_dir / "kimi26_smoke_v2_passed.json"
    if marker.exists():
        raise RuntimeError(f"smoke marker already exists: {marker}")
    smoke_dir = run_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    image = smoke_dir / "synthetic_blue_red.png"
    expected_image = synthetic_png()
    if image.exists():
        if image.read_bytes() != expected_image:
            raise RuntimeError(f"existing synthetic smoke image differs: {image}")
    else:
        image.write_bytes(expected_image)
    digest = hashlib.sha256(image.read_bytes()).hexdigest()

    problem = (
        "Synthetic API control only: which description matches the image?\n"
        "Options:\nA) blue left, red right\nB) red left, blue right"
    )
    solution = "<answer>A</answer>"
    completion = (
        "<think>The left half is blue and the right half is red, so option B is "
        "eliminated and option A matches the visible features. This synthetic control "
        "does not require a medical criterion.</think><answer>A</answer>"
    )
    cache_key = make_cache_key(
        image_sha256=digest,
        problem=problem,
        solution=solution,
        completion=completion,
        model_id=MODEL,
        provider_slugs=(PROVIDER,),
        max_token_parameter="max_tokens",
        max_judge_tokens=640,
        zdr_required=True,
        data_collection="deny",
        reasoning_enabled=False,
    )
    cache_path = run_dir / "judge" / "cache" / cache_key[:2] / f"{cache_key}.json"

    os.environ["PATHVLM_TRAINING_SEGMENT"] = "kimi26_nontraining_api_smoke_v2"
    judge = OpenRouterJudge(
        run_dir / "judge",
        model_id=MODEL,
        provider_slugs=(PROVIDER,),
        max_token_parameter="max_tokens",
        max_judge_tokens=640,
        zdr_required=True,
        data_collection="deny",
        reasoning_enabled=False,
        limit_usd=15,
        reserve_usd=0.05,
        max_unique_requests=411,
        retry_delays=(2.0, 5.0),
    )
    reward = judge.judge_one(
        image_path=str(image),
        image_sha256=digest,
        problem=problem,
        solution=solution,
        completion=completion,
        source_metadata={
            "record_index": -1,
            "smoke": True,
            "non_training": True,
        },
    )

    if not cache_path.is_file():
        raise RuntimeError(f"expected smoke cache record is missing: {cache_path}")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    served_model = str(cache.get("served_model") or "")
    served_provider = str(cache.get("served_provider") or "")
    if served_model != MODEL or normalize_provider(served_provider) != PROVIDER:
        raise RuntimeError(
            f"served route mismatch: model={served_model!r}, provider={served_provider!r}"
        )
    ledger = judge.ledger.snapshot()
    reservations = ledger.get("reservations", {})
    if ledger.get("completed_unique_requests", 0) < 1 or reservations:
        raise RuntimeError("smoke budget ledger did not settle the successful request")
    payload = {
        "schema_version": 1,
        "status": "passed",
        "model": MODEL,
        "provider": PROVIDER,
        "served_provider": served_provider,
        "zdr": True,
        "data_collection": "deny",
        "distillable_enforced": True,
        "non_training": True,
        "synthetic_image_sha256": digest,
        "generation_id": cache.get("generation_id"),
        "cost_usd": cache.get("cost_usd"),
        "process_reward": reward,
        "max_judge_tokens": 640,
        "evidence_max_characters": 160,
        "reasoning_enabled": False,
        "unresolved_reservations": len(reservations),
        "budget_ledger": str(run_dir / "judge" / "budget_ledger.json"),
        "cache_record": str(cache_path),
    }
    atomic_json(marker, payload)
    print(
        json.dumps(
            {
                "status": "passed",
                "model": MODEL,
                "provider": served_provider,
                "generation_id": payload["generation_id"],
                "cost_usd": payload["cost_usd"],
                "process_reward": reward,
                "marker": str(marker),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
