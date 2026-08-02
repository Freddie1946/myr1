#!/usr/bin/env python3
"""One paid, synthetic contract smoke for the full Kimi 2.6 Stage3 arm."""

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
MAX_JUDGE_TOKENS = 1024
EVIDENCE_TARGET_CHARACTERS = 160
EVIDENCE_MAX_CHARACTERS = 512
MAX_UNIQUE_REQUESTS = 12361


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
            "runs/stage3_process_grpo/kimi26_full3epoch_seed42_20260730"
        ),
    )
    parser.add_argument("--budget-limit-usd", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.budget_limit_usd <= 0:
        raise ValueError("--budget-limit-usd must be positive")
    run_dir = args.run_dir.resolve()
    marker = run_dir / "kimi26_formal_contract_smoke_passed.json"
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
        max_judge_tokens=MAX_JUDGE_TOKENS,
        zdr_required=True,
        data_collection="deny",
        reasoning_enabled=False,
    )
    cache_path = run_dir / "judge" / "cache" / cache_key[:2] / f"{cache_key}.json"

    os.environ["PATHVLM_TRAINING_SEGMENT"] = (
        "kimi26_full3epoch_nontraining_contract_smoke"
    )
    judge = OpenRouterJudge(
        run_dir / "judge",
        model_id=MODEL,
        provider_slugs=(PROVIDER,),
        max_token_parameter="max_tokens",
        max_judge_tokens=MAX_JUDGE_TOKENS,
        zdr_required=True,
        data_collection="deny",
        reasoning_enabled=False,
        limit_usd=args.budget_limit_usd,
        reserve_usd=0.05,
        max_unique_requests=MAX_UNIQUE_REQUESTS,
        retry_delays=(15.0, 45.0, 90.0),
        minimum_request_interval_seconds=4.0,
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
            "formal_contract": True,
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
    physical_attempts = int(ledger.get("completed_unique_requests", 0))
    if not 1 <= physical_attempts <= 4 or reservations:
        raise RuntimeError(
            "formal smoke ledger must settle one through four physical attempts"
        )

    payload = {
        "schema_version": 1,
        "status": "passed",
        "run_class": "complete_contemporary_kimi_stage3_arm",
        "formal_result": False,
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
        "max_judge_tokens": MAX_JUDGE_TOKENS,
        "evidence_target_characters": EVIDENCE_TARGET_CHARACTERS,
        "evidence_max_characters": EVIDENCE_MAX_CHARACTERS,
        "reasoning_enabled": False,
        "minimum_request_interval_seconds": 4,
        "retry_delays_seconds": [15, 45, 90],
        "cache_reuse_across_training_segments": False,
        "rule_fallback_enabled": True,
        "rule_fallback_total_limit": 24,
        "rule_fallback_consecutive_limit": 4,
        "rule_fallback_max_reward": 0.5,
        "budget_limit_usd": args.budget_limit_usd,
        "max_unique_requests": MAX_UNIQUE_REQUESTS,
        "maximum_physical_attempts_per_logical_judgment": 4,
        "smoke_physical_attempts": physical_attempts,
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
