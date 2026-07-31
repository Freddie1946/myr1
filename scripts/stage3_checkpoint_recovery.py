#!/usr/bin/env python3
"""Fail-closed checkpoint discovery and conservative Stage3 budget settlement."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage3_openrouter_judge import BudgetLedger


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_complete_checkpoint(path: Path, *, world_size: int = 8) -> dict[str, Any]:
    checkpoint = path.resolve()
    try:
        step = int(checkpoint.name.removeprefix("checkpoint-"))
    except ValueError as exc:
        raise ValueError(f"invalid checkpoint directory name: {checkpoint.name}") from exc
    if checkpoint.name != f"checkpoint-{step}" or step <= 0:
        raise ValueError(f"invalid checkpoint directory name: {checkpoint.name}")
    if not checkpoint.is_dir():
        raise ValueError(f"checkpoint is not a directory: {checkpoint}")

    required_files = (
        "trainer_state.json",
        "model.safetensors.index.json",
        "scheduler.pt",
        "latest",
        "training_args.bin",
    )
    missing = [name for name in required_files if not (checkpoint / name).is_file()]
    if missing:
        raise ValueError(f"checkpoint is missing required files: {missing}")
    empty = [name for name in required_files if (checkpoint / name).stat().st_size == 0]
    if empty:
        raise ValueError(f"checkpoint has empty required files: {empty}")

    trainer_state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if int(trainer_state.get("global_step", -1)) != step:
        raise ValueError("trainer_state global_step does not match checkpoint name")
    latest = (checkpoint / "latest").read_text(encoding="utf-8").strip()
    if latest != f"global_step{step}":
        raise ValueError(f"DeepSpeed latest pointer mismatch: {latest!r}")

    index = json.loads(
        (checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("model index has no weight_map")
    shard_names = sorted(set(weight_map.values()))
    if any(not isinstance(name, str) or not name for name in shard_names):
        raise ValueError("model index contains an invalid shard name")
    missing_shards = [
        name
        for name in shard_names
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0
    ]
    if missing_shards:
        raise ValueError(f"model shards are missing or empty: {missing_shards}")

    state_dir = checkpoint / f"global_step{step}"
    optimizer_files = sorted(state_dir.glob("*_optim_states.pt"))
    model_state_files = sorted(state_dir.glob("*_model_states.pt"))
    rng_files = sorted(checkpoint.glob("rng_state_*.pth"))
    expected_counts = {
        "optimizer_state_files": (len(optimizer_files), world_size),
        "model_state_files": (len(model_state_files), world_size),
        "rng_state_files": (len(rng_files), world_size),
    }
    bad_counts = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in expected_counts.items()
        if actual != expected
    }
    if bad_counts:
        raise ValueError(f"distributed checkpoint file counts differ: {bad_counts}")
    distributed_files = [*optimizer_files, *model_state_files, *rng_files]
    if any(path.stat().st_size == 0 for path in distributed_files):
        raise ValueError("distributed checkpoint contains an empty state file")

    return {
        "checkpoint": str(checkpoint),
        "global_step": step,
        "world_size": world_size,
        "model_shards": len(shard_names),
        "optimizer_state_files": len(optimizer_files),
        "model_state_files": len(model_state_files),
        "rng_state_files": len(rng_files),
        "status": "complete",
    }


def latest_complete_checkpoint(output_dir: Path, *, world_size: int = 8) -> dict[str, Any] | None:
    candidates: list[tuple[int, Path]] = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        if path.name == f"checkpoint-{step}" and step > 0:
            candidates.append((step, path))
    rejected: list[dict[str, Any]] = []
    for _, path in sorted(candidates, reverse=True):
        try:
            result = validate_complete_checkpoint(path, world_size=world_size)
            result["rejected_newer_candidates"] = rejected
            return result
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            rejected.append({"checkpoint": str(path.resolve()), "error": str(exc)})
    return None


def settle_unresolved(
    ledger_path: Path,
    *,
    limit_usd: float,
    reserve_usd: float,
    max_unique_requests: int,
    reason: str,
) -> dict[str, Any]:
    ledger = BudgetLedger(
        ledger_path,
        limit_usd=limit_usd,
        reserve_usd=reserve_usd,
        max_unique_requests=max_unique_requests,
    )
    before = ledger.snapshot()
    reservation_ids = sorted(before.get("reservations", {}))
    settled: list[dict[str, Any]] = []
    for request_id in reservation_ids:
        current = ledger.snapshot().get("reservations", {}).get(request_id)
        if current is None:
            continue
        amount = float(current["amount_usd"])
        ledger.commit(
            request_id,
            amount,
            {
                "status": "conservatively_settled_after_training_failure",
                "settlement_reason": reason,
                "prior_reservation_status": current.get("status"),
                "prior_error": current.get("error"),
            },
        )
        settled.append({"request_id": request_id, "cost_usd": amount})
    after = ledger.snapshot()
    if after.get("reservations"):
        raise RuntimeError("unresolved reservations remain after conservative settlement")
    return {
        "status": "settled",
        "ledger": str(ledger_path.resolve()),
        "settled_count": len(settled),
        "settled_usd": sum(row["cost_usd"] for row in settled),
        "settled": settled,
        "committed_spend_usd": after["committed_spend_usd"],
        "completed_unique_requests": after["completed_unique_requests"],
    }


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"timestamp": now_iso(), **event}, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    latest = subparsers.add_parser("latest")
    latest.add_argument("--output-dir", type=Path, required=True)
    latest.add_argument("--world-size", type=int, default=8)
    latest.add_argument("--path-only", action="store_true")

    validate = subparsers.add_parser("validate")
    validate.add_argument("checkpoint", type=Path)
    validate.add_argument("--world-size", type=int, default=8)

    settle = subparsers.add_parser("settle")
    settle.add_argument("--ledger", type=Path, required=True)
    settle.add_argument("--limit-usd", type=float, required=True)
    settle.add_argument("--reserve-usd", type=float, default=0.05)
    settle.add_argument("--max-unique-requests", type=int, default=12001)
    settle.add_argument("--reason", required=True)

    event = subparsers.add_parser("event")
    event.add_argument("--audit", type=Path, required=True)
    event.add_argument("--json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "latest":
        result = latest_complete_checkpoint(args.output_dir, world_size=args.world_size)
        if result is None:
            raise SystemExit(3)
        print(result["checkpoint"] if args.path_only else json.dumps(result, sort_keys=True))
    elif args.command == "validate":
        print(json.dumps(validate_complete_checkpoint(args.checkpoint, world_size=args.world_size), sort_keys=True))
    elif args.command == "settle":
        print(
            json.dumps(
                settle_unresolved(
                    args.ledger,
                    limit_usd=args.limit_usd,
                    reserve_usd=args.reserve_usd,
                    max_unique_requests=args.max_unique_requests,
                    reason=args.reason,
                ),
                sort_keys=True,
            )
        )
    else:
        value = json.loads(args.json)
        if not isinstance(value, dict):
            raise ValueError("event JSON must be an object")
        append_event(args.audit, value)


if __name__ == "__main__":
    main()
