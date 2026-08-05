#!/usr/bin/env python3
"""Fail-closed checkpoint discovery and conservative Stage3 budget settlement."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage3_openrouter_judge import BudgetLedger, atomic_json


LOG_TAIL_BYTES = 8 * 1024 * 1024

# Recovery is intentionally a whitelist.  Distributed launchers commonly
# collapse every worker exception into exit status 1, so an exit code alone
# cannot distinguish a transient Judge outage from a scientific/configuration
# failure.  Unknown failures stop for human diagnosis instead of being retried.
TERMINAL_FAILURE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "user_interrupt",
        ("KeyboardInterrupt", "got signal: 2", "SIGINT", "signal 2"),
    ),
    (
        "budget_or_request_gate",
        (
            "BudgetError",
            "hard cap refuses",
            "maximum unique OpenRouter request count reached",
            "budget breached",
        ),
    ),
    (
        "judge_identity_or_schema_mismatch",
        (
            "served model mismatch",
            "served provider is not in frozen provider set",
            "Judge output keys differ",
            "Judge evidence keys differ",
            "Judge event is not boolean",
        ),
    ),
    (
        "source_or_contract_drift",
        (
            "image SHA-256 mismatch",
            "source audit",
            "contract mismatch",
            "adapter count mismatch",
        ),
    ),
    (
        "resource_or_numeric_failure",
        (
            "CUDA out of memory",
            "OutOfMemoryError",
            "No space left on device",
            "NaN",
            "nan loss",
            "Inf detected",
        ),
    ),
    (
        "checkpoint_deserialization_compatibility",
        ("Weights only load failed",),
    ),
    (
        "total_rule_fallback_limit",
        ("rule fallback total limit reached",),
    ),
)

RECOVERABLE_FAILURE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "consecutive_judge_outage",
        ("rule fallback consecutive limit reached",),
    ),
    (
        "transient_judge_transport",
        (
            "http.client.IncompleteRead",
            "OpenRouter HTTP 520",
            "OpenRouter transport failure",
            "JudgeUnavailableForRuleFallback",
        ),
    ),
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_log_tail(path: Path, *, maximum_bytes: int = LOG_TAIL_BYTES) -> bytes:
    if maximum_bytes <= 0:
        raise ValueError("maximum log-tail bytes must be positive")
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - maximum_bytes), os.SEEK_SET)
        return handle.read()


def classify_training_failure(log_path: Path) -> dict[str, Any]:
    """Classify a failed distributed launch under a fail-closed recovery policy."""

    path = log_path.resolve()
    if not path.is_file():
        return {
            "action": "stop",
            "category": "missing_training_log",
            "matched_pattern": None,
            "log": str(path),
        }
    raw = _read_log_tail(path)
    text = raw.decode("utf-8", errors="replace")
    for category, patterns in TERMINAL_FAILURE_PATTERNS:
        for pattern in patterns:
            if pattern in text:
                return {
                    "action": "stop",
                    "category": category,
                    "matched_pattern": pattern,
                    "log": str(path),
                    "log_bytes": path.stat().st_size,
                    "tail_bytes": len(raw),
                    "tail_sha256": hashlib.sha256(raw).hexdigest(),
                }
    for category, patterns in RECOVERABLE_FAILURE_PATTERNS:
        for pattern in patterns:
            if pattern in text:
                return {
                    "action": "recover",
                    "category": category,
                    "matched_pattern": pattern,
                    "log": str(path),
                    "log_bytes": path.stat().st_size,
                    "tail_bytes": len(raw),
                    "tail_sha256": hashlib.sha256(raw).hexdigest(),
                }
    return {
        "action": "stop",
        "category": "unknown_failure_fail_closed",
        "matched_pattern": None,
        "log": str(path),
        "log_bytes": path.stat().st_size,
        "tail_bytes": len(raw),
        "tail_sha256": hashlib.sha256(raw).hexdigest(),
    }


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


def raise_request_cap(
    ledger_path: Path,
    *,
    limit_usd: float,
    reserve_usd: float,
    old_max_unique_requests: int,
    new_max_unique_requests: int,
    reason: str,
) -> dict[str, Any]:
    """Atomically raise a settled ledger's request cap with an audit amendment."""

    if new_max_unique_requests <= old_max_unique_requests:
        raise ValueError("new request cap must be greater than the old request cap")
    if not reason.strip():
        raise ValueError("request-cap amendment reason must not be empty")
    path = ledger_path.resolve()
    if not path.is_file():
        raise ValueError(f"budget ledger does not exist: {path}")
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        value = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "limit_usd": float(limit_usd),
            "reserve_usd": float(reserve_usd),
            "max_unique_requests": int(old_max_unique_requests),
        }
        mismatch = {
            key: {"expected": expected_value, "actual": value.get(key)}
            for key, expected_value in expected.items()
            if value.get(key) != expected_value
        }
        if mismatch:
            raise ValueError(f"budget ledger pre-amendment contract mismatch: {mismatch}")
        if value.get("reservations"):
            raise ValueError("cannot amend request cap while reservations are unresolved")
        completed = int(value.get("completed_unique_requests", -1))
        if completed < 0 or completed > old_max_unique_requests:
            raise ValueError("budget ledger completed-request count is invalid")
        if new_max_unique_requests < completed:
            raise ValueError("new request cap is below the completed-request count")
        amendment = {
            "timestamp": now_iso(),
            "field": "max_unique_requests",
            "old_value": int(old_max_unique_requests),
            "new_value": int(new_max_unique_requests),
            "reason": reason.strip(),
        }
        value.setdefault("contract_amendments", []).append(amendment)
        value["max_unique_requests"] = int(new_max_unique_requests)
        value["updated_at"] = now_iso()
        atomic_json(path, value)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return {
        "status": "request_cap_raised",
        "ledger": str(path),
        "completed_unique_requests": completed,
        "old_max_unique_requests": int(old_max_unique_requests),
        "new_max_unique_requests": int(new_max_unique_requests),
        "amendment": amendment,
    }


def raise_budget_and_request_caps(
    ledger_path: Path,
    *,
    old_limit_usd: float,
    new_limit_usd: float,
    reserve_usd: float,
    old_max_unique_requests: int,
    new_max_unique_requests: int,
    reason: str,
) -> dict[str, Any]:
    """Atomically raise both settled budget-ledger caps with audit amendments."""

    if new_limit_usd <= old_limit_usd:
        raise ValueError("new budget cap must be greater than the old budget cap")
    if new_max_unique_requests <= old_max_unique_requests:
        raise ValueError("new request cap must be greater than the old request cap")
    if not reason.strip():
        raise ValueError("contract-cap amendment reason must not be empty")
    path = ledger_path.resolve()
    if not path.is_file():
        raise ValueError(f"budget ledger does not exist: {path}")
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        value = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "limit_usd": float(old_limit_usd),
            "reserve_usd": float(reserve_usd),
            "max_unique_requests": int(old_max_unique_requests),
        }
        mismatch = {
            key: {"expected": expected_value, "actual": value.get(key)}
            for key, expected_value in expected.items()
            if value.get(key) != expected_value
        }
        if mismatch:
            raise ValueError(f"budget ledger pre-amendment contract mismatch: {mismatch}")
        if value.get("reservations"):
            raise ValueError("cannot amend contract caps while reservations are unresolved")
        completed = int(value.get("completed_unique_requests", -1))
        committed = float(value.get("committed_spend_usd", -1))
        if completed < 0 or completed > old_max_unique_requests:
            raise ValueError("budget ledger completed-request count is invalid")
        if committed < 0 or committed > old_limit_usd + 1e-12:
            raise ValueError("budget ledger committed spend is invalid")
        if new_max_unique_requests < completed:
            raise ValueError("new request cap is below the completed-request count")
        if new_limit_usd < committed:
            raise ValueError("new budget cap is below committed spend")
        timestamp = now_iso()
        amendments = [
            {
                "timestamp": timestamp,
                "field": "limit_usd",
                "old_value": float(old_limit_usd),
                "new_value": float(new_limit_usd),
                "reason": reason.strip(),
            },
            {
                "timestamp": timestamp,
                "field": "max_unique_requests",
                "old_value": int(old_max_unique_requests),
                "new_value": int(new_max_unique_requests),
                "reason": reason.strip(),
            },
        ]
        value.setdefault("contract_amendments", []).extend(amendments)
        value["limit_usd"] = float(new_limit_usd)
        value["max_unique_requests"] = int(new_max_unique_requests)
        value["updated_at"] = timestamp
        atomic_json(path, value)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return {
        "status": "budget_and_request_caps_raised",
        "ledger": str(path),
        "committed_spend_usd": committed,
        "completed_unique_requests": completed,
        "old_limit_usd": float(old_limit_usd),
        "new_limit_usd": float(new_limit_usd),
        "old_max_unique_requests": int(old_max_unique_requests),
        "new_max_unique_requests": int(new_max_unique_requests),
        "amendments": amendments,
    }


def raise_fallback_total_cap(
    ledger_path: Path,
    *,
    old_total_limit: int,
    new_total_limit: int,
    consecutive_limit: int,
    reason: str,
) -> dict[str, Any]:
    """Atomically raise the fallback total while preserving its consecutive gate."""

    if new_total_limit <= old_total_limit:
        raise ValueError("new fallback total cap must be greater than the old cap")
    if consecutive_limit <= 0 or consecutive_limit > old_total_limit:
        raise ValueError("invalid fallback consecutive limit")
    if not reason.strip():
        raise ValueError("fallback-cap amendment reason must not be empty")
    path = ledger_path.resolve()
    if not path.is_file():
        raise ValueError(f"fallback ledger does not exist: {path}")
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        value = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "total_limit": int(old_total_limit),
            "consecutive_limit": int(consecutive_limit),
        }
        mismatch = {
            key: {"expected": expected_value, "actual": value.get(key)}
            for key, expected_value in expected.items()
            if value.get(key) != expected_value
        }
        if mismatch:
            raise ValueError(f"fallback ledger pre-amendment contract mismatch: {mismatch}")
        total_used = int(value.get("total_used", -1))
        consecutive_used = int(value.get("consecutive_used", -1))
        if total_used < 0 or total_used > old_total_limit:
            raise ValueError("fallback total-used count is invalid")
        if consecutive_used < 0 or consecutive_used > consecutive_limit:
            raise ValueError("fallback consecutive-used count is invalid")
        timestamp = now_iso()
        amendment = {
            "timestamp": timestamp,
            "field": "total_limit",
            "old_value": int(old_total_limit),
            "new_value": int(new_total_limit),
            "reason": reason.strip(),
        }
        value.setdefault("contract_amendments", []).append(amendment)
        value["total_limit"] = int(new_total_limit)
        value["updated_at"] = timestamp
        atomic_json(path, value)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return {
        "status": "fallback_total_cap_raised",
        "ledger": str(path),
        "total_used": total_used,
        "consecutive_used": consecutive_used,
        "old_total_limit": int(old_total_limit),
        "new_total_limit": int(new_total_limit),
        "consecutive_limit": int(consecutive_limit),
        "amendment": amendment,
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

    raise_cap = subparsers.add_parser("raise-request-cap")
    raise_cap.add_argument("--ledger", type=Path, required=True)
    raise_cap.add_argument("--limit-usd", type=float, required=True)
    raise_cap.add_argument("--reserve-usd", type=float, default=0.05)
    raise_cap.add_argument("--old-max-unique-requests", type=int, required=True)
    raise_cap.add_argument("--new-max-unique-requests", type=int, required=True)
    raise_cap.add_argument("--reason", required=True)

    raise_contract = subparsers.add_parser("raise-contract-caps")
    raise_contract.add_argument("--ledger", type=Path, required=True)
    raise_contract.add_argument("--old-limit-usd", type=float, required=True)
    raise_contract.add_argument("--new-limit-usd", type=float, required=True)
    raise_contract.add_argument("--reserve-usd", type=float, default=0.05)
    raise_contract.add_argument("--old-max-unique-requests", type=int, required=True)
    raise_contract.add_argument("--new-max-unique-requests", type=int, required=True)
    raise_contract.add_argument("--reason", required=True)

    raise_fallback = subparsers.add_parser("raise-fallback-total-cap")
    raise_fallback.add_argument("--ledger", type=Path, required=True)
    raise_fallback.add_argument("--old-total-limit", type=int, required=True)
    raise_fallback.add_argument("--new-total-limit", type=int, required=True)
    raise_fallback.add_argument("--consecutive-limit", type=int, required=True)
    raise_fallback.add_argument("--reason", required=True)

    classify = subparsers.add_parser("classify")
    classify.add_argument("--log", type=Path, required=True)
    classify.add_argument("--action-only", action="store_true")

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
    elif args.command == "raise-request-cap":
        print(
            json.dumps(
                raise_request_cap(
                    args.ledger,
                    limit_usd=args.limit_usd,
                    reserve_usd=args.reserve_usd,
                    old_max_unique_requests=args.old_max_unique_requests,
                    new_max_unique_requests=args.new_max_unique_requests,
                    reason=args.reason,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "raise-contract-caps":
        print(
            json.dumps(
                raise_budget_and_request_caps(
                    args.ledger,
                    old_limit_usd=args.old_limit_usd,
                    new_limit_usd=args.new_limit_usd,
                    reserve_usd=args.reserve_usd,
                    old_max_unique_requests=args.old_max_unique_requests,
                    new_max_unique_requests=args.new_max_unique_requests,
                    reason=args.reason,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "raise-fallback-total-cap":
        print(
            json.dumps(
                raise_fallback_total_cap(
                    args.ledger,
                    old_total_limit=args.old_total_limit,
                    new_total_limit=args.new_total_limit,
                    consecutive_limit=args.consecutive_limit,
                    reason=args.reason,
                ),
                sort_keys=True,
            )
        )
    elif args.command == "classify":
        result = classify_training_failure(args.log)
        print(result["action"] if args.action_only else json.dumps(result, sort_keys=True))
    else:
        value = json.loads(args.json)
        if not isinstance(value, dict):
            raise ValueError("event JSON must be an object")
        append_event(args.audit, value)


if __name__ == "__main__":
    main()
