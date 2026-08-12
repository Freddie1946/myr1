#!/usr/bin/env python3
"""Verify online GRPO rewards against the frozen source rows and offline parser."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from grpo_pathmmu_audit import STRICT_PROMPT_SUFFIX, strict_prompt_text
from pathmmu_rewards import accuracy_reward, format_reward


def load_events(root: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in sorted(root.glob("rank_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            events.extend(json.loads(line) for line in handle if line.strip())
    return events


def prompt_text(event: dict[str, Any]) -> str:
    try:
        return str(event["prompt"][0]["content"][1]["text"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("malformed prompt in reward event") from exc


def prompt_problem(event: dict[str, Any]) -> str:
    text = prompt_text(event)
    if not text.endswith(STRICT_PROMPT_SUFFIX):
        raise RuntimeError("reward-event prompt does not use the strict PathMMU suffix")
    return text[: -len(STRICT_PROMPT_SUFFIX)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-dir", required=True, type=Path)
    parser.add_argument("--dataset-json", required=True, type=Path)
    parser.add_argument("--expected-steps", required=True, type=int)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--per-device-batch", required=True, type=int)
    parser.add_argument("--num-generations", required=True, type=int)
    parser.add_argument(
        "--reward-types",
        nargs="+",
        choices=("accuracy", "format", "format_scaled"),
        default=("accuracy", "format"),
    )
    parser.add_argument("--format-reward-weight", type=float, default=0.1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    records = json.loads(args.dataset_json.read_text(encoding="utf-8"))
    by_problem: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, record in enumerate(records):
        problem = str(record["problem"])
        if problem in by_problem:
            raise RuntimeError(f"dataset contains a duplicate problem at index {index}")
        by_problem[problem] = (index, record)

    events = load_events(args.events_dir)
    expected_per_type = (
        args.expected_steps * args.world_size * args.per_device_batch
    )
    errors: Counter[str] = Counter()
    keyed: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    accuracy_by_call: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)

    for event in events:
        reward_type = str(event.get("reward_type"))
        if reward_type not in set(args.reward_types):
            continue
        key = (
            int(event["rank"]),
            int(event["call_index"]),
            int(event["item_index"]),
            reward_type,
        )
        if key in keyed:
            errors["duplicate_event_key"] += 1
        keyed[key] = event

        text = prompt_text(event)
        actual_problem = prompt_problem(event)
        problem = str(event.get("problem", ""))
        source = by_problem.get(actual_problem)
        if source is None:
            errors["unknown_prompt_problem"] += 1
            continue
        source_index, record = source
        if problem != actual_problem or text != strict_prompt_text(actual_problem):
            errors["prompt_problem_mismatch"] += 1
        if int(event.get("record_index", -1)) != source_index:
            errors["record_index_mismatch"] += 1
        if str(event.get("solution", "")) != str(record["solution"]):
            errors["solution_mismatch"] += 1

        completion = event.get("completion", "")
        if reward_type == "accuracy":
            offline = accuracy_reward([completion], [record["solution"]])[0]
            accuracy_by_call[int(event["call_index"])].append(event)
        elif reward_type == "format":
            offline = format_reward([completion])[0]
        else:
            offline = args.format_reward_weight * format_reward([completion])[0]
        if float(event["reward"]) != float(offline):
            errors[f"{reward_type}_offline_reward_mismatch"] += 1

    counts = Counter(event.get("reward_type") for event in events)
    for reward_type in args.reward_types:
        if counts[reward_type] != expected_per_type:
            errors[f"{reward_type}_event_count"] += abs(
                counts[reward_type] - expected_per_type
            )

    for rank in range(args.world_size):
        for call_index in range(args.expected_steps):
            for reward_type in args.reward_types:
                local_count = sum(
                    key[0] == rank and key[1] == call_index and key[3] == reward_type
                    for key in keyed
                )
                if local_count != args.per_device_batch:
                    errors[f"{reward_type}_local_batch_count"] += 1

    group_count = 0
    for call_index in range(args.expected_steps):
        rows = sorted(
            accuracy_by_call[call_index],
            key=lambda event: (int(event["rank"]), int(event["item_index"])),
        )
        expected_global = args.world_size * args.per_device_batch
        if len(rows) != expected_global:
            errors["accuracy_global_batch_count"] += 1
            continue
        if expected_global % args.num_generations:
            errors["global_batch_not_divisible_by_generations"] += 1
            continue
        for start in range(0, expected_global, args.num_generations):
            group = rows[start : start + args.num_generations]
            group_count += 1
            if len({prompt_text(event) for event in group}) != 1:
                errors["generation_group_prompt_mismatch"] += 1
            if len({str(event.get("solution")) for event in group}) != 1:
                errors["generation_group_solution_mismatch"] += 1

    paired_format_type = next(
        (value for value in ("format", "format_scaled") if value in args.reward_types),
        None,
    )
    for key, accuracy_event in keyed.items():
        if key[3] != "accuracy":
            continue
        if paired_format_type is None:
            continue
        format_event = keyed.get((*key[:3], paired_format_type))
        if format_event is None:
            errors["missing_paired_format_event"] += 1
            continue
        for field in ("completion", "solution", "problem", "record_index", "prompt"):
            if accuracy_event.get(field) != format_event.get(field):
                errors[f"paired_event_{field}_mismatch"] += 1

    report = {
        "schema_version": 1,
        "passed": not errors,
        "events_dir": str(args.events_dir.resolve()),
        "dataset_json": str(args.dataset_json.resolve()),
        "expected_steps": args.expected_steps,
        "world_size": args.world_size,
        "per_device_batch": args.per_device_batch,
        "num_generations": args.num_generations,
        "reward_types": list(args.reward_types),
        "format_reward_weight": args.format_reward_weight if paired_format_type == "format_scaled" else None,
        "event_counts": dict(sorted(counts.items())),
        "verified_generation_groups": group_count,
        "errors": dict(sorted(errors.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
