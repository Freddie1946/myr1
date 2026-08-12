#!/usr/bin/env python3
"""Summarize reward-isolation checkpoints and apply a predeclared promotion rule."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from pathmmu_rewards import format_reward


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def reward_prefix(run_dir: Path, step: int, reward_mode: str) -> dict:
    events = []
    for path in sorted((run_dir / "online_reward_events").glob("rank_*.jsonl")):
        events.extend(load_jsonl(path))
    by_key = {}
    for event in events:
        if int(event["call_index"]) >= step:
            continue
        key = (int(event["rank"]), int(event["call_index"]), int(event["item_index"]))
        by_key.setdefault(key, {})[str(event["reward_type"])] = event
    accuracy_rows = [values["accuracy"] for values in by_key.values() if "accuracy" in values]
    expected = step * 8 * int(load_json(run_dir / "launch_contract.json")["per_device_train_batch_size"])
    if len(accuracy_rows) != expected:
        raise RuntimeError(f"{run_dir.name} step{step}: {len(accuracy_rows)} != {expected}")
    grouped: defaultdict[int, list[tuple[tuple[int, int, int], dict]]] = defaultdict(list)
    for key, values in by_key.items():
        grouped[key[1]].append((key, values))
    active = zero = total_groups = 0
    for call_index in range(step):
        rows = sorted(grouped[call_index], key=lambda item: (item[0][0], item[0][2]))
        for start in range(0, len(rows), 4):
            totals = []
            for _, values in rows[start : start + 4]:
                reward = float(values["accuracy"]["reward"])
                if reward_mode == "accuracy_format":
                    reward += float(values["format"]["reward"])
                totals.append(reward)
            total_groups += 1
            if max(totals) == min(totals):
                zero += 1
            else:
                active += 1
    sampled_format = statistics.fmean(
        format_reward([[{"role": "assistant", "content": str(row["completion"])}]])[0]
        for row in accuracy_rows
    )
    return {
        "sampled_pass_rate": statistics.fmean(float(row["reward"]) for row in accuracy_rows),
        "sampled_format_rate": sampled_format,
        "group_count": total_groups,
        "active_group_count": active,
        "zero_advantage_group_count": zero,
        "active_group_rate": active / total_groups,
    }


def flips(parent_path: Path, child_path: Path) -> dict:
    parent = load_jsonl(parent_path)
    child = load_jsonl(child_path)
    if len(parent) != len(child):
        raise RuntimeError("paired prediction counts differ")
    c2w = w2c = stable_correct = stable_wrong = 0
    for left, right in zip(parent, child):
        if left["source_record_sha256"] != right["source_record_sha256"]:
            raise RuntimeError("paired source hashes differ")
        before = bool(left["accuracy_reward"])
        after = bool(right["accuracy_reward"])
        if before and not after:
            c2w += 1
        elif not before and after:
            w2c += 1
        elif before:
            stable_correct += 1
        else:
            stable_wrong += 1
    return {
        "parent_correct_to_child_wrong": c2w,
        "parent_wrong_to_child_correct": w2c,
        "net_correct_flip": w2c - c2w,
        "stable_correct": stable_correct,
        "stable_wrong": stable_wrong,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--r0-dir", default="phase1/r0_short")
    parser.add_argument("--r1-dir", default="phase1/r1_short")
    args = parser.parse_args()
    arms = {"r0": (args.train_root / args.r0_dir, "accuracy_format"), "r1": (args.train_root / args.r1_dir, "accuracy_only")}
    parent = {
        split: load_json(args.eval_root / split / "parent" / "metrics.json")
        for split in ("train_probe", "pathmmu_val")
    }
    rows = []
    for arm, (run_dir, reward_mode) in arms.items():
        for step in (10, 25, 50):
            name = f"{arm}_step{step:03d}"
            split_metrics = {
                split: load_json(args.eval_root / split / name / "metrics.json")
                for split in ("train_probe", "pathmmu_val")
            }
            row = {
                "arm": arm,
                "reward_mode": reward_mode,
                "step": step,
                "exposure_fraction": step / 50,
                "train_probe": {
                    "correct": split_metrics["train_probe"]["correct"],
                    "count": split_metrics["train_probe"]["count"],
                    "accuracy": split_metrics["train_probe"]["accuracy"],
                    "format_rate": split_metrics["train_probe"]["format_correct"] / split_metrics["train_probe"]["count"],
                    "flips": flips(
                        args.eval_root / "train_probe/parent/predictions.jsonl",
                        args.eval_root / f"train_probe/{name}/predictions.jsonl",
                    ),
                },
                "pathmmu_val": {
                    "correct": split_metrics["pathmmu_val"]["correct"],
                    "count": split_metrics["pathmmu_val"]["count"],
                    "accuracy": split_metrics["pathmmu_val"]["accuracy"],
                    "format_rate": split_metrics["pathmmu_val"]["format_correct"] / split_metrics["pathmmu_val"]["count"],
                    "flips": flips(
                        args.eval_root / "pathmmu_val/parent/predictions.jsonl",
                        args.eval_root / f"pathmmu_val/{name}/predictions.jsonl",
                    ),
                },
                "rollout_prefix": reward_prefix(run_dir, step, reward_mode),
            }
            rows.append(row)

    final = {row["arm"]: row for row in rows if row["step"] == 50}
    aggregate_gain = (
        final["r1"]["train_probe"]["correct"] + final["r1"]["pathmmu_val"]["correct"]
        - final["r0"]["train_probe"]["correct"] - final["r0"]["pathmmu_val"]["correct"]
    )
    r1_noninferior = (
        final["r1"]["train_probe"]["correct"] >= final["r0"]["train_probe"]["correct"] - 2
        and final["r1"]["pathmmu_val"]["correct"] >= final["r0"]["pathmmu_val"]["correct"] - 2
    )
    r1_correctness_winner = aggregate_gain >= 4 and r1_noninferior
    format_drop = max(
        final["r0"][split]["format_rate"] - final["r1"][split]["format_rate"]
        for split in ("train_probe", "pathmmu_val")
    )
    r2_required = r1_correctness_winner and (
        format_drop > 0.01 or final["r1"]["rollout_prefix"]["sampled_format_rate"] < 0.98
    )
    selected = None if r2_required else ("r1" if r1_correctness_winner else "r0")
    payload = {
        "schema_version": 1,
        "status": "r2_required" if r2_required else "reward_selected",
        "test_accessed": False,
        "parent": {
            split: {"correct": value["correct"], "count": value["count"], "accuracy": value["accuracy"]}
            for split, value in parent.items()
        },
        "predeclared_gate": {
            "r1_minimum_aggregate_net_correct_gain": 4,
            "per_split_noninferiority_cases": 2,
            "noticeable_eval_format_drop": 0.01,
            "minimum_sampled_format_rate_without_r2": 0.98,
        },
        "aggregate_r1_minus_r0_correct": aggregate_gain,
        "r1_noninferior": r1_noninferior,
        "maximum_r1_eval_format_drop": format_drop,
        "r2_required": r2_required,
        "selected_reward_arm": selected,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
