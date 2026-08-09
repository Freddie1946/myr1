#!/usr/bin/env python3
"""Wait for future formal ratio tasks and back up terminal model-only outputs."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from backup_data_ratio_models_to_hf import atomic_json, now_iso, upload_one


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def formal_task_complete(formal_state: Path, task_id: str, expected_step: int) -> bool:
    state = load_json(formal_state)
    entry = state.get("tasks", {}).get(task_id, {})
    if entry.get("status") != "completed":
        if state.get("status") == "failed":
            raise RuntimeError(
                f"formal queue failed before {task_id} completed: {state.get('current_task')}"
            )
        return False
    audit = formal_state.parent / "tasks" / task_id / "output/pathvlm_ratio_train_state_audit.json"
    if not audit.is_file():
        raise RuntimeError(f"completed task has no train-state audit: {audit}")
    reached = int(load_json(audit).get("global_step", -1))
    if reached != expected_step:
        raise RuntimeError(f"{task_id} reached {reached}, expected {expected_step}")
    return True


def wait_for_existing_backup(path: Path, poll_seconds: int) -> None:
    while True:
        if path.is_file():
            state = load_json(path)
            if state.get("status") == "completed":
                return
            if state.get("status") == "failed":
                raise RuntimeError("completed-arm model backup failed; future backup will not overlap it")
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--completed-backup-state", type=Path, required=True)
    parser.add_argument("--hf-bin", default="/usr/local/bin/hf")
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args()
    formal_root = args.formal_root.resolve()
    state_root = args.state_root.resolve()
    formal_state = formal_root / "state.json"
    tasks = [
        {
            "task_id": "sft0250_rl0750_rule_rl",
            "repo_id": "Freddie1946/PathVLM-R1-Ratio-SFT0250-RuleRL0750-seed42",
            "expected_step": 1125,
        },
        {
            "task_id": "stage2_continue_rule_rl1000",
            "repo_id": "Freddie1946/PathVLM-R1-Stage2-Continued-RuleRL1000-seed42",
            "expected_step": 1500,
        },
        {
            "task_id": "base_rule_rl4000",
            "repo_id": "Freddie1946/PathVLM-R1-Base-RuleRL4000-seed42",
            "expected_step": 6000,
        },
    ]
    for task in tasks:
        task["source"] = formal_root / "tasks" / task["task_id"] / "output"
    state_path = state_root / "state.json"
    state = (
        load_json(state_path)
        if state_path.is_file()
        else {"schema_version": 1, "status": "waiting", "tasks": {}, "created_at": now_iso()}
    )
    state.update({"status": "waiting", "pid": os.getpid(), "updated_at": now_iso()})
    atomic_json(state_path, state)
    wait_for_existing_backup(args.completed_backup_state.resolve(), args.poll_seconds)
    api = HfApi()
    for task in tasks:
        if state["tasks"].get(task["task_id"], {}).get("status") == "completed":
            continue
        state["current_task"] = task["task_id"]
        state["tasks"][task["task_id"]] = {
            "status": "waiting_for_formal_completion",
            "repo_id": task["repo_id"],
            "expected_step": task["expected_step"],
        }
        atomic_json(state_path, state)
        while not formal_task_complete(formal_state, task["task_id"], task["expected_step"]):
            time.sleep(args.poll_seconds)
        state["status"] = "running"
        state["tasks"][task["task_id"]].update(
            {"status": "running", "started_at": now_iso()}
        )
        atomic_json(state_path, state)
        verification = upload_one(api, args.hf_bin, task, state_root)
        state["tasks"][task["task_id"]].update(
            {"status": "completed", "completed_at": now_iso(), "verification": verification}
        )
        state["updated_at"] = now_iso()
        atomic_json(state_path, state)
    state.update({"status": "completed", "completed_at": now_iso(), "current_task": None})
    atomic_json(state_path, state)


if __name__ == "__main__":
    main()
