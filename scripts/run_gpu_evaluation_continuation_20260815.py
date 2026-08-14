#!/usr/bin/env python3
"""Durable parallel continuation for the current PathMMU repeat-inference gaps.

Each lane owns one GPU and uses the existing resumable five-seed runner.  A
failed lane is retried up to three times; completed seed metrics and summaries
are always skipped on restart.  OmniMedVQA is deliberately excluded until the
manuscript-faithful unified evaluation contract is frozen.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORK = Path("/home/dataset-assist-0/czy/wjy")
REPO = WORK / "myr1"
RUNNER = REPO / "scripts/run_repeated_pathmmu_inference.sh"
OUTPUT_ROOT = WORK / "pathvlm_revision_eval_a100/runs/repeated_inference_pathmmu_test999_20260815"
TASKS = (
    {
        "id": "stage2_outcome_grpo",
        "gpu": 0,
        "model": WORK / "pathvlm_r1_v1_a100/transferred_checkpoints/outcome_grpo_n1000_seed42_epoch02_step1000",
    },
    {
        "id": "stage2_continued_rule_rl1000",
        "gpu": 1,
        "model": WORK / "pathvlm_r1_v1_a100/runs/data_ratio_rule_rl_ablation_v1/formal_sequence/tasks/stage2_continue_rule_rl1000/output/checkpoint-1500",
    },
    {
        "id": "stage3_gpt4o_step500",
        "gpu": 2,
        "model": WORK / "pathvlm_r1_v1_a100/runs/stage3_process_grpo/gpt4o_n8_parent_full_20260813/epoch_model_snapshots/checkpoint-500",
    },
    {
        "id": "stage3_gpt4o_step1000",
        "gpu": 4,
        "model": WORK / "pathvlm_r1_v1_a100/runs/stage3_process_grpo/gpt4o_n8_parent_full_20260813/epoch_model_snapshots/checkpoint-1000",
    },
    {
        "id": "rule_rl4000_checkpoint2500_provisional",
        "gpu": 7,
        "model": WORK / "pathvlm_r1_v1_a100/runs/data_ratio_rule_rl_ablation_v1/formal_sequence/tasks/base_rule_rl4000/output/checkpoint-2500",
    },
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def atomic_json(path: Path, value: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
        temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, path)


def gpu_memory_mib(gpu: int) -> int:
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(result.stdout.strip())


def run_task(
    task: dict[str, Any], state: dict[str, Any], state_path: Path, lock: threading.Lock,
    max_attempts: int, retry_seconds: int, poll_seconds: int,
) -> None:
    task_id = task["id"]
    model = Path(task["model"])
    if not model.is_dir():
        raise FileNotFoundError(model)
    summary = OUTPUT_ROOT / task_id / "summary.json"
    if summary.is_file():
        state["tasks"][task_id].update({"status": "completed", "completed_at": now(), "reused": True})
        atomic_json(state_path, state, lock)
        return

    while gpu_memory_mib(task["gpu"]) > 2048:
        state["tasks"][task_id].update({"status": "waiting_for_gpu", "updated_at": now()})
        atomic_json(state_path, state, lock)
        time.sleep(poll_seconds)

    log_path = state_path.parent / "logs" / f"{task_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_attempts + 1):
        state["tasks"][task_id].update({"status": "running", "attempt": attempt, "updated_at": now()})
        atomic_json(state_path, state, lock)
        command = [
            "bash", str(RUNNER), str(task["gpu"]), task_id, str(model), str(OUTPUT_ROOT), "-",
        ]
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{now()}] attempt={attempt} command={' '.join(command)}\n")
            log.flush()
            code = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
        if code == 0 and summary.is_file():
            state["tasks"][task_id].update({"status": "completed", "completed_at": now()})
            atomic_json(state_path, state, lock)
            return
        state["tasks"][task_id].update({"status": "retry_wait", "returncode": code, "updated_at": now()})
        atomic_json(state_path, state, lock)
        if attempt < max_attempts:
            time.sleep(retry_seconds)
    raise RuntimeError(f"{task_id} failed after {max_attempts} attempts; see {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-seconds", type=int, default=60)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    state_root = args.state_root.resolve()
    state_path = state_root / "state.json"
    lock = threading.Lock()
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": now(),
        "pid": os.getpid(),
        "contract": "PathMMU test999 sampled inference; seeds 42-46; T=0.7; top_p=0.9; top_k disabled",
        "omnimedvqa_policy": "not queued until unified manuscript-faithful contract is frozen",
        "tasks": {
            task["id"]: {"status": "queued", "gpu": task["gpu"], "model": str(task["model"])}
            for task in TASKS
        },
    }
    atomic_json(state_path, state, lock)
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(TASKS)) as executor:
        futures = {
            executor.submit(
                run_task, task, state, state_path, lock,
                args.max_attempts, args.retry_seconds, args.poll_seconds,
            ): task["id"]
            for task in TASKS
        }
        for future in concurrent.futures.as_completed(futures):
            task_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append(task_id)
                state["tasks"][task_id].update({"status": "failed", "error": repr(exc), "updated_at": now()})
                atomic_json(state_path, state, lock)
    state.update({"status": "failed" if failures else "completed", "completed_at": now(), "failures": failures})
    atomic_json(state_path, state, lock)
    if failures:
        raise SystemExit(f"failed tasks: {failures}")


if __name__ == "__main__":
    main()
