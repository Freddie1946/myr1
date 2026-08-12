#!/usr/bin/env python3
"""Run every mechanism-funnel smoke sequentially before any formal arm."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/home/dataset-assist-0/czy/wjy/myr1")
WORK = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100")
MODEL = WORK / "runs/formal_selected_rule_rl1000_20260811/parents/sft_step080_merged"
LAUNCHER = REPO / "scripts/launch_formal_selected_rule_rl1000.sh"
SFT_PREPARE = REPO / "scripts/prepare_formal_selected_sft3000.py"
SFT_LAUNCHER = REPO / "scripts/launch_formal_selected_sft3000.sh"
DECISION = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/"
    "visual_adaptation_followup_n1500_e3_2seed_20260811/architecture_decision.json"
)
BASE_MODEL = WORK / "models/Qwen2.5-VL-7B-Instruct-cc594898137f460bfe9f0759e9844b3ce807cfb5"
SFT_DATA = WORK / "data/pathmmu_image_disjoint_v2/llamafactory"


@dataclass(frozen=True)
class Task:
    name: str
    command: tuple[str, ...]
    env: dict[str, str]
    sentinels: tuple[Path, ...]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def rl_env(run_dir: Path, reward: str, batch: int, lr: str, port: int, **extra: str) -> dict[str, str]:
    values = {
        "MODEL_PATH": str(MODEL),
        "RUN_DIR": str(run_dir),
        "MODE": "smoke",
        "RUN_ROLE": "mechanism_funnel_preformal_smoke",
        "PER_DEVICE_BATCH": str(batch),
        "GRADIENT_CHECKPOINTING": "false",
        "MAX_COMPLETION_LENGTH": "384",
        "LEARNING_RATE": lr,
        "REWARD_MODE": reward,
        "GENERATION_TEMPERATURE": "0.9",
        "GENERATION_TOP_P": "1.0",
        "GENERATION_TOP_K": "0",
        "GENERATION_TYPICAL_P": "1.0",
        "GENERATION_REPETITION_PENALTY": "1.0",
        "MAX_STEPS_OVERRIDE": "2",
        "SAVE_LIMIT_OVERRIDE": "3",
        "MASTER_PORT": str(port),
    }
    values.update(extra)
    return values


def build_tasks(root: Path) -> list[Task]:
    tasks: list[Task] = []
    specs = (
        ("r0_g20_l1", "accuracy_format", 10, "1.0e-6", 33801),
        ("r1_g20_l1", "accuracy_only", 10, "1.0e-6", 33802),
        ("g20_l3", "accuracy_only", 10, "3.0e-6", 33803),
        ("g2_l1", "accuracy_only", 1, "1.0e-6", 33804),
    )
    for name, reward, batch, lr, port in specs:
        run_dir = root / name
        tasks.append(Task(
            name=name,
            command=("bash", str(LAUNCHER)),
            env=rl_env(run_dir, reward, batch, lr, port),
            sentinels=(run_dir / "reward_alignment_verification.json",),
        ))

    gated = root / "g20_l10_gated_resume"
    tasks.append(Task(
        name="g20_l10_gate_segment1",
        command=("bash", str(LAUNCHER)),
        env=rl_env(
            gated, "accuracy_only", 10, "1.0e-5", 33805,
            EXPLICIT_SAVE_STEPS="1,2", STOP_AFTER_SAVED_STEP="1",
            REWARD_LOG_SUBDIR="online_reward_events_segment1",
            LAUNCH_CONTRACT_NAME="launch_contract_segment1.json",
            TRAIN_LOG_NAME="train_segment1.log",
            ALIGNMENT_OUTPUT_NAME="reward_alignment_segment1.json",
            TRAINING_SEGMENT="smoke_gate_segment1",
        ),
        sentinels=(gated / "reward_alignment_segment1.json", gated / "output/checkpoint-1/scheduler.pt"),
    ))
    tasks.append(Task(
        name="g20_l10_gate_segment2_resume",
        command=("bash", str(LAUNCHER)),
        env=rl_env(
            gated, "accuracy_only", 10, "1.0e-5", 33806,
            EXPLICIT_SAVE_STEPS="1,2", STOP_AFTER_SAVED_STEP="2",
            RESUME_FROM_CHECKPOINT=str(gated / "output/checkpoint-1"),
            VERIFY_EXPECTED_STEPS="1",
            REWARD_LOG_SUBDIR="online_reward_events_segment2",
            LAUNCH_CONTRACT_NAME="launch_contract_segment2.json",
            TRAIN_LOG_NAME="train_segment2.log",
            ALIGNMENT_OUTPUT_NAME="reward_alignment_segment2.json",
            TRAINING_SEGMENT="smoke_gate_segment2_resume",
        ),
        sentinels=(gated / "reward_alignment_segment2.json", gated / "output/checkpoint-2/scheduler.pt"),
    ))

    sft_dir = root / "l_r32_sft3000"
    tasks.append(Task(
        name="l_r32_sft3000",
        command=("bash", str(SFT_LAUNCHER), str(sft_dir)),
        env={"MIN_CHECKPOINT_COUNT": "0"},
        sentinels=(sft_dir / "output/train_results.json", sft_dir / "output/adapter_model.safetensors"),
    ))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=WORK / "runs/rule_rl_mechanism_funnel_20260811/preformal_smokes_v1",
    )
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    lock_path = args.root / ".queue.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another smoke queue owns this root") from exc

        sft_dir = args.root / "l_r32_sft3000"
        if not sft_dir.exists():
            subprocess.run(
                [
                    str(WORK / "envs/sft/bin/python"), str(SFT_PREPARE),
                    "--decision", str(DECISION), "--model", str(BASE_MODEL),
                    "--dataset-dir", str(SFT_DATA), "--run-dir", str(sft_dir),
                    "--seed", "42", "--epochs", "3", "--micro-batch", "12",
                    "--lora-rank", "32", "--lora-alpha", "64", "--max-steps", "2",
                ],
                check=True,
                cwd=REPO,
            )

        tasks = build_tasks(args.root)
        state_path = args.root / "queue_state.json"
        state = {
            "schema_version": 1,
            "status": "running",
            "started_at": utc_now(),
            "root": str(args.root.resolve()),
            "formal_training_started": False,
            "tasks": [],
        }
        write_json_atomic(state_path, state)
        for task in tasks:
            if all(path.is_file() for path in task.sentinels):
                state["tasks"].append({"name": task.name, "status": "already_completed"})
                write_json_atomic(state_path, state)
                continue
            if any(path.exists() for path in task.sentinels):
                raise RuntimeError(f"partial sentinels for {task.name}: {task.sentinels}")
            row = {"name": task.name, "status": "running", "started_at": utc_now()}
            state["tasks"].append(row)
            write_json_atomic(state_path, state)
            env = os.environ.copy()
            env.update(task.env)
            try:
                subprocess.run(task.command, check=True, cwd=REPO, env=env)
                missing = [str(path) for path in task.sentinels if not path.is_file()]
                if missing:
                    raise RuntimeError(f"task succeeded without sentinels: {missing}")
            except Exception as exc:
                row.update(status="failed", finished_at=utc_now(), error=str(exc))
                state["status"] = "failed"
                state["finished_at"] = utc_now()
                write_json_atomic(state_path, state)
                raise
            row.update(status="completed", finished_at=utc_now())
            write_json_atomic(state_path, state)
        state["status"] = "all_smokes_passed"
        state["finished_at"] = utc_now()
        write_json_atomic(state_path, state)
        (args.root / "ALL_SMOKES_PASSED").write_text(utc_now() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
