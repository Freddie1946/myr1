#!/usr/bin/env python3
"""Guarded formal Phase 1 queue: R0/R1, L-r32, evaluation, selection."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/home/dataset-assist-0/czy/wjy/myr1")
WORK = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100")
EVAL_WORK = Path("/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100")
MODEL = WORK / "runs/formal_selected_rule_rl1000_20260811/parents/sft_step080_merged"
LAUNCHER = REPO / "scripts/launch_formal_selected_rule_rl1000.sh"
SFT_PREPARE = REPO / "scripts/prepare_formal_selected_sft3000.py"
SFT_LAUNCHER = REPO / "scripts/launch_formal_selected_sft3000.sh"
DECISION = EVAL_WORK / "runs/visual_adaptation_followup_n1500_e3_2seed_20260811/architecture_decision.json"
BASE_MODEL = WORK / "models/Qwen2.5-VL-7B-Instruct-cc594898137f460bfe9f0759e9844b3ce807cfb5"
SFT_DATA = WORK / "data/pathmmu_image_disjoint_v2/llamafactory"
PYTHON = WORK / "envs/grpo/bin/python"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_step(state_path: Path, state: dict, name: str, command: list[str], env: dict[str, str], sentinels: list[Path]) -> None:
    if all(path.is_file() for path in sentinels):
        state["steps"].append({"name": name, "status": "already_completed"})
        write_json(state_path, state)
        return
    if any(path.exists() for path in sentinels):
        raise RuntimeError(f"partial sentinels for {name}: {sentinels}")
    row = {"name": name, "status": "running", "started_at": now()}
    state["steps"].append(row)
    write_json(state_path, state)
    merged_env = os.environ.copy()
    merged_env.update(env)
    try:
        subprocess.run(command, check=True, cwd=REPO, env=merged_env)
        missing = [str(path) for path in sentinels if not path.is_file()]
        if missing:
            raise RuntimeError(f"missing task sentinels: {missing}")
    except Exception as exc:
        row.update(status="failed", finished_at=now(), error=str(exc))
        state.update(status="failed", finished_at=now())
        write_json(state_path, state)
        raise
    row.update(status="completed", finished_at=now())
    write_json(state_path, state)


def rl_env(run_dir: Path, reward_mode: str, port: int) -> dict[str, str]:
    return {
        "MODEL_PATH": str(MODEL), "RUN_DIR": str(run_dir), "MODE": "formal",
        "RUN_ROLE": "reward_isolation_one_epoch", "PER_DEVICE_BATCH": "10",
        "GRADIENT_CHECKPOINTING": "false", "MAX_COMPLETION_LENGTH": "384",
        "LEARNING_RATE": "1.0e-6", "REWARD_MODE": reward_mode,
        "GENERATION_TEMPERATURE": "0.9", "GENERATION_TOP_P": "1.0",
        "GENERATION_TOP_K": "0", "GENERATION_TYPICAL_P": "1.0",
        "GENERATION_REPETITION_PENALTY": "1.0", "MAX_STEPS_OVERRIDE": "50",
        "EXPLICIT_SAVE_STEPS": "10,25,50", "SAVE_LIMIT_OVERRIDE": "3",
        "MASTER_PORT": str(port),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-root", type=Path,
        default=WORK / "runs/rule_rl_mechanism_funnel_20260811/preformal_smokes_v1",
    )
    parser.add_argument(
        "--train-root", type=Path,
        default=WORK / "runs/rule_rl_mechanism_funnel_20260811/formal_v1",
    )
    parser.add_argument(
        "--eval-root", type=Path,
        default=EVAL_WORK / "runs/rule_rl_mechanism_funnel_20260811/formal_v1/phase1_core",
    )
    parser.add_argument("--wait-seconds", type=int, default=60)
    args = parser.parse_args()
    args.train_root.mkdir(parents=True, exist_ok=True)
    with (args.train_root / ".phase1.lock").open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another Phase 1 queue is active") from exc
        state_path = args.train_root / "phase1_state.json"
        state = {"schema_version": 1, "status": "waiting_for_all_smokes", "started_at": now(), "steps": []}
        write_json(state_path, state)
        while not (args.smoke_root / "ALL_SMOKES_PASSED").is_file():
            smoke_state = args.smoke_root / "queue_state.json"
            if smoke_state.is_file() and json.loads(smoke_state.read_text()).get("status") == "failed":
                state.update(status="blocked_by_failed_smoke", finished_at=now())
                write_json(state_path, state)
                return
            time.sleep(args.wait_seconds)
        state["status"] = "running"
        state["all_smokes_passed_at"] = (args.smoke_root / "ALL_SMOKES_PASSED").read_text().strip()
        state["formal_training_started_at"] = now()
        write_json(state_path, state)

        r0 = args.train_root / "phase1/r0_short"
        r1 = args.train_root / "phase1/r1_short"
        run_step(state_path, state, "r0_short", ["bash", str(LAUNCHER)], rl_env(r0, "accuracy_format", 33901), [r0 / "reward_alignment_verification.json", r0 / "output/checkpoint-50/adapter_model.safetensors"])
        run_step(state_path, state, "r1_short", ["bash", str(LAUNCHER)], rl_env(r1, "accuracy_only", 33902), [r1 / "reward_alignment_verification.json", r1 / "output/checkpoint-50/adapter_model.safetensors"])

        sft = args.train_root / "phase3/l_r32_sft3000"
        if not sft.exists():
            subprocess.run([
                str(WORK / "envs/sft/bin/python"), str(SFT_PREPARE),
                "--decision", str(DECISION), "--model", str(BASE_MODEL),
                "--dataset-dir", str(SFT_DATA), "--run-dir", str(sft),
                "--seed", "42", "--epochs", "3", "--micro-batch", "12",
                "--lora-rank", "32", "--lora-alpha", "64",
            ], check=True, cwd=REPO)
        run_step(state_path, state, "l_r32_sft3000", ["bash", str(SFT_LAUNCHER), str(sft)], {}, [sft / "output/train_results.json", sft / "output/checkpoint-96/adapter_model.safetensors"])

        run_step(
            state_path, state, "phase1_core_evaluation",
            [str(PYTHON), str(REPO / "scripts/run_rule_rl_core_evaluations.py"),
             "--train-root", str(args.train_root), "--eval-root", str(args.eval_root),
             "--arm", "r0:phase1/r0_short", "--arm", "r1:phase1/r1_short"],
            {}, [args.eval_root / "evaluation_state.json", args.eval_root / "pathmmu_val/r1_step050/metrics.json"],
        )
        selection = args.train_root / "phase1/reward_selection.json"
        run_step(
            state_path, state, "phase1_reward_selection",
            [str(PYTHON), str(REPO / "scripts/summarize_rule_rl_reward_isolation.py"),
             "--train-root", str(args.train_root), "--eval-root", str(args.eval_root), "--output", str(selection)],
            {"PYTHONPATH": str(REPO / "scripts")}, [selection],
        )
        decision = json.loads(selection.read_text(encoding="utf-8"))
        state["status"] = "r2_required" if decision["r2_required"] else "phase1_complete_reward_selected"
        state["selected_reward_arm"] = decision["selected_reward_arm"]
        state["finished_at"] = now()
        write_json(state_path, state)
        marker = "R2_REQUIRED" if decision["r2_required"] else "PHASE1_COMPLETE"
        (args.train_root / marker).write_text(now() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
