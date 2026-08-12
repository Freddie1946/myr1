#!/usr/bin/env python3
"""Run the two mandatory lean Phase-2 screens sequentially, then stop."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/home/dataset-assist-0/czy/wjy/myr1")
WORK = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100")
EVAL_WORK = Path("/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100")
MODEL = WORK / "runs/formal_selected_rule_rl1000_20260811/parents/sft_step080_merged"
LAUNCHER = REPO / "scripts/launch_formal_selected_rule_rl1000.sh"
PYTHON = WORK / "envs/grpo/bin/python"
DECISION = REPO / "protocol/rule_rl_phase1_reward_override_20260811.json"
L32_SUMMARY = EVAL_WORK / "runs/rule_rl_mechanism_funnel_20260811/formal_v1/l_r32_sft_core/core_summary.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
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
    merged = os.environ.copy(); merged.update(env)
    try:
        subprocess.run(command, cwd=REPO, env=merged, check=True)
        missing = [str(path) for path in sentinels if not path.is_file()]
        if missing:
            raise RuntimeError(f"missing sentinels: {missing}")
    except Exception as exc:
        row.update(status="failed", finished_at=now(), error=str(exc))
        state.update(status="failed", finished_at=now())
        write_json(state_path, state)
        raise
    row.update(status="completed", finished_at=now())
    write_json(state_path, state)


def rl_env(run_dir: Path, *, per_device: int, lr: str, max_steps: int, saves: str, stop: int, port: int) -> dict[str, str]:
    return {
        "MODEL_PATH": str(MODEL), "RUN_DIR": str(run_dir), "MODE": "formal",
        "RUN_ROLE": "lean_mechanism_screening_half_epoch", "PER_DEVICE_BATCH": str(per_device),
        "MAX_COMPLETION_LENGTH": "384", "LEARNING_RATE": lr, "REWARD_MODE": "accuracy_only",
        "LORA_R": "16", "LORA_ALPHA": "32", "GENERATION_TEMPERATURE": "0.9",
        "GENERATION_TOP_P": "1.0", "GENERATION_TOP_K": "0", "GRADIENT_CHECKPOINTING": "false",
        "MAX_STEPS_OVERRIDE": str(max_steps), "EXPLICIT_SAVE_STEPS": saves,
        "STOP_AFTER_SAVED_STEP": str(stop), "VERIFY_EXPECTED_STEPS": str(stop),
        "SAVE_LIMIT_OVERRIDE": "3", "MASTER_PORT": str(port),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, default=WORK / "runs/rule_rl_mechanism_funnel_20260811/formal_v1/phase2_lean")
    parser.add_argument("--eval-root", type=Path, default=EVAL_WORK / "runs/rule_rl_mechanism_funnel_20260811/formal_v1/phase2_lean")
    args = parser.parse_args()
    decision = json.loads(DECISION.read_text())
    if decision["adjudication"]["provisional_reward"] != "accuracy_only" or decision["adjudication"]["r2_run_now"]:
        raise RuntimeError("Phase-1 adjudication does not authorize R1 screening")
    if not L32_SUMMARY.is_file():
        raise RuntimeError(f"L-r32 core evaluation must complete first: {L32_SUMMARY}")
    args.train_root.mkdir(parents=True, exist_ok=True)
    with (args.train_root / ".queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = args.train_root / "queue_state.json"
        state = {"schema_version": 1, "status": "running", "started_at": now(), "steps": [], "test_accessed": False}
        write_json(state_path, state)
        lr3 = args.train_root / "g20_lr3"
        g2 = args.train_root / "g2_lr1"
        run_step(state_path, state, "r1_g20_lr3_step25", ["bash", str(LAUNCHER)], rl_env(lr3, per_device=10, lr="3e-6", max_steps=50, saves="10,25", stop=25, port=34101), [lr3 / "output/checkpoint-25/adapter_model.safetensors", lr3 / "reward_alignment_verification.json"])
        run_step(state_path, state, "r1_g2_lr1_step250", ["bash", str(LAUNCHER)], rl_env(g2, per_device=1, lr="1e-6", max_steps=500, saves="100,250", stop=250, port=34102), [g2 / "output/checkpoint-250/adapter_model.safetensors", g2 / "reward_alignment_verification.json"])
        run_step(state_path, state, "matched_exposure_evaluation", [str(PYTHON), str(REPO / "scripts/run_rule_rl_lean_phase2_evaluations.py"), "--train-root", str(args.train_root), "--eval-root", str(args.eval_root)], {}, [args.eval_root / "evaluation_state.json", args.eval_root / "pathmmu_val/g2_lr1_step250/metrics.json"])
        summary = args.train_root / "matched_exposure_summary.json"
        run_step(state_path, state, "matched_exposure_summary", [str(PYTHON), str(REPO / "scripts/summarize_rule_rl_lean_phase2.py"), "--train-root", str(args.train_root), "--eval-root", str(args.eval_root), "--phase1-train-root", str(args.train_root.parent), "--phase1-eval-root", str(EVAL_WORK / "runs/rule_rl_mechanism_funnel_20260811/formal_v1/phase1_core"), "--output", str(summary)], {"PYTHONPATH": str(REPO / "scripts")}, [summary])
        state.update(status="complete_manual_analysis_required", finished_at=now())
        write_json(state_path, state)
        (args.train_root / "LEAN_PHASE2_COMPLETE").write_text(now() + "\n")


if __name__ == "__main__":
    main()
