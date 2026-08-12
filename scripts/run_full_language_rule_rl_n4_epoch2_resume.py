#!/usr/bin/env python3
"""Resume n4 at step500, retain AdamW moments, and restart only the LR schedule."""

from __future__ import annotations

import collections
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/home/dataset-assist-0/czy/wjy/myr1")
PYTHON = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/grpo/bin/python")
PARENT = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/"
    "full_language_rule_rl_capacity_20260811/model_snapshots/checkpoint-500"
)
RESUME = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/"
    "full_language_rule_rl_capacity_20260811/formal_g2_lr1_step100/output/checkpoint-500"
)
DATASET_YAML = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/"
    "pathmmu_image_disjoint_v2/grpo/pathvlm_rl_n1000.yaml"
)
DATASET_JSON = DATASET_YAML.with_suffix(".json")
DEEPSPEED = REPO / "configs/deepspeed/ds_z3_gpu_torch_adamw.json"
RUN = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/"
    "full_language_rule_rl_n4_epoch2_resume_20260812"
)
STATE = RUN / "sequence_state.json"
SAVE_STEPS = tuple(range(600, 1001, 100))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def gpu_processes() -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def environment(root: Path, *, smoke: bool) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PATHVLM_ALLOW_LANGUAGE_LORA", None)
    env.update({
        "PATHVLM_SKIP_FINAL_MODEL_SAVE": "true",
        "PATHVLM_REWARD_LOG_DIR": str(root / "online_reward_events"),
        "PATHVLM_TRAINING_SEGMENT": "n4_epoch2_resume_smoke" if smoke else "n4_epoch2_resume_formal",
        "PATHVLM_TRAIN_STATE_AUDIT_NAME": "pathvlm_train_state_audit.json",
        "PATHVLM_REQUIRE_SOURCE_AUDIT": "true",
        "PATHVLM_IMAGE_HASH_MANIFEST": str(
            REPO / "data/pathmmu_image_disjoint_v2/image_content_sha256.json"
        ),
        "PATHVLM_GENERATION_TEMPERATURE": "0.9", "PATHVLM_GENERATION_TOP_P": "1.0",
        "PATHVLM_GENERATION_TOP_K": "0", "PATHVLM_GENERATION_TYPICAL_P": "1.0",
        "PATHVLM_GENERATION_REPETITION_PENALTY": "1.0",
        "PATHVLM_RESUME_FROM_CHECKPOINT": str(RESUME),
        "PATHVLM_RESET_SCHEDULER_ON_RESUME_STEPS": "2" if smoke else "500",
        "PATHVLM_RESET_SCHEDULER_PEAK_LR": "5e-7",
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1", "WANDB_MODE": "disabled",
        "TOKENIZERS_PARALLELISM": "false", "DEBUG_MODE": "false",
        "OMP_NUM_THREADS": "8", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    if smoke:
        for key in ("PATHVLM_EXPLICIT_SAVE_STEPS", "PATHVLM_STOP_AFTER_SAVED_STEP",
                    "PATHVLM_EPOCH_SNAPSHOT_STEPS", "PATHVLM_EPOCH_SNAPSHOT_DIR",
                    "PATHVLM_SNAPSHOT_STEP_OFFSET"):
            env.pop(key, None)
    else:
        text = ",".join(map(str, SAVE_STEPS))
        env.update({
            "PATHVLM_EXPLICIT_SAVE_STEPS": text,
            "PATHVLM_STOP_AFTER_SAVED_STEP": "1000",
            "PATHVLM_EPOCH_SNAPSHOT_STEPS": text,
            "PATHVLM_EPOCH_SNAPSHOT_DIR": str(RUN / "model_snapshots"),
            "PATHVLM_SNAPSHOT_STEP_OFFSET": "0",
        })
    return env


def command(output: Path, max_steps: int, port: int, save_limit: int) -> list[str]:
    return [
        str(PYTHON), "-m", "torch.distributed.run", "--nproc_per_node=8",
        f"--master_port={port}", str(REPO / "scripts/grpo_pathmmu.py"),
        "--deepspeed", str(DEEPSPEED), "--output_dir", str(output),
        "--model_name_or_path", str(PARENT), "--dataset_name", str(DATASET_YAML),
        "--image_root", "/", "--reward_funcs", "accuracy",
        "--freeze_vision_modules", "true", "--max_pixels", "65536",
        "--min_pixels", "3136", "--num_generations", "4",
        "--max_completion_length", "384", "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "1", "--learning_rate", "5e-7",
        "--logging_steps", "1", "--bf16", "true", "--torch_dtype", "bfloat16",
        "--gradient_checkpointing", "true", "--attn_implementation", "sdpa",
        "--beta", "0.04", "--num_iterations", "1", "--temperature", "0.9",
        "--save_strategy", "no", "--save_steps", "100", "--save_total_limit",
        str(save_limit), "--save_only_model", "false", "--report_to", "none",
        "--seed", "42", "--data_seed", "42", "--remove_unused_columns", "false",
        "--max_steps", str(max_steps),
    ]


def verify_audit(path: Path, expected_global_step: int, expected_new_rows: int) -> dict:
    audit = json.loads(path.read_text())
    params = audit["trainability"]["parameters"]
    rows = [row for row in audit["log_history"] if "grad_norm" in row and row.get("step", 0) > 500]
    restart = audit.get("scheduler_restart_on_resume")
    if not (
        audit["global_step"] == expected_global_step and len(rows) == expected_new_rows
        and audit["resume_from_checkpoint"] == str(RESUME)
        and restart and restart["optimizer_state_resumed"]
        and float(restart["peak_lr"]) == 5e-7
        and params["language"]["trainable"] == params["language"]["total"]
        and params["visual"]["trainable"] == 0
        and params["multimodal_projector"]["trainable"] == 0
        and all(math.isfinite(float(row["grad_norm"])) for row in rows)
        and any(float(row["grad_norm"]) > 0 for row in rows)
        and 0 < float(rows[0]["learning_rate"]) <= 5e-7
        and float(rows[-1]["learning_rate"]) == 0.0
    ):
        raise RuntimeError("n4 epoch2 resume audit gate failed")
    return audit


def main() -> None:
    if RUN.exists():
        raise FileExistsError(RUN)
    for path in (PARENT / "model.safetensors.index.json", RESUME / "trainer_state.json",
                 RESUME / "scheduler.pt", DATASET_YAML, DATASET_JSON, DEEPSPEED):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not (RESUME / "global_step500").is_dir():
        raise FileNotFoundError(RESUME / "global_step500")
    if gpu_processes():
        raise RuntimeError("GPU process active before n4 epoch2 resume")
    RUN.mkdir(parents=True)
    state = {"schema_version": 1, "status": "running", "stage": "smoke",
             "started_at": now(), "resume_checkpoint": str(RESUME),
             "optimizer_state_resumed": True, "scheduler_state_restarted": True,
             "scheduler_peak_lr": 5e-7, "scheduler_restart_steps": 500}
    write_json(STATE, state)
    try:
        smoke = RUN / "smoke"
        smoke.mkdir()
        with (smoke / "train.log").open("w") as handle:
            subprocess.run(command(smoke / "output", 502, 34243, 1), cwd=REPO,
                           env=environment(smoke, smoke=True), stdout=handle,
                           stderr=subprocess.STDOUT, check=True)
        verify_audit(smoke / "output/pathvlm_train_state_audit.json", 502, 2)
        state["stage"] = "formal_training"
        write_json(STATE, state)
        formal = RUN / "formal"
        formal.mkdir()
        contract = {
            "schema_version": 1, "parent_policy": str(PARENT.resolve()),
            "resume_checkpoint": str(RESUME.resolve()), "start_step": 500,
            "stop_step": 1000, "optimizer_state_resumed": True,
            "scheduler_state_loaded_then_replaced": True,
            "scheduler_restart": "linear_5e-7_to_zero_over_steps501_1000",
            "rollout_n": 4, "prompts_per_update": 2,
            "new_prompt_exposure": 1000, "new_trajectory_exposure": 4000,
            "save_steps": list(SAVE_STEPS), "model_snapshot_steps": list(SAVE_STEPS),
            "full_checkpoint_save_total_limit": 2, "reward": "accuracy_only",
            "beta": 0.04, "num_iterations": 1, "temperature": 0.9,
            "top_p": 1.0, "top_k": 0, "max_completion_length": 384,
            "vision_encoder": "frozen", "projector": "frozen",
            "language_model": "full_parameter", "test_used_for_training": False,
        }
        write_json(formal / "launch_contract.json", contract)
        with (formal / "train.log").open("w") as handle:
            subprocess.run(command(formal / "output", 1000, 34244, 2), cwd=REPO,
                           env=environment(formal, smoke=False), stdout=handle,
                           stderr=subprocess.STDOUT, check=True)
        audit = verify_audit(formal / "output/pathvlm_train_state_audit.json", 1000, 500)
        subprocess.run([
            str(PYTHON), str(REPO / "scripts/verify_grpo_reward_alignment.py"),
            "--events-dir", str(formal / "online_reward_events"),
            "--dataset-json", str(DATASET_JSON), "--expected-steps", "500",
            "--world-size", "8", "--per-device-batch", "1", "--num-generations", "4",
            "--reward-types", "accuracy", "--format-reward-weight", "0.1",
            "--output", str(formal / "reward_alignment_verification.json"),
        ], cwd=REPO, check=True)
        events = []
        for path in (formal / "online_reward_events").glob("rank_*.jsonl"):
            events.extend(json.loads(line) for line in path.open() if line.strip())
        counts = collections.Counter(int(row["record_index"]) for row in events)
        if set(counts) != set(range(1000)) or set(counts.values()) != {4}:
            raise RuntimeError("second epoch does not cover RL1000 exactly once")
        for step in SAVE_STEPS:
            snapshot = RUN / f"model_snapshots/checkpoint-{step}"
            manifest = json.loads((snapshot / "snapshot_manifest.json").read_text())
            if manifest["global_step"] != step or not (snapshot / "model.safetensors.index.json").is_file():
                raise RuntimeError(f"bad model snapshot at cumulative step {step}")
        new_rows = [row for row in audit["log_history"] if "grad_norm" in row and row.get("step", 0) > 500]
        write_json(RUN / "completion_verification.json", {
            "schema_version": 1, "status": "passed", "global_step": 1000,
            "new_prompt_count": 1000, "new_trajectory_count": len(events),
            "rollouts_per_prompt": 4, "model_snapshot_count": len(SAVE_STEPS),
            "first_restarted_lr": new_rows[0]["learning_rate"],
            "final_learning_rate": new_rows[-1]["learning_rate"],
        })
        state.update(status="completed", stage="complete", finished_at=now(), global_step=1000)
        write_json(STATE, state)
    except BaseException:
        state.update(status="failed", finished_at=now())
        write_json(STATE, state)
        raise


if __name__ == "__main__":
    main()
