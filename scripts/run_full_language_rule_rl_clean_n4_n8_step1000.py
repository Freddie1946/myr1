#!/usr/bin/env python3
"""Queue clean n=4 and n=8 full-language GRPO runs after n=4 continuation."""

from __future__ import annotations

import collections
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/home/dataset-assist-0/czy/wjy/myr1")
PYTHON = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/grpo/bin/python")
RUNS = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs")
PARENT = RUNS / "formal_selected_rule_rl1000_20260811/parents/sft_step080_merged"
PREREQUISITE = RUNS / "full_language_rule_rl_n4_epoch2_resume_20260812/sequence_state.json"
DATASET_YAML = RUNS.parent / "data/pathmmu_image_disjoint_v2/grpo/pathvlm_rl_n1000.yaml"
DATASET_JSON = DATASET_YAML.with_suffix(".json")
DEEPSPEED = REPO / "configs/deepspeed/ds_z3_gpu_torch_adamw.json"
QUEUE_ROOT = RUNS / "full_language_rule_rl_clean_n4_n8_step1000_20260812"
QUEUE_STATE = QUEUE_ROOT / "sequence_state.json"
SAVE_STEPS = (200, 400, 600, 800, 1000)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def gpu_processes() -> str:
    return subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def wait_for_prerequisite(state: dict) -> None:
    while True:
        if PREREQUISITE.is_file():
            prerequisite = json.loads(PREREQUISITE.read_text())
            status = prerequisite.get("status")
            if status == "completed" and prerequisite.get("global_step") == 1000:
                return
            if status == "failed":
                raise RuntimeError("n4 continuation failed; clean comparison queue not started")
        state.update(stage="waiting_for_n4_continuation", updated_at=now())
        write_json(QUEUE_STATE, state)
        time.sleep(60)


def environment(run: Path, label: str) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("PATHVLM_ALLOW_LANGUAGE_LORA", "PATHVLM_RESUME_FROM_CHECKPOINT"):
        env.pop(key, None)
    saves = ",".join(map(str, SAVE_STEPS))
    env.update({
        "PATHVLM_SKIP_FINAL_MODEL_SAVE": "true",
        "PATHVLM_EXPLICIT_SAVE_STEPS": saves,
        "PATHVLM_STOP_AFTER_SAVED_STEP": "1000",
        "PATHVLM_EPOCH_SNAPSHOT_STEPS": saves,
        "PATHVLM_EPOCH_SNAPSHOT_DIR": str(run / "model_snapshots"),
        "PATHVLM_SNAPSHOT_STEP_OFFSET": "0",
        "PATHVLM_REWARD_LOG_DIR": str(run / "online_reward_events"),
        "PATHVLM_TRAINING_SEGMENT": label,
        "PATHVLM_TRAIN_STATE_AUDIT_NAME": "pathvlm_train_state_audit.json",
        "PATHVLM_REQUIRE_SOURCE_AUDIT": "true",
        "PATHVLM_IMAGE_HASH_MANIFEST": str(
            REPO / "data/pathmmu_image_disjoint_v2/image_content_sha256.json"
        ),
        "PATHVLM_GENERATION_TEMPERATURE": "0.9",
        "PATHVLM_GENERATION_TOP_P": "1.0",
        "PATHVLM_GENERATION_TOP_K": "0",
        "PATHVLM_GENERATION_TYPICAL_P": "1.0",
        "PATHVLM_GENERATION_REPETITION_PENALTY": "1.0",
        "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false",
        "DEBUG_MODE": "false", "OMP_NUM_THREADS": "8",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    return env


def command(run: Path, n: int, port: int) -> list[str]:
    per_device = 1 if n == 4 else 2
    return [
        str(PYTHON), "-m", "torch.distributed.run", "--nproc_per_node=8",
        f"--master_port={port}", str(REPO / "scripts/grpo_pathmmu.py"),
        "--deepspeed", str(DEEPSPEED), "--output_dir", str(run / "output"),
        "--model_name_or_path", str(PARENT), "--dataset_name", str(DATASET_YAML),
        "--image_root", "/", "--reward_funcs", "accuracy",
        "--freeze_vision_modules", "true", "--max_pixels", "65536",
        "--min_pixels", "3136", "--num_generations", str(n),
        "--max_completion_length", "384", "--per_device_train_batch_size",
        str(per_device), "--gradient_accumulation_steps", "1",
        "--learning_rate", "1.0e-6", "--logging_steps", "1", "--bf16", "true",
        "--torch_dtype", "bfloat16", "--gradient_checkpointing", "true",
        "--attn_implementation", "sdpa", "--beta", "0.04",
        "--num_iterations", "1", "--temperature", "0.9",
        "--save_strategy", "no", "--save_steps", "200", "--save_total_limit", "1",
        "--save_only_model", "false", "--report_to", "none", "--seed", "42",
        "--data_seed", "42", "--remove_unused_columns", "false", "--max_steps", "1000",
    ]


def verify(run: Path, n: int) -> None:
    audit = json.loads((run / "output/pathvlm_train_state_audit.json").read_text())
    params = audit["trainability"]["parameters"]
    history = [row for row in audit["log_history"] if "grad_norm" in row]
    if not (
        audit["global_step"] == 1000 and len(history) == 1000
        and params["language"]["trainable"] == params["language"]["total"]
        and params["visual"]["trainable"] == 0
        and params["multimodal_projector"]["trainable"] == 0
        and all(math.isfinite(float(row["grad_norm"])) for row in history)
        and any(float(row["grad_norm"]) > 0 for row in history)
    ):
        raise RuntimeError(f"n={n} trainability audit failed")
    events = []
    for path in (run / "online_reward_events").glob("rank_*.jsonl"):
        events.extend(json.loads(line) for line in path.open() if line.strip())
    counts = collections.Counter(int(row["record_index"]) for row in events)
    expected_per_prompt = 2 * n
    if set(counts) != set(range(1000)) or set(counts.values()) != {expected_per_prompt}:
        raise RuntimeError(f"n={n} two-epoch prompt coverage failed")
    for step in SAVE_STEPS:
        snapshot = run / f"model_snapshots/checkpoint-{step}"
        if not (snapshot / "model.safetensors.index.json").is_file():
            raise RuntimeError(f"n={n} snapshot {step} missing")
    write_json(run / "completion_verification.json", {
        "schema_version": 1, "status": "passed", "global_step": 1000,
        "rollout_n": n, "prompt_exposure": 2000,
        "trajectory_exposure": 2000 * n, "save_steps": list(SAVE_STEPS),
        "full_checkpoint_save_total_limit": 1,
    })


def run_arm(n: int, state: dict) -> None:
    run = QUEUE_ROOT / f"n{n}_fresh_step1000"
    if run.exists():
        raise FileExistsError(run)
    run.mkdir(parents=True)
    contract = {
        "schema_version": 1, "parent": str(PARENT.resolve()), "rollout_n": n,
        "optimizer_steps": 1000, "prompts_per_update": 2,
        "prompt_exposure": 2000, "trajectory_exposure": 2000 * n,
        "learning_rate": 1e-6, "scheduler": "linear_to_zero_1000",
        "reward": "accuracy_only", "beta": 0.04, "num_iterations": 1,
        "temperature": 0.9, "top_p": 1.0, "top_k": 0,
        "max_completion_length": 384, "save_steps": list(SAVE_STEPS),
        "save_total_limit": 1, "vision_encoder": "frozen",
        "projector": "frozen", "language_model": "full_parameter",
        "test_used_for_training_or_selection": False,
    }
    write_json(run / "launch_contract.json", contract)
    state.update(stage=f"n{n}_fresh_step1000", updated_at=now())
    write_json(QUEUE_STATE, state)
    with (run / "train.log").open("w") as handle:
        subprocess.run(command(run, n, 34250 + n), cwd=REPO,
                       env=environment(run, f"full_language_rule_rl_n{n}_fresh_step1000"),
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    subprocess.run([
        str(PYTHON), str(REPO / "scripts/verify_grpo_reward_alignment.py"),
        "--events-dir", str(run / "online_reward_events"),
        "--dataset-json", str(DATASET_JSON), "--expected-steps", "1000",
        "--world-size", "8", "--per-device-batch", "1" if n == 4 else "2",
        "--num-generations", str(n), "--reward-types", "accuracy",
        "--format-reward-weight", "0.1",
        "--output", str(run / "reward_alignment_verification.json"),
    ], cwd=REPO, check=True)
    verify(run, n)


def main() -> None:
    if QUEUE_STATE.exists():
        raise FileExistsError(QUEUE_STATE)
    for path in (PARENT / "model.safetensors.index.json", PARENT / "merge_manifest.json",
                 DATASET_YAML, DATASET_JSON, DEEPSPEED):
        if not path.is_file():
            raise FileNotFoundError(path)
    state = {"schema_version": 1, "status": "running", "stage": "queued",
             "started_at": now(), "prerequisite": str(PREREQUISITE),
             "arms": ["n4_fresh_step1000", "n8_fresh_step1000"]}
    write_json(QUEUE_STATE, state)
    try:
        wait_for_prerequisite(state)
        if gpu_processes():
            raise RuntimeError("GPU process active after prerequisite completion")
        run_arm(4, state)
        if gpu_processes():
            raise RuntimeError("GPU process remained after n=4 arm")
        run_arm(8, state)
        state.update(status="completed", stage="complete", finished_at=now())
        write_json(QUEUE_STATE, state)
    except BaseException as error:
        state.update(status="failed", finished_at=now(), error=repr(error))
        write_json(QUEUE_STATE, state)
        raise


if __name__ == "__main__":
    main()
