#!/usr/bin/env python3
"""Smoke and run the matched full-language GRPO n=8 arm to step 500."""

from __future__ import annotations

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
    "formal_selected_rule_rl1000_20260811/parents/sft_step080_merged"
)
DATASET_YAML = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/"
    "pathmmu_image_disjoint_v2/grpo/pathvlm_rl_n1000.yaml"
)
DATASET_JSON = DATASET_YAML.with_suffix(".json")
DEEPSPEED = REPO / "configs/deepspeed/ds_z3_gpu_torch_adamw.json"
RUN = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/"
    "full_language_rule_rl_n8_capacity_20260812"
)
EVAL = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/"
    "full_language_rule_rl_n8_capacity_20260812/step500_gate"
)
STATE = RUN / "sequence_state.json"


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


def training_command(
    output: Path, per_device: int, grad_accum: int, max_steps: int,
    master_port: int,
) -> list[str]:
    return [
        str(PYTHON), "-m", "torch.distributed.run", "--nproc_per_node=8",
        f"--master_port={master_port}", str(REPO / "scripts/grpo_pathmmu.py"),
        "--deepspeed", str(DEEPSPEED), "--output_dir", str(output),
        "--model_name_or_path", str(PARENT), "--dataset_name", str(DATASET_YAML),
        "--image_root", "/", "--reward_funcs", "accuracy",
        "--freeze_vision_modules", "true", "--max_pixels", "65536",
        "--min_pixels", "3136", "--num_generations", "8",
        "--max_completion_length", "384", "--per_device_train_batch_size",
        str(per_device), "--gradient_accumulation_steps", str(grad_accum),
        "--learning_rate", "1.0e-6", "--logging_steps", "1", "--bf16", "true",
        "--torch_dtype", "bfloat16", "--gradient_checkpointing", "true",
        "--attn_implementation", "sdpa", "--beta", "0.04",
        "--num_iterations", "1", "--temperature", "0.9", "--save_strategy", "no",
        "--save_steps", "100", "--save_total_limit", "1", "--save_only_model", "false",
        "--report_to", "none", "--seed", "42", "--data_seed", "42",
        "--remove_unused_columns", "false", "--max_steps", str(max_steps),
    ]


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "PATHVLM_ALLOW_LANGUAGE_LORA", "PATHVLM_RESUME_FROM_CHECKPOINT",
        "PATHVLM_EXPLICIT_SAVE_STEPS", "PATHVLM_STOP_AFTER_SAVED_STEP",
        "PATHVLM_EPOCH_SNAPSHOT_STEPS", "PATHVLM_EPOCH_SNAPSHOT_DIR",
    ):
        env.pop(key, None)
    env.update({
        "PATHVLM_SKIP_FINAL_MODEL_SAVE": "true",
        "PATHVLM_REQUIRE_SOURCE_AUDIT": "true",
        "PATHVLM_IMAGE_HASH_MANIFEST": str(
            REPO / "data/pathmmu_image_disjoint_v2/image_content_sha256.json"
        ),
        "PATHVLM_GENERATION_TEMPERATURE": "0.9",
        "PATHVLM_GENERATION_TOP_P": "1.0", "PATHVLM_GENERATION_TOP_K": "0",
        "PATHVLM_GENERATION_TYPICAL_P": "1.0",
        "PATHVLM_GENERATION_REPETITION_PENALTY": "1.0",
        "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false",
        "DEBUG_MODE": "false", "OMP_NUM_THREADS": "8",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    return env


def verify_trainability(audit_path: Path, expected_step: int) -> dict:
    audit = json.loads(audit_path.read_text())
    trainability = audit["trainability"]
    params = trainability["parameters"]
    history = [row for row in audit["log_history"] if "grad_norm" in row]
    if not (
        audit["global_step"] == expected_step and trainability["passed"]
        and trainability["language_mode"] == "full"
        and params["language"]["trainable"] == params["language"]["total"]
        and params["visual"]["trainable"] == 0
        and params["multimodal_projector"]["trainable"] == 0
        and len(history) == expected_step
        and all(math.isfinite(float(row["grad_norm"])) and float(row["grad_norm"]) > 0
                for row in history)
    ):
        raise RuntimeError("n8 trainability/gradient verification failed")
    return audit


def run_smoke(per_device: int, grad_accum: int, label: str, port: int) -> bool:
    root = RUN / f"smoke_{label}"
    root.mkdir(parents=True)
    env = base_env()
    env.update({
        "PATHVLM_REWARD_LOG_DIR": str(root / "online_reward_events"),
        "PATHVLM_TRAINING_SEGMENT": f"full_language_rule_rl_n8_smoke_{label}",
        "PATHVLM_TRAIN_STATE_AUDIT_NAME": "pathvlm_train_state_audit.json",
    })
    with (root / "train.log").open("w") as handle:
        code = subprocess.run(
            training_command(root / "output", per_device, grad_accum, 2, port),
            cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT,
        ).returncode
    if code != 0:
        return False
    audit = root / "output/pathvlm_train_state_audit.json"
    verify_trainability(audit, 2)
    generation_batches = 2 * grad_accum
    subprocess.run([
        str(PYTHON), str(REPO / "scripts/verify_grpo_reward_alignment.py"),
        "--events-dir", str(root / "online_reward_events"),
        "--dataset-json", str(DATASET_JSON), "--expected-steps", str(generation_batches),
        "--world-size", "8", "--per-device-batch", str(per_device),
        "--num-generations", "8", "--reward-types", "accuracy",
        "--format-reward-weight", "0.1", "--output",
        str(root / "reward_alignment_verification.json"),
    ], cwd=REPO, check=True)
    return True


def verify_coverage(events_dir: Path) -> None:
    events = []
    for path in events_dir.glob("rank_*.jsonl"):
        events.extend(json.loads(line) for line in path.open() if line.strip())
    indices = [int(event["record_index"]) for event in events]
    counts = {index: indices.count(index) for index in set(indices)}
    if set(counts) != set(range(1000)) or set(counts.values()) != {8}:
        raise RuntimeError("n8 step500 does not cover each RL1000 prompt exactly eight times")
    write_json(RUN / "full_epoch_prompt_coverage_verification.json", {
        "schema_version": 1, "status": "passed", "unique_prompt_count": 1000,
        "rollouts_per_prompt": 8, "event_count": len(events),
        "record_index_min": 0, "record_index_max": 999,
    })


def main() -> None:
    if STATE.exists() or EVAL.exists():
        raise FileExistsError("n8 run state or evaluation already exists")
    for path in (PARENT / "model.safetensors.index.json", PARENT / "merge_manifest.json",
                 DATASET_YAML, DATASET_JSON, DEEPSPEED):
        if not path.is_file():
            raise FileNotFoundError(path)
    if gpu_processes():
        raise RuntimeError("GPU process is active before n8 smoke")
    state = {
        "schema_version": 1, "status": "running", "stage": "smoke_mbs2_gacc1",
        "started_at": now(), "automatic_retry": False,
        "comparison_parent": str(PARENT), "rollout_n": 8,
    }
    write_json(STATE, state)
    try:
        if run_smoke(2, 1, "mbs2_gacc1", 34223):
            per_device, grad_accum = 2, 1
        else:
            if gpu_processes():
                raise RuntimeError("failed n8 smoke left GPU processes active")
            state["stage"] = "smoke_fallback_mbs1_gacc2"
            write_json(STATE, state)
            if not run_smoke(1, 2, "mbs1_gacc2", 34224):
                raise RuntimeError("both predeclared n8 memory configurations failed smoke")
            per_device, grad_accum = 1, 2
        formal = RUN / "formal"
        formal.mkdir()
        output = formal / "output"
        snapshots = RUN / "model_snapshots"
        events = formal / "online_reward_events"
        env = base_env()
        env.update({
            "PATHVLM_EXPLICIT_SAVE_STEPS": "500",
            "PATHVLM_STOP_AFTER_SAVED_STEP": "500",
            "PATHVLM_EPOCH_SNAPSHOT_STEPS": "500",
            "PATHVLM_EPOCH_SNAPSHOT_DIR": str(snapshots),
            "PATHVLM_REWARD_LOG_DIR": str(events),
            "PATHVLM_TRAINING_SEGMENT": "full_language_rule_rl_n8_formal_step500",
            "PATHVLM_TRAIN_STATE_AUDIT_NAME": "pathvlm_train_state_audit.json",
        })
        contract = {
            "schema_version": 1, "parent": str(PARENT.resolve()), "rollout_n": 8,
            "optimizer_steps": 500, "prompts_per_update": 2,
            "prompt_exposure": 1000, "trajectory_exposure": 8000,
            "per_device_train_batch_size": per_device,
            "gradient_accumulation_steps": grad_accum,
            "world_size": 8, "learning_rate": 1e-6, "scheduler": "linear_to_zero_500",
            "reward": "accuracy_only", "beta": 0.04, "num_iterations": 1,
            "temperature": 0.9, "top_p": 1.0, "top_k": 0,
            "max_completion_length": 384, "vision_encoder": "frozen",
            "projector": "frozen", "language_model": "full_parameter",
            "test_used_for_training_or_selection": False,
        }
        write_json(formal / "launch_contract.json", contract)
        state.update(stage="formal_training", selected_microbatch=per_device,
                     selected_gradient_accumulation=grad_accum)
        write_json(STATE, state)
        with (formal / "train.log").open("w") as handle:
            subprocess.run(
                training_command(output, per_device, grad_accum, 500, 34225),
                cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True,
            )
        audit = verify_trainability(output / "pathvlm_train_state_audit.json", 500)
        generation_batches = 500 * grad_accum
        subprocess.run([
            str(PYTHON), str(REPO / "scripts/verify_grpo_reward_alignment.py"),
            "--events-dir", str(events), "--dataset-json", str(DATASET_JSON),
            "--expected-steps", str(generation_batches), "--world-size", "8",
            "--per-device-batch", str(per_device), "--num-generations", "8",
            "--reward-types", "accuracy", "--format-reward-weight", "0.1",
            "--output", str(formal / "reward_alignment_verification.json"),
        ], cwd=REPO, check=True)
        verify_coverage(events)
        snapshot = snapshots / "checkpoint-500"
        if not (snapshot / "model.safetensors.index.json").is_file():
            raise RuntimeError("n8 step500 snapshot is missing")
        state["stage"] = "evaluation"
        write_json(STATE, state)
        subprocess.run([
            str(PYTHON), str(REPO / "scripts/run_full_language_rule_rl_step100_evaluation.py"),
            "--model", str(snapshot), "--output", str(EVAL),
        ], cwd=REPO, check=True)
        state.update(status="completed", stage="complete", finished_at=now(),
                     global_step=int(audit["global_step"]))
        write_json(STATE, state)
    except BaseException:
        state.update(status="failed", finished_at=now())
        write_json(STATE, state)
        raise


if __name__ == "__main__":
    main()
