#!/usr/bin/env python3
"""Benchmark safe system-only rule-RL variants before the long 0+4000 arm."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from run_data_ratio_ablation_sequence import (
    Task,
    atomic_json,
    clean_training_env,
    gpu_preflight,
    require_port_free,
    reward_audit_summary,
    run_logged,
    sha256_file,
)


RL_PORT = 29821
BASELINE_STEPS_PER_SECOND = 0.038


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def wait_for_idle() -> None:
    last_error: Exception | None = None
    for _ in range(60):
        try:
            gpu_preflight()
            require_port_free(RL_PORT)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"GPUs/port did not become idle after candidate exit: {last_error}")


def candidate_command(
    *, repo: Path, install: Path, output: Path, deepspeed: Path,
    gradient_checkpointing: bool, max_steps: int,
) -> list[str]:
    return [
        str(install / "envs/grpo/bin/python"), "-m", "torch.distributed.run",
        "--nproc_per_node=8", f"--master_port={RL_PORT}",
        str(repo / "scripts/grpo_pathmmu.py"),
        "--deepspeed", str(deepspeed),
        "--output_dir", str(output),
        "--model_name_or_path", str(
            install / "models/Qwen2.5-VL-7B-Instruct-cc594898137f460bfe9f0759e9844b3ce807cfb5"
        ),
        "--dataset_name", str(
            install / "data/data_ratio_rule_rl_ablation_v1/grpo/pathvlm_ratio_base_rule_rl4000.yaml"
        ),
        "--image_root", "/", "--reward_funcs", "accuracy", "format",
        "--freeze_vision_modules", "true", "--max_pixels", "65536",
        "--min_pixels", "3136", "--num_generations", "4",
        "--max_completion_length", "192", "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "1", "--learning_rate", "1.0e-6",
        "--logging_steps", "1", "--bf16", "true", "--torch_dtype", "bfloat16",
        "--gradient_checkpointing", str(gradient_checkpointing).lower(),
        "--attn_implementation", "sdpa", "--beta", "0.04", "--num_iterations", "1",
        "--save_strategy", "no", "--report_to", "none", "--seed", "42",
        "--data_seed", "42", "--remove_unused_columns", "false",
        "--max_steps", str(max_steps),
    ]


def validate(output: Path, rewards: Path, expected_steps: int) -> dict[str, Any]:
    audit_path = output / "throughput_smoke_train_state_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if int(audit.get("global_step", -1)) != expected_steps:
        raise RuntimeError(f"smoke reached {audit.get('global_step')}, expected {expected_steps}")
    if audit.get("trainability", {}).get("passed") is not True:
        raise RuntimeError("trainability gate failed")
    metrics = audit.get("train_metrics", {})
    numeric = [float(value) for value in metrics.values() if isinstance(value, (int, float))]
    if not numeric or not all(math.isfinite(value) for value in numeric):
        raise RuntimeError("smoke metrics are missing or non-finite")
    reward_summary = reward_audit_summary(rewards)
    steps_per_second = float(metrics.get("train_steps_per_second", 0.0))
    if steps_per_second <= 0:
        raise RuntimeError("smoke has no positive train_steps_per_second")
    return {
        "audit": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "train_metrics": metrics,
        "reward_audit": reward_summary,
        "steps_per_second": steps_per_second,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--install-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    install = args.install_root.resolve()
    report_path = args.report.resolve()
    smoke_root = install / "runs/data_ratio_rule_rl_ablation_v1/throughput_smoke"
    smoke_root.mkdir(parents=True, exist_ok=True)
    task = Task(
        "base_rule_rl4000_throughput_smoke", "rule_rl",
        dataset_yaml="grpo/pathvlm_ratio_base_rule_rl4000.yaml",
        parent_kind="base", sample_count=4000, max_steps=args.max_steps,
    )
    candidates = [
        {
            "name": "z3_gpu_no_gradient_checkpointing_sdpa",
            "deepspeed": repo / "configs/deepspeed/ds_z3_gpu_torch_adamw.json",
            "gradient_checkpointing": False,
        },
        {
            "name": "z3_gpu_gradient_checkpointing_sdpa",
            "deepspeed": repo / "configs/deepspeed/ds_z3_gpu_torch_adamw.json",
            "gradient_checkpointing": True,
        },
        {
            "name": "z3_optimizer_offload_no_gradient_checkpointing_sdpa",
            "deepspeed": repo / "configs/deepspeed/ds_z3_optimizer_offload_torch_adamw.json",
            "gradient_checkpointing": False,
        },
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at": now_iso(),
        "formal_result": False,
        "test_accessed": False,
        "task": asdict(task),
        "baseline_steps_per_second": BASELINE_STEPS_PER_SECOND,
        "selection_threshold_steps_per_second": BASELINE_STEPS_PER_SECOND * 1.03,
        "candidates": [],
    }
    atomic_json(report_path, report)
    if args.baseline_only:
        baseline = repo / "configs/deepspeed/ds_z3_optimizer_offload_torch_adamw.json"
        report.update(
            {
                "status": "completed",
                "completed_at": now_iso(),
                "selection_reason": "explicit fail-closed baseline fallback after smoke infrastructure failure",
                "selected": {
                    "name": "baseline_system_fallback",
                    "deepspeed": str(baseline),
                    "deepspeed_sha256": sha256_file(baseline),
                    "gradient_checkpointing": True,
                    "attn_implementation": "sdpa",
                    "measured_steps_per_second": BASELINE_STEPS_PER_SECOND,
                    "fallback": True,
                },
            }
        )
        atomic_json(report_path, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    for index, candidate in enumerate(candidates, 1):
        wait_for_idle()
        root = smoke_root / f"attempt{index:02d}_{candidate['name']}"
        output = root / "output"
        rewards = root / "reward_events"
        command = candidate_command(
            repo=repo, install=install, output=output,
            deepspeed=candidate["deepspeed"],
            gradient_checkpointing=candidate["gradient_checkpointing"],
            max_steps=args.max_steps,
        )
        env = clean_training_env(
            {
                "PATHVLM_REWARD_LOG_DIR": str(rewards),
                "PATHVLM_TRAINING_SEGMENT": candidate["name"],
                "PATHVLM_TRAIN_STATE_AUDIT_NAME": "throughput_smoke_train_state_audit.json",
                "PATHVLM_REQUIRE_SOURCE_AUDIT": "true",
                "PATHVLM_IMAGE_HASH_MANIFEST": str(
                    repo / "data/pathmmu_image_disjoint_v2/image_content_sha256.json"
                ),
                "PATHVLM_SKIP_FINAL_MODEL_SAVE": "true",
                "DEBUG_MODE": "false",
            }
        )
        item: dict[str, Any] = {
            "name": candidate["name"],
            "deepspeed": str(candidate["deepspeed"]),
            "deepspeed_sha256": sha256_file(candidate["deepspeed"]),
            "gradient_checkpointing": candidate["gradient_checkpointing"],
            "attn_implementation": "sdpa",
            "started_at": now_iso(),
            "log": str(root / "train.log"),
        }
        try:
            rc = run_logged(command, cwd=repo / "vendor/open-r1-multimodal", env=env, log=root / "train.log")
            if rc:
                raise RuntimeError(f"candidate exited {rc}")
            item.update(validate(output, rewards, args.max_steps))
            item["status"] = "passed"
        except Exception as exc:
            item.update(
                {
                    "status": "failed",
                    "failure": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
        item["completed_at"] = now_iso()
        report["candidates"].append(item)
        atomic_json(report_path, report)
        time.sleep(5)

    eligible = [
        item for item in report["candidates"]
        if item.get("status") == "passed"
        and float(item["steps_per_second"]) >= report["selection_threshold_steps_per_second"]
    ]
    if eligible:
        best = max(eligible, key=lambda item: float(item["steps_per_second"]))
        selected = {
            "name": best["name"],
            "deepspeed": best["deepspeed"],
            "deepspeed_sha256": best["deepspeed_sha256"],
            "gradient_checkpointing": best["gradient_checkpointing"],
            "attn_implementation": best["attn_implementation"],
            "measured_steps_per_second": best["steps_per_second"],
            "fallback": False,
        }
    else:
        baseline = repo / "configs/deepspeed/ds_z3_optimizer_offload_torch_adamw.json"
        selected = {
            "name": "baseline_system_fallback",
            "deepspeed": str(baseline),
            "deepspeed_sha256": sha256_file(baseline),
            "gradient_checkpointing": True,
            "attn_implementation": "sdpa",
            "measured_steps_per_second": BASELINE_STEPS_PER_SECOND,
            "fallback": True,
        }
    report.update({"status": "completed", "completed_at": now_iso(), "selected": selected})
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
