#!/usr/bin/env python3
"""Resume the formal ratio queue with a smoke-selected system-only tail policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_data_ratio_ablation_sequence import (
    RL_MASTER_PORT,
    Runner,
    SequenceStop,
    Task,
    clean_training_env,
    gate_rule_rl,
    latest_checkpoint,
    model_files,
    run_logged,
    sha256_file,
)


class OptimizedTailRunner(Runner):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.performance_report_path = args.performance_report.resolve()
        self.performance_report = json.loads(
            self.performance_report_path.read_text(encoding="utf-8")
        )
        if self.performance_report.get("status") != "completed":
            raise SequenceStop("throughput smoke report is not complete")
        if self.performance_report.get("test_accessed") is not False:
            raise SequenceStop("throughput smoke touched test")

    def system_contract(self, task: Task) -> dict[str, Any]:
        if task.task_id == "stage2_continue_rule_rl1000":
            config = self.repo / "configs/deepspeed/ds_z3_optimizer_offload_torch_adamw.json"
            return {
                "policy": "matched_stage3_training_stack_checkpoint_only_optimization",
                "deepspeed": str(config),
                "deepspeed_sha256": sha256_file(config),
                "gradient_checkpointing": True,
                "attn_implementation": "sdpa",
                "save_steps": 250,
                "logging_steps": 5,
            }
        selected = dict(self.performance_report["selected"])
        config = Path(selected["deepspeed"]).resolve()
        if sha256_file(config) != selected["deepspeed_sha256"]:
            raise SequenceStop("selected DeepSpeed config changed after smoke")
        return {
            "policy": "smoke_selected_system_only_optimization",
            "smoke_report": str(self.performance_report_path),
            "smoke_report_sha256": sha256_file(self.performance_report_path),
            "selection": selected["name"],
            "fallback": bool(selected["fallback"]),
            "deepspeed": str(config),
            "deepspeed_sha256": selected["deepspeed_sha256"],
            "gradient_checkpointing": bool(selected["gradient_checkpointing"]),
            "attn_implementation": str(selected["attn_implementation"]),
            "save_steps": 50 if task.gate_only else 500,
            "logging_steps": 5,
        }

    def run_rule_rl(self, task: Task, task_dir: Path, attempt: int) -> dict[str, Any]:
        if task.task_id == "sft0250_rl0750_rule_rl":
            return super().run_rule_rl(task, task_dir, attempt)
        if task.task_id not in {
            "stage2_continue_rule_rl1000", "base_rule_rl4000_gate50", "base_rule_rl4000"
        }:
            return super().run_rule_rl(task, task_dir, attempt)
        parent = self.resolve_parent(task)
        output = task_dir / "output"
        alias = task_dir / "parent_Qwen2.5-VL-7B-Instruct"
        if alias.is_symlink():
            if alias.resolve() != parent.resolve():
                raise SequenceStop(f"parent alias changed for {task.task_id}")
        elif alias.exists():
            raise SequenceStop(f"parent alias is not a symlink: {alias}")
        else:
            alias.parent.mkdir(parents=True, exist_ok=True)
            alias.symlink_to(parent, target_is_directory=True)
        dataset_yaml = (self.data_root / str(task.dataset_yaml)).resolve()
        if not dataset_yaml.is_file() or "test" in str(dataset_yaml).lower():
            raise SequenceStop(f"invalid rule-RL dataset: {dataset_yaml}")
        completed_audit = output / "pathvlm_ratio_train_state_audit.json"
        if completed_audit.is_file():
            audit = json.loads(completed_audit.read_text(encoding="utf-8"))
            if int(audit.get("global_step", -1)) == task.max_steps:
                for previous_reward_dir in sorted(task_dir.glob("online_reward_events_attempt*"), reverse=True):
                    try:
                        result = gate_rule_rl(task, task_dir, output, previous_reward_dir)
                    except SequenceStop:
                        continue
                    result.update(
                        {
                            "output": str(output), "parent": str(parent), "resumed_from": None,
                            "dataset_yaml": str(dataset_yaml),
                            "dataset_yaml_sha256": sha256_file(dataset_yaml),
                            "recovered_completed_output_without_retraining": True,
                            "reward_audit_reused": str(previous_reward_dir),
                        }
                    )
                    return result
        resume = latest_checkpoint(output)
        system = self.system_contract(task)
        contract_path = task_dir / f"runtime_system_contract_attempt{attempt:02d}.json"
        contract_path.write_text(json.dumps(system, indent=2) + "\n", encoding="utf-8")
        command = [
            str(self.grpo_python), "-m", "torch.distributed.run",
            "--nproc_per_node=8", f"--master_port={RL_MASTER_PORT}",
            str(self.repo / "scripts/grpo_pathmmu.py"), "--deepspeed", system["deepspeed"],
            "--output_dir", str(output), "--model_name_or_path", str(alias),
            "--dataset_name", str(dataset_yaml), "--image_root", "/",
            "--reward_funcs", "accuracy", "format", "--freeze_vision_modules", "true",
            "--max_pixels", "65536", "--min_pixels", "3136", "--num_generations", "4",
            "--max_completion_length", "192", "--per_device_train_batch_size", "1",
            "--gradient_accumulation_steps", "1", "--learning_rate", "1.0e-6",
            "--logging_steps", str(system["logging_steps"]), "--bf16", "true",
            "--torch_dtype", "bfloat16", "--gradient_checkpointing",
            str(system["gradient_checkpointing"]).lower(), "--attn_implementation",
            system["attn_implementation"], "--beta", "0.04", "--num_iterations", "1",
            "--save_strategy", "steps", "--save_steps", str(system["save_steps"]),
            "--save_total_limit", "2", "--save_only_model", "false", "--report_to", "none",
            "--seed", "42", "--data_seed", "42", "--remove_unused_columns", "false",
            "--max_steps", str(task.max_steps),
        ]
        reward_dir = task_dir / f"online_reward_events_attempt{attempt:02d}"
        env_extra = {
            "PATHVLM_REWARD_LOG_DIR": str(reward_dir),
            "PATHVLM_TRAINING_SEGMENT": task.task_id,
            "PATHVLM_TRAIN_STATE_AUDIT_NAME": "pathvlm_ratio_train_state_audit.json",
            "PATHVLM_REQUIRE_SOURCE_AUDIT": "true",
            "PATHVLM_IMAGE_HASH_MANIFEST": str(
                self.repo / "data/pathmmu_image_disjoint_v2/image_content_sha256.json"
            ),
            "DEBUG_MODE": "false",
        }
        if resume is not None:
            env_extra["PATHVLM_RESUME_FROM_CHECKPOINT"] = str(resume)
        rc = run_logged(
            command, cwd=self.repo / "vendor/open-r1-multimodal",
            env=clean_training_env(env_extra), log=task_dir / f"train_attempt{attempt:02d}.log",
        )
        if rc:
            raise SequenceStop(f"rule-RL task {task.task_id} failed with exit {rc}")
        result = gate_rule_rl(task, task_dir, output, reward_dir)
        result.update(
            {
                "output": str(output), "parent": str(parent),
                "resumed_from": str(resume) if resume else None,
                "dataset_yaml": str(dataset_yaml),
                "dataset_yaml_sha256": sha256_file(dataset_yaml),
                "runtime_system_contract": str(contract_path),
                "runtime_system_contract_sha256": sha256_file(contract_path),
            }
        )
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("formal",), required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--install-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--performance-report", type=Path, required=True)
    args = parser.parse_args()
    OptimizedTailRunner(args).run(preflight_only=False)


if __name__ == "__main__":
    main()
