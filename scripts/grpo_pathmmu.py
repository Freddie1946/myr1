#!/usr/bin/env python3
"""Audited PathMMU wrapper around the online Qwen2.5-VL GRPO trainer."""

from __future__ import annotations

import json
import os
from pathlib import Path

from grpo_pathmmu_audit import aligned_solutions, append_audit_events, trainability_report
from open_r1 import grpo_rec
from open_r1.trainer import Qwen2VLGRPOTrainer
from pathmmu_rewards import accuracy_reward, format_reward
from trl import TrlParser, get_peft_config


def audited_accuracy_reward(completions, solution, **kwargs):
    rewards = accuracy_reward(completions, solution, **kwargs)
    solutions = aligned_solutions(solution, len(completions))
    append_audit_events("accuracy", list(completions), solutions, rewards)
    return rewards


def audited_format_reward(completions, solution=None, **kwargs):
    rewards = format_reward(completions, **kwargs)
    solutions = aligned_solutions(solution, len(completions))
    append_audit_events("format", list(completions), solutions, rewards)
    return rewards


def main() -> None:
    grpo_rec.reward_funcs_registry["accuracy"] = audited_accuracy_reward
    grpo_rec.reward_funcs_registry["format"] = audited_format_reward
    parser = TrlParser((grpo_rec.GRPOScriptArguments, grpo_rec.GRPOConfig, grpo_rec.GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    reward_funcs = [grpo_rec.reward_funcs_registry[name] for name in script_args.reward_funcs]
    dataset = grpo_rec.LazySupervisedDataset(script_args.dataset_name, script_args)
    peft_config = get_peft_config(model_args)
    if peft_config is not None:
        raise RuntimeError("formal PathMMU GRPO gate forbids PEFT/LoRA")
    trainer = Qwen2VLGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=None,
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
    )
    report = trainability_report(trainer.model)
    if int(os.getenv("RANK", "0")) == 0:
        print("PATHVLM_TRAINABILITY_JSON=" + json.dumps(report, sort_keys=True), flush=True)
    if not report["passed"]:
        raise RuntimeError("formal trainability/freeze gate failed: " + json.dumps(report))

    train_result = trainer.train()
    trainer.save_model(training_args.output_dir)
    if trainer.is_world_process_zero():
        metrics = dict(train_result.metrics)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        audit = {
            "trainability": report,
            "train_metrics": metrics,
            "log_history": trainer.state.log_history,
            "global_step": trainer.state.global_step,
        }
        audit_path = Path(training_args.output_dir) / "pathvlm_train_state_audit.json"
        audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        print("PATHVLM_TRAIN_RESULT_JSON=" + json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
