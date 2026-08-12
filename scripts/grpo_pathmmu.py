#!/usr/bin/env python3
"""Audited PathMMU wrapper around the online Qwen2.5-VL GRPO trainer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import MethodType
from typing import Any

from grpo_pathmmu_audit import (
    STRICT_PROMPT_CONTRACT,
    aligned_solutions,
    append_audit_events,
    create_model_only_snapshot,
    replace_legacy_json_prompt,
    strict_prompt_text,
    trainability_report,
)
from open_r1 import grpo_rec
from open_r1.trainer import Qwen2VLGRPOTrainer
from pathmmu_rewards import accuracy_reward, format_reward
if os.getenv("PATHVLM_STAGE3_JUDGE_BACKEND", "openrouter") == "aigcbest":
    from stage3_aigcbest_judge import process_reward
else:
    from stage3_openrouter_judge import process_reward
from trl import TrlParser, get_peft_config
from transformers import TrainerCallback


SOURCE_AUDIT_FIELDS = (
    "record_index", "image_path", "image_sha256", "problem", "prompt_contract", "prompt"
)


class RelativeLinearScheduler:
    """Linear decay whose clock starts when a resumed segment starts.

    A normal Transformers scheduler can inherit the Trainer's cumulative
    ``global_step`` after DeepSpeed resume.  For an epoch-boundary scheduler
    restart that would immediately exhaust a short, newly-created schedule.
    This deliberately small scheduler owns a segment-local counter instead.
    """

    def __init__(self, optimizer, *, total_steps: int, peak_lr: float):
        if total_steps <= 0 or peak_lr <= 0:
            raise ValueError("relative linear scheduler requires positive values")
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.peak_lr = float(peak_lr)
        self.completed_steps = 0
        self._set_lr(self.peak_lr)

    def _set_lr(self, value: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(value)
            group["initial_lr"] = self.peak_lr
        self._last_lr = [float(value) for _ in self.optimizer.param_groups]

    def step(self, *args, **kwargs) -> None:
        del args, kwargs
        self.completed_steps += 1
        factor = max(0.0, 1.0 - self.completed_steps / self.total_steps)
        self._set_lr(self.peak_lr * factor)

    def get_last_lr(self) -> list[float]:
        return list(self._last_lr)

    def get_lr(self) -> list[float]:
        return self.get_last_lr()

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "peak_lr": self.peak_lr,
            "completed_steps": self.completed_steps,
            "_last_lr": self.get_last_lr(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.total_steps = int(state_dict["total_steps"])
        self.peak_lr = float(state_dict["peak_lr"])
        self.completed_steps = int(state_dict["completed_steps"])
        factor = max(0.0, 1.0 - self.completed_steps / self.total_steps)
        self._set_lr(self.peak_lr * factor)


class ReadOnlySchedulerProxy:
    """Expose an engine-owned scheduler to Trainer without stepping it twice."""

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def step(self, *args, **kwargs) -> None:
        del args, kwargs

    def get_last_lr(self) -> list[float]:
        return self.scheduler.get_last_lr()

    def get_lr(self) -> list[float]:
        return self.scheduler.get_lr()

    def state_dict(self) -> dict[str, Any]:
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.scheduler.load_state_dict(state_dict)


def aligned_source_metadata(kwargs: dict[str, Any], count: int) -> list[dict[str, Any]]:
    require_exact = os.getenv("PATHVLM_REQUIRE_SOURCE_AUDIT", "").lower() == "true"
    rows = [{} for _ in range(count)]
    for field in SOURCE_AUDIT_FIELDS:
        values = kwargs.get("prompts" if field == "prompt" else field)
        if values is None:
            if require_exact:
                raise RuntimeError(f"formal reward source metadata is missing: {field}")
            continue
        values = list(values)
        if require_exact and len(values) != count:
            raise RuntimeError(
                f"formal reward source metadata length mismatch: {field}: "
                f"{len(values)} != {count}"
            )
        aligned = values[:count]
        if len(aligned) != count:
            raise RuntimeError(f"formal reward source metadata length mismatch: {field}")
        for row, value in zip(rows, aligned):
            row[field] = value
    if require_exact:
        for index, row in enumerate(rows):
            try:
                prompt_text = str(row["prompt"][0]["content"][1]["text"])
                problem = str(row["problem"])
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(
                    f"formal reward prompt metadata is malformed at item {index}"
                ) from exc
            if prompt_text != strict_prompt_text(problem):
                raise RuntimeError(
                    f"formal reward prompt/problem alignment mismatch at item {index}"
                )
    return rows


def audited_accuracy_reward(completions, solution, **kwargs):
    require_exact = os.getenv("PATHVLM_REQUIRE_SOURCE_AUDIT", "").lower() == "true"
    solutions = aligned_solutions(solution, len(completions), require_exact=require_exact)
    rewards = accuracy_reward(completions, solutions, **kwargs)
    metadata = aligned_source_metadata(kwargs, len(completions))
    append_audit_events("accuracy", list(completions), solutions, rewards, metadata)
    return rewards


def audited_format_reward(completions, solution=None, **kwargs):
    require_exact = os.getenv("PATHVLM_REQUIRE_SOURCE_AUDIT", "").lower() == "true"
    rewards = format_reward(completions, **kwargs)
    solutions = aligned_solutions(solution, len(completions), require_exact=require_exact)
    metadata = aligned_source_metadata(kwargs, len(completions))
    append_audit_events("format", list(completions), solutions, rewards, metadata)
    return rewards


def audited_scaled_format_reward(completions, solution=None, **kwargs):
    """Apply the predeclared scalar to format reward while retaining raw auditability."""
    weight = float(os.getenv("PATHVLM_FORMAT_REWARD_WEIGHT", "0.1"))
    if not 0.0 <= weight <= 1.0:
        raise RuntimeError("PATHVLM_FORMAT_REWARD_WEIGHT must be in [0, 1]")
    require_exact = os.getenv("PATHVLM_REQUIRE_SOURCE_AUDIT", "").lower() == "true"
    raw_rewards = format_reward(completions, **kwargs)
    rewards = [weight * float(value) for value in raw_rewards]
    solutions = aligned_solutions(solution, len(completions), require_exact=require_exact)
    metadata = aligned_source_metadata(kwargs, len(completions))
    append_audit_events("format_scaled", list(completions), solutions, rewards, metadata)
    return rewards


def audited_process_reward(completions, solution, **kwargs):
    require_exact = os.getenv("PATHVLM_REQUIRE_SOURCE_AUDIT", "").lower() == "true"
    solutions = aligned_solutions(solution, len(completions), require_exact=require_exact)
    rewards = process_reward(completions, solutions, **kwargs)
    metadata = aligned_source_metadata(kwargs, len(completions))
    append_audit_events("process", list(completions), solutions, rewards, metadata)
    return rewards


class PathMMUStrictPromptDataset(grpo_rec.LazySupervisedDataset):
    """Replace the vendored grounding-era JSON instruction with the scored contract."""

    def __init__(self, data_path, script_args):
        super().__init__(data_path, script_args)
        manifest_path = os.getenv("PATHVLM_IMAGE_HASH_MANIFEST")
        if not manifest_path:
            raise RuntimeError("PATHVLM_IMAGE_HASH_MANIFEST is required")
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        images = payload.get("images")
        if not isinstance(images, dict) or not images:
            raise RuntimeError("image hash manifest has no images mapping")
        self.image_hashes = images

    def __getitem__(self, index):
        item = super().__getitem__(index)
        try:
            text_part = item["prompt"][0]["content"][1]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("unexpected PathMMU multimodal prompt structure") from exc
        original = str(text_part.get("text", ""))
        text_part["text"] = replace_legacy_json_prompt(original, item["problem"])
        source = self.list_data_dict[index]
        image_path = Path(source["image"]).resolve()
        image_sha256 = self.image_hashes.get(image_path.name)
        if not image_sha256:
            raise RuntimeError(f"image hash is not frozen for {image_path.name}")
        item["record_index"] = index
        item["image_path"] = str(image_path)
        item["image_sha256"] = image_sha256
        item["prompt_contract"] = STRICT_PROMPT_CONTRACT
        return item


class StopAfterSaveCallback(TrainerCallback):
    """Stop an engineering segment only after its requested checkpoint is saved."""

    def on_save(self, args, state, control, **kwargs):
        target = int(os.getenv("PATHVLM_STOP_AFTER_SAVED_STEP", "0"))
        if target and state.global_step == target:
            control.should_training_stop = True
        return control


class ExplicitSaveStepsCallback(TrainerCallback):
    """Request resumable checkpoints at an exact, non-uniform set of steps."""

    def on_step_end(self, args, state, control, **kwargs):
        targets_text = os.getenv("PATHVLM_EXPLICIT_SAVE_STEPS", "").strip()
        if not targets_text:
            return control
        targets = {int(value) for value in targets_text.split(",") if value.strip()}
        if state.global_step in targets:
            control.should_save = True
        return control


def freeze_generation_contract(trainer, training_args) -> dict[str, Any]:
    """Override vendored hard-coded sampling defaults and return the effective contract."""
    temperature = float(os.getenv("PATHVLM_GENERATION_TEMPERATURE", "0.9"))
    top_p = float(os.getenv("PATHVLM_GENERATION_TOP_P", "1.0"))
    top_k = int(os.getenv("PATHVLM_GENERATION_TOP_K", "0"))
    typical_p = float(os.getenv("PATHVLM_GENERATION_TYPICAL_P", "1.0"))
    repetition_penalty = float(os.getenv("PATHVLM_GENERATION_REPETITION_PENALTY", "1.0"))
    if not temperature > 0:
        raise RuntimeError("generation temperature must be positive")
    if not 0 < top_p <= 1 or not 0 < typical_p <= 1:
        raise RuntimeError("generation top_p and typical_p must be in (0, 1]")
    if top_k < 0 or repetition_penalty <= 0:
        raise RuntimeError("generation top_k/repetition_penalty is invalid")

    config = trainer.generation_config
    config.do_sample = True
    config.temperature = temperature
    config.top_p = top_p
    config.top_k = top_k
    config.typical_p = typical_p
    config.repetition_penalty = repetition_penalty
    config.max_new_tokens = int(training_args.max_completion_length)
    config.pad_token_id = trainer.processing_class.pad_token_id
    config.eos_token_id = trainer.processing_class.eos_token_id
    return {
        "do_sample": bool(config.do_sample),
        "temperature": float(config.temperature),
        "top_p": float(config.top_p),
        "top_k": int(config.top_k),
        "top_k_disabled": int(config.top_k) == 0,
        "typical_p": float(config.typical_p),
        "repetition_penalty": float(config.repetition_penalty),
        "max_new_tokens": int(config.max_new_tokens),
        "pad_token_id": config.pad_token_id,
        "eos_token_id": config.eos_token_id,
        "stop_contract": "first_eos_or_max_new_tokens",
    }


class EpochSnapshotCallback(TrainerCallback):
    """Archive predeclared model-only snapshots after full checkpoint saves."""

    def on_save(self, args, state, control, **kwargs):
        targets_text = os.getenv("PATHVLM_EPOCH_SNAPSHOT_STEPS", "").strip()
        root_text = os.getenv("PATHVLM_EPOCH_SNAPSHOT_DIR", "").strip()
        if not targets_text and not root_text:
            return control
        if not targets_text or not root_text:
            raise RuntimeError("both epoch snapshot environment variables are required")
        targets = {int(value) for value in targets_text.split(",")}
        if state.global_step not in targets or not state.is_world_process_zero:
            return control
        step_offset = int(os.getenv("PATHVLM_SNAPSHOT_STEP_OFFSET", "0"))
        if step_offset < 0:
            raise RuntimeError("PATHVLM_SNAPSHOT_STEP_OFFSET must be non-negative")
        cumulative_step = int(state.global_step) + step_offset
        destination = Path(root_text) / f"checkpoint-{cumulative_step}"
        create_model_only_snapshot(
            Path(args.output_dir) / f"checkpoint-{state.global_step}",
            destination,
            global_step=cumulative_step,
            epoch=float(state.epoch or 0.0),
        )
        return control


def install_scheduler_restart_on_resume(trainer) -> dict[str, Any] | None:
    """Preserve resumed optimizer moments while replacing an exhausted scheduler.

    DeepSpeed restores the optimizer before ``Trainer._load_optimizer_and_scheduler``
    returns.  We first allow the trusted local checkpoint to load normally, then
    replace only the scheduler and explicitly restore the requested peak LR.
    """
    steps_text = os.getenv("PATHVLM_RESET_SCHEDULER_ON_RESUME_STEPS", "").strip()
    if not steps_text:
        return None
    reset_steps = int(steps_text)
    peak_lr = float(os.getenv("PATHVLM_RESET_SCHEDULER_PEAK_LR", "0"))
    if reset_steps <= 0 or peak_lr <= 0:
        raise RuntimeError("scheduler restart requires positive steps and peak LR")
    original = trainer._load_optimizer_and_scheduler

    def load_optimizer_then_restart(self, checkpoint):
        if checkpoint is None:
            raise RuntimeError("scheduler restart is permitted only with a resume checkpoint")
        original(checkpoint)
        if self.optimizer is None or not getattr(self.optimizer, "param_groups", None):
            raise RuntimeError("resumed optimizer is unavailable for scheduler restart")
        restarted_scheduler = RelativeLinearScheduler(
            self.optimizer, total_steps=reset_steps, peak_lr=peak_lr
        )
        # DeepSpeed owns the actual optimizer step and advances the scheduler
        # attached to its engine.  Keep the Trainer and engine references exact.
        deepspeed_engine = getattr(self, "deepspeed", None)
        if deepspeed_engine is None:
            wrapped = getattr(self, "model_wrapped", None)
            if wrapped is not None and hasattr(wrapped, "lr_scheduler"):
                deepspeed_engine = wrapped
        if deepspeed_engine is not None:
            deepspeed_engine.lr_scheduler = restarted_scheduler
            self.lr_scheduler = ReadOnlySchedulerProxy(restarted_scheduler)
        else:
            self.lr_scheduler = restarted_scheduler
        if int(os.getenv("RANK", "0")) == 0:
            print(
                "PATHVLM_SCHEDULER_RESTART_JSON="
                + json.dumps({
                    "optimizer_state_resumed": True,
                    "scheduler_state_resumed_then_replaced": True,
                    "reset_steps": reset_steps,
                    "peak_lr": peak_lr,
                    "scheduler_type": str(self.args.lr_scheduler_type),
                }, sort_keys=True),
                flush=True,
            )

    trainer._load_optimizer_and_scheduler = MethodType(
        load_optimizer_then_restart, trainer
    )
    return {
        "optimizer_state_resumed": True,
        "scheduler_state_resumed_then_replaced": True,
        "reset_steps": reset_steps,
        "peak_lr": peak_lr,
        "scheduler_type": str(trainer.args.lr_scheduler_type),
    }


def main() -> None:
    grpo_rec.reward_funcs_registry["accuracy"] = audited_accuracy_reward
    grpo_rec.reward_funcs_registry["format"] = audited_format_reward
    grpo_rec.reward_funcs_registry["format_scaled"] = audited_scaled_format_reward
    grpo_rec.reward_funcs_registry["process"] = audited_process_reward
    parser = TrlParser((grpo_rec.GRPOScriptArguments, grpo_rec.GRPOConfig, grpo_rec.GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    reward_funcs = [grpo_rec.reward_funcs_registry[name] for name in script_args.reward_funcs]
    dataset = PathMMUStrictPromptDataset(script_args.dataset_name, script_args)
    peft_config = get_peft_config(model_args)
    allow_language_lora = os.getenv("PATHVLM_ALLOW_LANGUAGE_LORA", "").lower() == "true"
    if peft_config is not None and not allow_language_lora:
        raise RuntimeError("formal PathMMU GRPO gate forbids PEFT/LoRA unless explicitly enabled")
    if allow_language_lora and peft_config is None:
        raise RuntimeError("PATHVLM_ALLOW_LANGUAGE_LORA requires --use_peft true")
    trainer = Qwen2VLGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=peft_config,
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
        callbacks=[ExplicitSaveStepsCallback(), StopAfterSaveCallback(), EpochSnapshotCallback()],
    )
    generation_contract = freeze_generation_contract(trainer, training_args)
    scheduler_restart = install_scheduler_restart_on_resume(trainer)
    report = trainability_report(
        trainer.model, language_mode="lora" if allow_language_lora else "full"
    )
    if int(os.getenv("RANK", "0")) == 0:
        print("PATHVLM_TRAINABILITY_JSON=" + json.dumps(report, sort_keys=True), flush=True)
        print(
            "PATHVLM_EFFECTIVE_GENERATION_CONTRACT_JSON="
            + json.dumps(generation_contract, sort_keys=True),
            flush=True,
        )
    if not report["passed"]:
        raise RuntimeError("formal trainability/freeze gate failed: " + json.dumps(report))

    resume = os.getenv("PATHVLM_RESUME_FROM_CHECKPOINT") or None
    train_result = trainer.train(resume_from_checkpoint=resume)
    stop_target = int(os.getenv("PATHVLM_STOP_AFTER_SAVED_STEP", "0"))
    if stop_target and trainer.state.global_step != stop_target:
        raise RuntimeError(
            f"requested stop after saved step {stop_target}, reached {trainer.state.global_step}"
        )
    if os.getenv("PATHVLM_SKIP_FINAL_MODEL_SAVE", "").lower() != "true":
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
            "training_segment": os.getenv("PATHVLM_TRAINING_SEGMENT", "unspecified"),
            "resume_from_checkpoint": resume,
            "stop_after_saved_step": stop_target or None,
            "effective_generation_contract": generation_contract,
            "scheduler_restart_on_resume": scheduler_restart,
        }
        audit_name = os.getenv("PATHVLM_TRAIN_STATE_AUDIT_NAME", "pathvlm_train_state_audit.json")
        audit_path = Path(training_args.output_dir) / audit_name
        audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        print("PATHVLM_TRAIN_RESULT_JSON=" + json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
