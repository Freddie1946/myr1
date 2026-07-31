#!/usr/bin/env python3
"""Audited PathMMU wrapper around the online Qwen2.5-VL GRPO trainer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from grpo_pathmmu_audit import (
    STRICT_PROMPT_CONTRACT,
    aligned_solutions,
    append_audit_events,
    create_model_only_snapshot,
    replace_legacy_json_prompt,
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


def aligned_source_metadata(kwargs: dict[str, Any], count: int) -> list[dict[str, Any]]:
    rows = [{} for _ in range(count)]
    for field in SOURCE_AUDIT_FIELDS:
        values = kwargs.get("prompts" if field == "prompt" else field)
        if values is None:
            if os.getenv("PATHVLM_REQUIRE_SOURCE_AUDIT", "").lower() == "true":
                raise RuntimeError(f"formal reward source metadata is missing: {field}")
            continue
        aligned = list(values)[:count]
        if len(aligned) != count:
            raise RuntimeError(f"formal reward source metadata length mismatch: {field}")
        for row, value in zip(rows, aligned):
            row[field] = value
    return rows


def audited_accuracy_reward(completions, solution, **kwargs):
    rewards = accuracy_reward(completions, solution, **kwargs)
    solutions = aligned_solutions(solution, len(completions))
    metadata = aligned_source_metadata(kwargs, len(completions))
    append_audit_events("accuracy", list(completions), solutions, rewards, metadata)
    return rewards


def audited_format_reward(completions, solution=None, **kwargs):
    rewards = format_reward(completions, **kwargs)
    solutions = aligned_solutions(solution, len(completions))
    metadata = aligned_source_metadata(kwargs, len(completions))
    append_audit_events("format", list(completions), solutions, rewards, metadata)
    return rewards


def audited_process_reward(completions, solution, **kwargs):
    rewards = process_reward(completions, solution, **kwargs)
    solutions = aligned_solutions(solution, len(completions))
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
        destination = Path(root_text) / f"checkpoint-{state.global_step}"
        create_model_only_snapshot(
            Path(args.output_dir) / f"checkpoint-{state.global_step}",
            destination,
            global_step=state.global_step,
            epoch=float(state.epoch or 0.0),
        )
        return control


def main() -> None:
    grpo_rec.reward_funcs_registry["accuracy"] = audited_accuracy_reward
    grpo_rec.reward_funcs_registry["format"] = audited_format_reward
    grpo_rec.reward_funcs_registry["process"] = audited_process_reward
    parser = TrlParser((grpo_rec.GRPOScriptArguments, grpo_rec.GRPOConfig, grpo_rec.GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    reward_funcs = [grpo_rec.reward_funcs_registry[name] for name in script_args.reward_funcs]
    dataset = PathMMUStrictPromptDataset(script_args.dataset_name, script_args)
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
        callbacks=[StopAfterSaveCallback(), EpochSnapshotCallback()],
    )
    report = trainability_report(trainer.model)
    if int(os.getenv("RANK", "0")) == 0:
        print("PATHVLM_TRAINABILITY_JSON=" + json.dumps(report, sort_keys=True), flush=True)
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
        }
        audit_name = os.getenv("PATHVLM_TRAIN_STATE_AUDIT_NAME", "pathvlm_train_state_audit.json")
        audit_path = Path(training_args.output_dir) / audit_name
        audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        print("PATHVLM_TRAIN_RESULT_JSON=" + json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
