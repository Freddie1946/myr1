"""PathMMU GRPO training entrypoint.

This is derived from the existing grpo_rec.py but removes grounding/bbox-specific
format reward logic and adds an optional GPT-4o process reward.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import yaml
from PIL import Image
from torch.utils.data import Dataset
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

from open_r1.trainer import GRPOConfig, Qwen2VLGRPOTrainer
from pathvlm_core.answer_parser import exact_answer_match, has_think_answer_format
from pathvlm_core.gpt_process_reward import call_gpt4o_process_reward

os.environ.setdefault("WANDB_MODE", "disabled")

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a pathology image question, "
    "and the Assistant solves it. The Assistant first reasons in <think></think> tags, "
    "then gives the final answer in <answer></answer> tags."
)

QUESTION_TEMPLATE = (
    "{Question}\n"
    "First output the thinking process in <think></think> tags. "
    "Then output the final answer in <answer></answer> tags."
)


@dataclass
class GRPOScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "Reward functions: accuracy, format, process"},
    )
    max_pixels: Optional[int] = field(default=12845056)
    min_pixels: Optional[int] = field(default=3136)
    image_root: Optional[str] = field(default=None)


@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


class PathMMUDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments):
        self.script_args = script_args
        self.list_data_dict: list[dict] = []
        if not data_path.endswith((".yaml", ".yml")):
            raise ValueError(f"Unsupported file type: {data_path}")
        with open(data_path, "r", encoding="utf-8") as handle:
            yaml_data = yaml.safe_load(handle)
        for item in yaml_data.get("datasets", []):
            json_path = os.path.expandvars(item["json_path"])
            sampling_strategy = item.get("sampling_strategy", "all")
            rows = self._load_rows(json_path)
            rows = self._apply_sampling(rows, sampling_strategy)
            print(f"Loaded {len(rows)} samples from {json_path}")
            self.list_data_dict.extend(rows)

    @staticmethod
    def _load_rows(json_path: str) -> list[dict]:
        if json_path.endswith(".jsonl"):
            with open(json_path, "r", encoding="utf-8") as handle:
                return [json.loads(line) for line in handle]
        if json_path.endswith(".json"):
            with open(json_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, list) else data.get("results", [])
        raise ValueError(f"Unsupported data file: {json_path}")

    @staticmethod
    def _apply_sampling(rows: list[dict], sampling_strategy: str) -> list[dict]:
        if ":" not in sampling_strategy:
            return rows
        strategy, number = sampling_strategy.split(":")
        count = math.ceil(int(number[:-1]) * len(rows) / 100) if number.endswith("%") else int(number)
        if strategy == "first":
            return rows[:count]
        if strategy == "end":
            return rows[-count:]
        if strategy == "random":
            rows = rows[:]
            random.shuffle(rows)
            return rows[:count]
        return rows

    def __len__(self) -> int:
        return len(self.list_data_dict)

    def __getitem__(self, index: int) -> dict:
        example = self.list_data_dict[index]
        image = None
        if example.get("image"):
            image_path = example["image"]
            if self.script_args.image_root and not os.path.isabs(image_path):
                image_path = os.path.join(self.script_args.image_root, image_path)
            image = Image.open(image_path).convert("RGB")
        prompt_content = []
        if image is not None:
            prompt_content.append({"type": "image"})
        prompt_content.append({"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])})
        return {
            "image": image,
            "problem": example["problem"],
            "solution": example["solution"],
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt_content}],
        }


def accuracy_reward(completions, solution, **kwargs):
    return [1.0 if exact_answer_match(completion[0]["content"], gold) else 0.0 for completion, gold in zip(completions, solution)]


def format_reward(completions, **kwargs):
    return [1.0 if has_think_answer_format(completion[0]["content"]) else 0.0 for completion in completions]


def process_reward(completions, solution, problem=None, image=None, **kwargs):
    if os.environ.get("PATHVLM_ENABLE_GPT_REWARD", "false").lower() != "true":
        return [0.0 for _ in completions]
    problems = problem or [""] * len(completions)
    images = image or [None] * len(completions)
    rewards = []
    for completion, gold, question, pil_image in zip(completions, solution, problems, images):
        output = completion[0]["content"]
        is_correct = exact_answer_match(output, gold)
        scores = call_gpt4o_process_reward(question, output, gold, is_correct, pil_image)
        rewards.append((scores["integrity_score"] + scores["knowledge_score"]) / 2.0)
    return rewards


reward_funcs_registry = {"accuracy": accuracy_reward, "format": format_reward, "process": process_reward}


def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[name] for name in script_args.reward_funcs]
    print("reward_funcs:", script_args.reward_funcs)
    dataset = PathMMUDataset(script_args.dataset_name, script_args)
    trainer = Qwen2VLGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
