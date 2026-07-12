#!/usr/bin/env python3
"""PathMMU wrapper around the repository's online Qwen2.5-VL GRPO entry."""

from open_r1 import grpo_rec
from pathmmu_rewards import accuracy_reward, format_reward
from trl import TrlParser


def main() -> None:
    grpo_rec.reward_funcs_registry["accuracy"] = accuracy_reward
    grpo_rec.reward_funcs_registry["format"] = format_reward
    parser = TrlParser((grpo_rec.GRPOScriptArguments, grpo_rec.GRPOConfig, grpo_rec.GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    grpo_rec.main(script_args, training_args, model_args)


if __name__ == "__main__":
    main()

