"""Fail-closed alignment helpers for custom GRPO reward functions.

The multimodal samplers in this vendor tree already repeat each dataset row
``num_generations`` times before a local batch reaches the trainer.  Custom
reward kwargs must therefore be copied one-for-one from the local ``inputs``;
repeating them again silently pairs completions with another prompt's target
when the per-device batch size is greater than one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


EXCLUDED_REWARD_COLUMNS = frozenset({"prompt", "completion"})


def build_aligned_reward_kwargs(
    inputs: Sequence[Mapping[str, Any]], *, expected_count: int
) -> dict[str, list[Any]]:
    """Copy custom-reward columns one-for-one and verify their batch length."""
    if len(inputs) != expected_count:
        raise RuntimeError(
            "GRPO reward input/completion length mismatch: "
            f"inputs={len(inputs)}, completions={expected_count}"
        )
    if not inputs:
        return {}

    keys = [key for key in inputs[0] if key not in EXCLUDED_REWARD_COLUMNS]
    for index, example in enumerate(inputs):
        missing = [key for key in keys if key not in example]
        if missing:
            raise RuntimeError(
                f"GRPO reward input {index} is missing columns: {sorted(missing)}"
            )

    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
    for key, values in reward_kwargs.items():
        if len(values) != expected_count:
            raise RuntimeError(
                f"GRPO reward column length mismatch for {key}: "
                f"{len(values)} != {expected_count}"
            )
    return reward_kwargs


def require_reward_output_count(
    rewards: Sequence[Any], *, expected_count: int, reward_name: str
) -> None:
    """Reject custom reward functions that return a truncated/expanded batch."""
    if len(rewards) != expected_count:
        raise RuntimeError(
            f"GRPO reward output length mismatch for {reward_name}: "
            f"{len(rewards)} != {expected_count}"
        )
