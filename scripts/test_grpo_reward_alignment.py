#!/usr/bin/env python3
"""CUDA-independent regression tests for multimodal GRPO reward alignment."""

from __future__ import annotations

import sys
from pathlib import Path


VENDOR_SRC = Path(__file__).resolve().parents[1] / "vendor" / "open-r1-multimodal" / "src"
sys.path.insert(0, str(VENDOR_SRC))

from open_r1.trainer.reward_alignment import (  # noqa: E402
    build_aligned_reward_kwargs,
    require_reward_output_count,
)


def repeated_row(question: str, solution: str) -> dict[str, object]:
    return {
        "prompt": [{"role": "user", "content": question}],
        "problem": question,
        "solution": solution,
        "record_index": int(question.removeprefix("q")),
    }


# This is the exact local shape that exposed the defect: per-device batch 10,
# four generations per prompt, with one group crossing the rank boundary.
inputs = (
    [repeated_row("q0", "A")] * 4
    + [repeated_row("q1", "B")] * 4
    + [repeated_row("q2", "C")] * 2
)
kwargs = build_aligned_reward_kwargs(inputs, expected_count=10)
assert kwargs["problem"] == ["q0"] * 4 + ["q1"] * 4 + ["q2"] * 2
assert kwargs["solution"] == ["A"] * 4 + ["B"] * 4 + ["C"] * 2
assert kwargs["record_index"] == [0] * 4 + [1] * 4 + [2] * 2
assert "prompt" not in kwargs

try:
    build_aligned_reward_kwargs(inputs, expected_count=40)
except RuntimeError as exc:
    assert "input/completion length mismatch" in str(exc)
else:
    raise AssertionError("expanded reward kwargs must be rejected")

try:
    require_reward_output_count([1.0] * 9, expected_count=10, reward_name="accuracy")
except RuntimeError as exc:
    assert "output length mismatch" in str(exc)
else:
    raise AssertionError("truncated reward output must be rejected")

require_reward_output_count([1.0] * 10, expected_count=10, reward_name="accuracy")
print("GRPO reward alignment regression tests: PASS")
