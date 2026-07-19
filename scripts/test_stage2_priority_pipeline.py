#!/usr/bin/env python3
"""CUDA-independent contract tests for the Stage-2 priority pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import formal_machine.run_stage2_priority_pipeline as pipeline


def prediction(index: int, correct: bool) -> dict:
    return {"index": index, "accuracy_reward": 1.0 if correct else 0.0}


parent = [prediction(index, index < 6) for index in range(10)]
child = [prediction(index, index in {0, 1, 2, 3, 6, 7}) for index in range(10)]
paired = pipeline.exact_mcnemar(parent, child)
assert paired["parent_only_correct"] == 2
assert paired["child_only_correct"] == 2
assert paired["significant_child_degradation"] is False

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    command = pipeline.build_train_command(
        Path("/python"), Path("/parent_Qwen2.5-VL"), Path("/data.yaml"), root,
        max_steps=50, epochs=None, save_strategy="steps", save_steps=25, save_total_limit=1,
    )
    joined = " ".join(command)
    assert "--max_steps 50" in joined
    assert "--save_steps 25" in joined
    assert "--remove_unused_columns false" in joined
    assert "validation" not in joined.lower()
    assert "test" not in joined.lower()

protocol = json.loads(
    (pipeline.REPO / "protocol" / pipeline.PROTOCOL_NAME).read_text(encoding="utf-8")
)
assert protocol["rl_sample_count"] == 1000
assert protocol["pilot_resume_step"] == 25
assert protocol["formal_steps"] == 1500
assert protocol["test_authorized"] is False
assert protocol["other_combinations_authorized"] is False
print("Stage-2 priority pipeline tests: PASS")
