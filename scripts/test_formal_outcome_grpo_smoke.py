#!/usr/bin/env python3
"""Pure regression tests for the formal one-step Outcome-GRPO gate auditor."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "formal_outcome_grpo_gate", REPO / "formal_machine/run_formal_outcome_grpo_smoke.py"
)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    reward_dir = root / "rewards"
    reward_dir.mkdir()
    solutions = ["<answer>B) target</answer>"] * 4 + ["<answer>C) target</answer>"] * 4
    choices = ["B", "A", "B", "A", "C", "D", "C", "D"]
    for rank, (solution, choice) in enumerate(zip(solutions, choices)):
        completion = f"<think>reason</think><answer>{choice}</answer>"
        accuracy = float(choice in solution[:20])
        common = {
            "rank": rank, "local_rank": rank, "pid": 1000 + rank,
            "call_index": 0, "item_index": 0,
            "completion": completion, "solution": solution,
        }
        events = [
            {**common, "reward_type": "accuracy", "reward": accuracy},
            {**common, "reward_type": "format", "reward": 1.0},
        ]
        (reward_dir / f"rank_{rank:02d}.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
        )
    audit = gate.audit_rewards(REPO, reward_dir)
    assert audit["event_count"] == 16
    assert audit["completion_count"] == 8
    assert audit["parser_consistency"] is True
    assert audit["positive_reward_variance"] is True
    assert audit["positive_format_reward_count"] == 8
    assert audit["format_reward_observed_positive"] is True

    output = root / "output"
    output.mkdir()
    (output / "pathvlm_train_state_audit.json").write_text(
        json.dumps(
            {
                "global_step": 1,
                "trainability": {"passed": True},
                "log_history": [
                    {"loss": 0.25, "grad_norm": 0.5, "reward": 1.5, "reward_std": 0.4}
                ],
            }
        ),
        encoding="utf-8",
    )
    state = gate.audit_training_state(output)
    assert state["gradient_finite_nonzero"] is True
    assert state["trainer_reward_std_positive"] is True
    assert state["global_step"] == 1

command = gate.build_command(
    REPO, Path("/formal"), Path("/formal/envs/grpo/bin/python"),
    Path("/run/parent_Qwen2.5-VL"), Path("/run/output"),
)
joined = " ".join(map(str, command)).lower()
assert "--max_steps 1" in joined
assert "pathvlm_rl_smoke_n0008.yaml" in joined
assert "test" not in joined
assert "picked.json" not in joined
print("formal Outcome-GRPO smoke auditor tests: PASS")
