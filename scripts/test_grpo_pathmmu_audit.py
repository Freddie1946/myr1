#!/usr/bin/env python3
"""Regression tests for distributed reward logging and trainability accounting."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch

from grpo_pathmmu_audit import (
    STRICT_PROMPT_CONTRACT,
    aligned_solutions,
    append_audit_events,
    create_model_only_snapshot,
    replace_legacy_json_prompt,
    strict_prompt_text,
    trainability_report,
)


def wrapped(text: str):
    return [[{"role": "assistant", "content": text}]]


prompt = strict_prompt_text("Question?\nOptions:\nA) one\nB) two")
assert STRICT_PROMPT_CONTRACT == "pathmmu_think_answer_only_v2"
assert prompt.endswith("<answer> </answer> tags.")
assert "JSON" not in prompt
assert prompt.count("Question?") == 1
legacy = prompt + " Output the final answer in JSON format."
assert replace_legacy_json_prompt(legacy, "Question?\nOptions:\nA) one\nB) two") == prompt
try:
    replace_legacy_json_prompt(prompt, "Question?")
except RuntimeError:
    pass
else:
    raise AssertionError("legacy prompt guard must fail when the expected suffix is absent")


with tempfile.TemporaryDirectory() as temporary:
    os.environ["PATHVLM_REWARD_LOG_DIR"] = temporary
    os.environ["RANK"] = "3"
    os.environ["LOCAL_RANK"] = "3"
    completions = wrapped("<think>reason</think><answer>B</answer>")
    # The vendored trainer repeats reward kwargs by num_generations even though
    # this rank owns one completion. The wrapper must audit the aligned prefix.
    solutions = ["<answer>B) target</answer>"] * 4
    aligned = aligned_solutions(solutions, len(completions))
    assert aligned == ["<answer>B) target</answer>"]
    metadata = [{"record_index": 17, "image_sha256": "abc", "problem": "Question?"}]
    os.environ["PATHVLM_TRAINING_SEGMENT"] = "segment_a"
    append_audit_events("accuracy", completions, aligned, [1.0], metadata)
    append_audit_events("format", completions, aligned, [1.0], metadata)
    path = Path(temporary) / "rank_03.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(events) == 2
    assert {event["reward_type"] for event in events} == {"accuracy", "format"}
    assert all(event["rank"] == 3 and event["local_rank"] == 3 for event in events)
    assert events[0]["completion"] == events[1]["completion"]
    assert events[0]["solution"] == events[1]["solution"]
    assert events[0]["item_index"] == events[1]["item_index"] == 0
    assert all(event["record_index"] == 17 for event in events)
    assert all(event["training_segment"] == "segment_a" for event in events)


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    checkpoint = root / "checkpoint-25"
    checkpoint.mkdir()
    files = {
        "config.json": b"{}\n",
        "preprocessor_config.json": b"{}\n",
        "tokenizer_config.json": b"{}\n",
        "model.safetensors": b"weights",
        "optimizer.pt": b"must not be retained",
    }
    for name, content in files.items():
        (checkpoint / name).write_bytes(content)
    destination = root / "epoch_snapshots" / "checkpoint-25"
    destination.parent.mkdir()
    manifest = create_model_only_snapshot(checkpoint, destination, global_step=25, epoch=1.0)
    assert manifest["resumable"] is False
    assert not (destination / "optimizer.pt").exists()
    assert (destination / "model.safetensors").read_bytes() == b"weights"
    assert (destination / "snapshot_manifest.json").is_file()


class FakeModel:
    def __init__(self):
        self.language = torch.nn.Parameter(torch.zeros(7), requires_grad=True)
        self.visual = torch.nn.Parameter(torch.zeros(5), requires_grad=False)
        self.projector = torch.nn.Parameter(torch.zeros(3), requires_grad=False)

    def named_parameters(self):
        return iter(
            [
                ("model.layers.0.weight", self.language),
                ("visual.blocks.0.weight", self.visual),
                ("visual.merger.linear.weight", self.projector),
            ]
        )


report = trainability_report(FakeModel())
assert report["passed"] is True
assert report["parameters"]["language"] == {"total": 7, "trainable": 7}
assert report["parameters"]["visual"] == {"total": 8, "trainable": 0}
assert report["parameters"]["multimodal_projector"] == {"total": 3, "trainable": 0}
print("GRPO PathMMU audit tests: PASS")
