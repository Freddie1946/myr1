#!/usr/bin/env python3
"""Regression tests for corrected all-candidate validation."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load("validation_1024_runner", REPO / "formal_machine/run_formal_validation_1024_all_candidates.py")
INFERENCE = load("validation_1024_inference", REPO / "scripts/infer_and_score_pathmmu_with_tokens.py")


class CorrectedValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def rows(self, cap_hit: bool = False) -> tuple[Path, list[dict], Path]:
        results = self.root / "results"
        results.mkdir()
        records, rows = [], []
        for index in range(385):
            solution = "<think>target</think><answer>A) target</answer>"
            completion = "<think>reason</think><answer>A) selected</answer>"
            record = {"image": f"/images/{index}.png", "problem": f"question {index}",
                      "solution": solution}
            records.append(record)
            length = RUNNER.MAX_NEW_TOKENS if cap_hit and index == 0 else 20
            rows.append({"index": index, **record, "completion": completion,
                         "generated_token_count": length, "ended_with_eos": not (cap_hit and index == 0),
                         "reached_generation_cap": cap_hit and index == 0,
                         "predicted_choice": "A", "target_choice": "A",
                         "accuracy_reward": 1.0, "format_reward": 1.0})
        predictions = results / "predictions.jsonl"
        predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        chat = self.root / "chat_template.json"
        chat.write_text("{}\n", encoding="utf-8")
        metrics = INFERENCE.summarize_rows(rows, RUNNER.MAX_NEW_TOKENS)
        metrics.update({"chat_template_file": str(chat.resolve()),
                        "chat_template_sha256": RUNNER.CHAT_TEMPLATE_SHA256})
        (results / "metrics.json").write_text(json.dumps(metrics) + "\n", encoding="utf-8")
        return results, records, chat

    def test_eos_id_normalization(self) -> None:
        self.assertEqual(INFERENCE.eos_ids(None), set())
        self.assertEqual(INFERENCE.eos_ids(7), {7})
        self.assertEqual(INFERENCE.eos_ids([7, 9]), {7, 9})

    def test_offline_audit_passes(self) -> None:
        metrics, audit = RUNNER.audit_results(*self.rows())
        self.assertEqual(metrics["max_new_tokens"], 1024)
        self.assertEqual(audit["accuracy_correct"], 385)
        self.assertEqual(audit["generation_cap_hit_count"], 0)

    def test_generation_cap_hit_is_preserved_and_scored(self) -> None:
        metrics, audit = RUNNER.audit_results(*self.rows(cap_hit=True))
        self.assertEqual(metrics["generation_cap_hit_count"], 1)
        self.assertEqual(audit["generation_cap_hit_count"], 1)

    def test_selection_accuracy_format_then_earliest(self) -> None:
        rows = []
        for epoch in range(1, 4):
            rows.append({"family": "outcome_grpo", "sample_count": 1000, "epoch": epoch,
                         "metrics": {"mean_accuracy_reward": 0.5, "mean_format_reward": 0.9}})
        rows[1]["metrics"]["mean_accuracy_reward"] = 0.6
        rows[2]["metrics"]["mean_accuracy_reward"] = 0.6
        rows[2]["metrics"]["mean_format_reward"] = 0.95
        self.assertEqual(RUNNER.select(rows, "outcome_grpo", 1000, 3)["epoch"], 3)
        rows[1]["metrics"]["mean_format_reward"] = 0.95
        self.assertEqual(RUNNER.select(rows, "outcome_grpo", 1000, 3)["epoch"], 2)

    def test_selection_rejects_missing_candidate(self) -> None:
        with self.assertRaises(RUNNER.ValidationStop):
            RUNNER.select([], "sft", 3000, 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
