#!/usr/bin/env python3
"""Isolated regression tests for the formal SFT epoch validation curve."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "formal_sft_validation_curve",
    REPO / "formal_machine" / "run_formal_sft_validation_curve.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CurveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_snapshot(self) -> Path:
        snapshot = self.root / "checkpoint-250"
        snapshot.mkdir()
        names = ["config.json", "model.safetensors.index.json", "preprocessor_config.json",
                 "tokenizer_config.json", "trainer_state.json"]
        files = []
        for name in names:
            path = snapshot / name
            path.write_text(f"{name}\n", encoding="utf-8")
            files.append({"name": name, "size_bytes": path.stat().st_size,
                          "sha256": digest(path), "transfer": "hardlink"})
        payload = {"epoch": 1.0, "global_step": 250, "resumable": False, "files": files}
        (snapshot / "snapshot_manifest.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        return snapshot

    def test_valid_snapshot_passes(self) -> None:
        evidence = MODULE.verify_snapshot(self.make_snapshot(), 1, 250)
        self.assertTrue(evidence["verified"])
        self.assertEqual(evidence["file_count"], 5)

    def test_corrupt_snapshot_stops(self) -> None:
        snapshot = self.make_snapshot()
        (snapshot / "config.json").write_text("corrupt\n", encoding="utf-8")
        with self.assertRaises(MODULE.CurveStop):
            MODULE.verify_snapshot(snapshot, 1, 250)

    def test_wrong_snapshot_step_stops(self) -> None:
        with self.assertRaises(MODULE.CurveStop):
            MODULE.verify_snapshot(self.make_snapshot(), 1, 500)

    def make_results(self, corrupt_index: bool = False) -> tuple[Path, list[dict]]:
        results = self.root / "results"
        results.mkdir()
        records, rows = [], []
        for index in range(385):
            solution = "<think>target</think><answer>A) target</answer>"
            completion = "<think>reason</think><answer>A) selected</answer>"
            record = {"image": f"/images/{index}.png", "problem": f"question {index}",
                      "solution": solution}
            records.append(record)
            rows.append({"index": index + (1 if corrupt_index and index == 0 else 0), **record,
                         "completion": completion, "predicted_choice": "A",
                         "target_choice": "A", "accuracy_reward": 1.0,
                         "format_reward": 1.0})
        (results / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        metrics = {"count": 385, "mean_accuracy_reward": 1.0,
                   "mean_format_reward": 1.0, "max_new_tokens": 192,
                   "do_sample": False, "test_accessed": False}
        (results / "metrics.json").write_text(json.dumps(metrics) + "\n", encoding="utf-8")
        return results, records

    def test_offline_rescore_passes(self) -> None:
        metrics, audit = MODULE.audit_results(*self.make_results())
        self.assertEqual(metrics["mean_accuracy_reward"], 1.0)
        self.assertTrue(audit["parser_consistency"])
        self.assertEqual(audit["accuracy_correct"], 385)

    def test_prediction_index_mismatch_stops(self) -> None:
        with self.assertRaises(MODULE.CurveStop):
            MODULE.audit_results(*self.make_results(corrupt_index=True))

    def test_selection_rule_accuracy_then_format_then_earliest(self) -> None:
        rows = []
        for epoch in range(1, 11):
            rows.append({"sample_count": 2000, "epoch": epoch, "label": f"e{epoch}",
                         "global_step": epoch * 250,
                         "metrics": {"mean_accuracy_reward": 0.5,
                                     "mean_format_reward": 0.9}})
        rows[2]["metrics"]["mean_accuracy_reward"] = 0.6
        rows[4]["metrics"]["mean_accuracy_reward"] = 0.6
        rows[4]["metrics"]["mean_format_reward"] = 0.95
        self.assertEqual(MODULE.select_epoch(rows, 2000)["epoch"], 5)
        rows[2]["metrics"]["mean_format_reward"] = 0.95
        self.assertEqual(MODULE.select_epoch(rows, 2000)["epoch"], 3)

    def test_selection_requires_ten_epochs(self) -> None:
        with self.assertRaises(MODULE.CurveStop):
            MODULE.select_epoch([], 3000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
