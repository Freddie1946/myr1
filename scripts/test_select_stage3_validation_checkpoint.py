#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from select_stage3_validation_checkpoint import EXPECTED_DATA_SHA256, select


class SelectStage3ValidationCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.run = root / "run"
        self.validation = root / "validation"
        for step in (500, 1000, 1500):
            (self.run / "epoch_model_snapshots" / f"checkpoint-{step}").mkdir(parents=True)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def write_metrics(self, step: int, correct: int, format_correct: int) -> None:
        out = self.validation / f"checkpoint-{step}"
        out.mkdir(parents=True)
        predictions = out / "predictions.jsonl"
        predictions.write_text('{"index":0}\n')
        metrics = {
            "status": "completed",
            "split_role": "validation_0385",
            "test_accessed": False,
            "count": 385,
            "correct": correct,
            "accuracy": correct / 385,
            "format_correct": format_correct,
            "format_accuracy": format_correct / 385,
            "choice_extracted": 385,
            "generation_cap_hit_count": 0,
            "model_path": str((self.run / "epoch_model_snapshots" / f"checkpoint-{step}").resolve()),
            "data_sha256": EXPECTED_DATA_SHA256,
            "predictions_file": str(predictions.resolve()),
            "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
        }
        (out / "metrics.json").write_text(json.dumps(metrics))

    def test_accuracy_wins(self) -> None:
        self.write_metrics(500, 200, 380)
        self.write_metrics(1000, 201, 300)
        self.write_metrics(1500, 199, 385)
        self.assertEqual(select(self.run, self.validation)["selected_step"], 1000)

    def test_format_then_earliest_break_ties(self) -> None:
        self.write_metrics(500, 200, 380)
        self.write_metrics(1000, 200, 381)
        self.write_metrics(1500, 200, 381)
        self.assertEqual(select(self.run, self.validation)["selected_step"], 1000)

    def test_test_access_fails(self) -> None:
        for step in (500, 1000, 1500):
            self.write_metrics(step, 200, 380)
        path = self.validation / "checkpoint-500" / "metrics.json"
        metrics = json.loads(path.read_text())
        metrics["test_accessed"] = True
        path.write_text(json.dumps(metrics))
        with self.assertRaisesRegex(ValueError, "validation contract mismatch"):
            select(self.run, self.validation)

    def test_generic_validation_smoke_metrics_are_supported_explicitly(self) -> None:
        for step in (500, 1000, 1500):
            self.write_metrics(step, 200 + step // 500, 385)
            path = self.validation / f"checkpoint-{step}" / "metrics.json"
            metrics = json.loads(path.read_text())
            metrics["split_role"] = "validation_smoke"
            metrics.pop("predictions_file")
            metrics.pop("format_accuracy")
            path.write_text(json.dumps(metrics))
        selected = select(
            self.run, self.validation, expected_split_role="validation_smoke"
        )
        self.assertEqual(selected["selected_step"], 1500)
        self.assertEqual(selected["inference_split_role"], "validation_smoke")


if __name__ == "__main__":
    unittest.main()
