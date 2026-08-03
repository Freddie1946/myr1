#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from verify_local_baseline_smoke import verify


class VerifyLocalBaselineSmokeTests(unittest.TestCase):
    def fixture(self, root: Path, task: str, rows, answer_scope=None):
        predictions = root / "predictions.jsonl"
        predictions.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        digest = hashlib.sha256(predictions.read_bytes()).hexdigest()
        metrics = root / "metrics.json"
        value = {
                    "status": "completed",
                    "task": task,
                    "split_role": "validation_smoke" if task == "pathmmu" else "adapter_smoke",
                    "count": len(rows),
                    "data_sha256": "d" * 64,
                    "model_config_sha256": "m" * 64,
                    "predictions_sha256": digest,
        }
        if answer_scope is not None:
            value["answer_scope"] = answer_scope
        metrics.write_text(json.dumps(value), encoding="utf-8")
        return metrics, predictions

    def run_verify(self, task, metrics, predictions, count, answer_scope=None):
        return verify(
            task=task,
            metrics_path=metrics,
            predictions_path=predictions,
            expected_count=count,
            expected_data_sha256="d" * 64,
            expected_model_config_sha256="m" * 64,
            minimum_nonempty_rate=0.80,
            minimum_parseable_rate=0.80,
            maximum_cap_hit_rate=0.20,
            expected_pathvqa_answer_scope=answer_scope,
        )

    def test_wrong_but_parseable_pathmmu_answer_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [{
                "index": 0, "source_record_sha256": "a" * 64,
                "completion": "<answer>B</answer>", "predicted_choice": "B",
                "accuracy_reward": 0.0, "reached_generation_cap": False,
            }]
            metrics, predictions = self.fixture(Path(directory), "pathmmu", rows)
            result = self.run_verify("pathmmu", metrics, predictions, 1)
            self.assertEqual(result["status"], "passed")
            self.assertFalse(result["accuracy_used_as_gate"])

    def test_pathmmu_accepts_legacy_metrics_without_task_field(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [{
                "index": 0, "source_record_sha256": "a" * 64,
                "completion": "<answer>A</answer>", "predicted_choice": "A",
                "reached_generation_cap": False,
            }]
            metrics, predictions = self.fixture(Path(directory), "pathmmu", rows)
            value = json.loads(metrics.read_text())
            del value["task"]
            metrics.write_text(json.dumps(value))
            result = self.run_verify("pathmmu", metrics, predictions, 1)
            self.assertEqual(result["status"], "passed")

    def test_pathvqa_requires_both_answer_types(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [
                {"index": 0, "source_record_sha256": "a" * 64, "completion": "tissue", "answer_type": "free_form", "reached_generation_cap": False},
                {"index": 1, "source_record_sha256": "b" * 64, "completion": "yes", "answer_type": "yes_no", "normalized_completion": "yes", "reached_generation_cap": False},
            ]
            metrics, predictions = self.fixture(Path(directory), "pathvqa", rows)
            result = self.run_verify("pathvqa", metrics, predictions, 2)
            self.assertEqual(result["parseable_rate"], 1.0)

    def test_pathvqa_yes_no_only_scope_does_not_require_free_form(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [
                {"index": 0, "source_record_sha256": "a" * 64, "completion": "yes", "answer_type": "yes_no", "normalized_completion": "yes", "reached_generation_cap": False},
                {"index": 1, "source_record_sha256": "b" * 64, "completion": "no", "answer_type": "yes_no", "normalized_completion": "no", "reached_generation_cap": False},
            ]
            metrics, predictions = self.fixture(
                Path(directory), "pathvqa", rows, answer_scope="yes_no_only"
            )
            result = self.run_verify(
                "pathvqa", metrics, predictions, 2, answer_scope="yes_no_only"
            )
            self.assertEqual(result["parseable_count"], 2)

    def test_empty_or_unparseable_output_fails_aggregate_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [{
                "index": 0, "source_record_sha256": "a" * 64,
                "completion": "", "official_predicted_choice": None,
                "reached_generation_cap": False,
            }]
            metrics, predictions = self.fixture(Path(directory), "omnimedvqa", rows)
            with self.assertRaisesRegex(ValueError, "aggregate smoke behavior failed"):
                self.run_verify("omnimedvqa", metrics, predictions, 1)

    def test_prediction_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [{
                "index": 0, "source_record_sha256": "a" * 64,
                "completion": "A", "predicted_choice": "A",
                "reached_generation_cap": False,
            }]
            metrics, predictions = self.fixture(Path(directory), "pathmmu", rows)
            value = json.loads(metrics.read_text())
            value["predictions_sha256"] = "0" * 64
            metrics.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                self.run_verify("pathmmu", metrics, predictions, 1)


if __name__ == "__main__":
    unittest.main()
