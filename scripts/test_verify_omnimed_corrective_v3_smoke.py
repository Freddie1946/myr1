#!/usr/bin/env python3

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from verify_omnimed_corrective_v3_smoke import verify


class VerifyOmnimedCorrectiveV3SmokeTests(unittest.TestCase):
    def fixture(self, root: Path, available: int = 16, caps: int = 0):
        predictions = root / "predictions.jsonl"
        rows = []
        for index in range(16):
            rows.append({
                "index": index,
                "completion": "A" if index < available else "reasoning only",
                "strict_final_answer_available": index < available,
                "reached_generation_cap": index < caps,
            })
        predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
        digest = hashlib.sha256(predictions.read_bytes()).hexdigest()
        metrics = root / "metrics.json"
        metrics.write_text(json.dumps({
            "status": "completed", "task": "omnimedvqa", "split_role": "adapter_smoke",
            "count": 16, "generation_contract": "omnimed_corrective_v3_192",
            "max_new_tokens": 192, "data_sha256": "d" * 64,
            "model_config_sha256": "m" * 64, "predictions_sha256": digest,
            "primary_metric": "strict_final_option_accuracy",
        }))
        return metrics, predictions

    def test_passes_without_accuracy_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics, predictions = self.fixture(Path(directory), available=12, caps=4)
            result = verify(metrics, predictions, "d" * 64, "m" * 64)
            self.assertEqual(result["status"], "passed")
            self.assertFalse(result["accuracy_used_as_gate"])

    def test_low_strict_coverage_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics, predictions = self.fixture(Path(directory), available=11)
            with self.assertRaisesRegex(ValueError, "strict_final_answer_coverage"):
                verify(metrics, predictions, "d" * 64, "m" * 64)


if __name__ == "__main__":
    unittest.main()
