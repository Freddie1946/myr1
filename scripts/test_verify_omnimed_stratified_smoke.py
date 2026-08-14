#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from verify_omnimed_stratified_smoke import EXPECTED_SOURCES, output_faults, verify


class VerifyOmniMedStratifiedSmokeTests(unittest.TestCase):
    def fixture(self, root: Path, *, bad_completion: str | None = None, unparseable: bool = False):
        rows = []
        index = 0
        for source, count in EXPECTED_SOURCES.items():
            for _ in range(count):
                completion = "<think>brief</think><answer>A</answer>"
                if index == 0 and bad_completion is not None:
                    completion = bad_completion
                rows.append({
                    "index": index,
                    "dataset": source,
                    "source_record_sha256": "a" * 64,
                    "completion": completion,
                    "strict_final_answer_available": not (unparseable and index == 0),
                    "reached_generation_cap": False,
                })
                index += 1
        predictions = root / "predictions.jsonl"
        predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
        metrics = root / "metrics.json"
        metrics.write_text(json.dumps({
            "status": "completed",
            "task": "omnimedvqa",
            "split_role": "adapter_smoke",
            "count": 64,
            "selected_count": 64,
            "generation_contract": "omnimed_domain_think_answer_v4_1024",
            "max_new_tokens": 1024,
            "data_sha256": "d" * 64,
            "model_config_sha256": "m" * 64,
            "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
        }))
        return metrics, predictions

    def test_clean_balanced_panel_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics, predictions = self.fixture(Path(directory))
            result = verify(
                metrics, predictions,
                expected_data_sha256="d" * 64,
                expected_model_config_sha256="m" * 64,
            )
            self.assertEqual(result["status"], "passed")

    def test_unicode_format_and_loop_are_detected(self) -> None:
        faults = output_faults("A \u00ad A \u00ad A \u00ad A \u00ad A \u00ad A \u00ad A \u00ad A")
        self.assertTrue(any("unicode_control_or_format" in value for value in faults))
        loop_faults = output_faults("a b " * 12)
        self.assertTrue(any("repeated" in value for value in loop_faults))

    def test_garbled_output_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics, predictions = self.fixture(
                Path(directory), bad_completion="word word word word word word word word word"
            )
            with self.assertRaisesRegex(ValueError, "garbled_or_repetitive_outputs"):
                verify(
                    metrics, predictions,
                    expected_data_sha256="d" * 64,
                    expected_model_config_sha256="m" * 64,
                )


if __name__ == "__main__":
    unittest.main()
