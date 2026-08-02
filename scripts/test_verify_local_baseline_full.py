#!/usr/bin/env python3

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from verify_local_baseline_full import verify


class VerifyLocalBaselineFullTests(unittest.TestCase):
    def make_artifacts(self, root: Path, *, bad_hash: bool = False) -> tuple[Path, Path, Path]:
        predictions = root / "predictions.jsonl"
        rows = [
            {"index": index, "source_record_sha256": f"{index:064x}"}
            for index in range(2)
        ]
        predictions.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        digest = hashlib.sha256(predictions.read_bytes()).hexdigest()
        common = {
            "status": "completed",
            "task": "pathvqa",
            "split_role": "external_test",
            "count": 2,
            "data_sha256": "a" * 64,
            "model_config_sha256": "b" * 64,
            "predictions_sha256": "0" * 64 if bad_hash else digest,
        }
        metrics = root / "metrics.json"
        metrics.write_text(json.dumps(common), encoding="utf-8")
        config = root / "run_config.json"
        config.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "task": "pathvqa",
                    "split_role": "external_test",
                    "selected_count": 2,
                    "data_sha256": "a" * 64,
                    "model_config_sha256": "b" * 64,
                }
            ),
            encoding="utf-8",
        )
        return metrics, predictions, config

    def test_valid_external_full_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metrics, predictions, config = self.make_artifacts(Path(temporary))
            result = verify(
                task="pathvqa",
                metrics_path=metrics,
                predictions_path=predictions,
                run_config_path=config,
                expected_count=2,
                expected_split_role="external_test",
                expected_data_sha256="a" * 64,
                expected_model_config_sha256="b" * 64,
            )
            self.assertEqual(result["status"], "passed")

    def test_prediction_hash_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metrics, predictions, config = self.make_artifacts(
                Path(temporary), bad_hash=True
            )
            with self.assertRaisesRegex(ValueError, "contract mismatch"):
                verify(
                    task="pathvqa",
                    metrics_path=metrics,
                    predictions_path=predictions,
                    run_config_path=config,
                    expected_count=2,
                    expected_split_role="external_test",
                    expected_data_sha256="a" * 64,
                    expected_model_config_sha256="b" * 64,
                )


if __name__ == "__main__":
    unittest.main()
