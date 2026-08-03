#!/usr/bin/env python3

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from verify_local_baseline_full import verify


class VerifyLocalBaselineFullTests(unittest.TestCase):
    def make_artifacts(
        self, root: Path, *, bad_hash: bool = False, answer_scope: str | None = None
    ) -> tuple[Path, Path, Path]:
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
        if answer_scope is not None:
            common["answer_scope"] = answer_scope
        metrics = root / "metrics.json"
        metrics.write_text(json.dumps(common), encoding="utf-8")
        config = root / "run_config.json"
        config_value = {
                    "status": "completed",
                    "task": "pathvqa",
                    "split_role": "external_test",
                    "selected_count": 2,
                    "data_sha256": "a" * 64,
                    "model_config_sha256": "b" * 64,
        }
        if answer_scope is not None:
            config_value["answer_scope"] = answer_scope
        config.write_text(json.dumps(config_value), encoding="utf-8")
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

    def test_valid_yes_no_only_pathvqa_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            metrics, predictions, config = self.make_artifacts(
                Path(temporary), answer_scope="yes_no_only"
            )
            result = verify(
                task="pathvqa",
                metrics_path=metrics,
                predictions_path=predictions,
                run_config_path=config,
                expected_count=2,
                expected_split_role="external_test",
                expected_data_sha256="a" * 64,
                expected_model_config_sha256="b" * 64,
                expected_pathvqa_answer_scope="yes_no_only",
            )
            self.assertEqual(result["pathvqa_answer_scope"], "yes_no_only")

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
