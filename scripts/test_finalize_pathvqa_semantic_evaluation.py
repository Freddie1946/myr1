#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from finalize_pathvqa_semantic_evaluation import (
    IncompleteEvaluationError,
    finalize,
)
from pathvqa_llm_judge import PROMPT_VERSION, cache_key


def write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class FinalizePathVQATests(unittest.TestCase):
    def fixtures(self, root: Path):
        predictions = [
            {
                "index": 0,
                "answer_type": "yes_no",
                "status": "passed",
                "source_record_sha256": "0" * 64,
                "score": {
                    "answer_type": "yes_no",
                    "contract_aligned_exact_match": True,
                },
            },
            {
                "index": 1,
                "answer_type": "yes_no",
                "status": "failed",
                "source_record_sha256": "1" * 64,
            },
            {
                "index": 2,
                "answer_type": "free_form",
                "status": "passed",
                "source_record_sha256": "2" * 64,
                "question": "what?",
                "answer": "histones",
                "completion": "Histones",
                "score": {"answer_type": "free_form", "strict_exact_match": True},
            },
            {
                "index": 3,
                "answer_type": "free_form",
                "status": "passed",
                "source_record_sha256": "3" * 64,
                "question": "what?",
                "answer": "histones",
                "completion": "Histones",
                "score": {"answer_type": "free_form", "strict_exact_match": True},
            },
            {
                "index": 4,
                "answer_type": "free_form",
                "status": "failed",
                "source_record_sha256": "4" * 64,
                "question": "failed question?",
                "answer": "reference",
                "completion": "stale completion",
            },
        ]
        key = cache_key("what?", "histones", "Histones", "gpt-5-mini")
        judgments = [
            {
                "schema_version": 1,
                "prompt_version": PROMPT_VERSION,
                "cache_key": key,
                "index": 2,
                "source_record_sha256": "2" * 64,
                "answer_type": "free_form",
                "candidate": "Histones",
                "candidate_extraction_source": "raw_completion",
                "normalized_reference": "histones",
                "judgment": {
                    "correct": True,
                    "reason": "Equivalent.",
                    "error_type": "correct",
                },
                "model": "gpt-5-mini",
                "served_model": "gpt-5-mini-2025-08-07",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "attempts": [{"attempt": 1, "status": "completed"}],
            }
        ]
        failed_key = cache_key(
            "failed question?", "reference", "stale completion", "gpt-5-mini"
        )
        judgments.append(
            {
                "schema_version": 1,
                "prompt_version": PROMPT_VERSION,
                "cache_key": failed_key,
                "index": 4,
                "source_record_sha256": "4" * 64,
                "answer_type": "free_form",
                "candidate": "stale completion",
                "candidate_extraction_source": "raw_completion",
                "normalized_reference": "reference",
                "judgment": {
                    "correct": False,
                    "reason": "Historical verdict.",
                    "error_type": "contradiction",
                },
                "model": "gpt-5-mini",
                "served_model": "gpt-5-mini-2025-08-07",
                "usage": {},
                "attempts": [],
            }
        )
        skipped = [
            {
                "schema_version": 1,
                "index": 4,
                "source_record_sha256": "4" * 64,
                "prediction_status": "failed",
                "reason": "unavailable",
                "model": "gpt-5-mini",
            }
        ]
        metrics = {
            "schema_version": 1,
            "formal_result": True,
            "model": "model-under-test",
            "expected_count": 5,
            "recorded_count": 5,
            "passed_count": 3,
            "failed_count": 2,
            "yes_no_count": 1,
            "yes_no_contract_aligned_correct": 1,
            "free_form_count": 2,
        }
        names = (
            "predictions.jsonl",
            "judgments.jsonl",
            "skipped.jsonl",
            "failures.jsonl",
            "metrics.json",
            "semantic_metrics.json",
        )
        paths = {name: root / name for name in names}
        write_jsonl(paths["predictions.jsonl"], predictions)
        write_jsonl(paths["judgments.jsonl"], judgments)
        write_jsonl(paths["skipped.jsonl"], skipped)
        paths["metrics.json"].write_text(json.dumps(metrics), encoding="utf-8")
        return paths, judgments

    def run_finalize(self, paths):
        return finalize(
            predictions_path=paths["predictions.jsonl"],
            judgments_path=paths["judgments.jsonl"],
            skipped_path=paths["skipped.jsonl"],
            failures_path=paths["failures.jsonl"],
            source_metrics_path=paths["metrics.json"],
            output_path=paths["semantic_metrics.json"],
            judge_model="gpt-5-mini",
        )

    def test_duplicate_semantics_share_judgment_but_score_per_case(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, _ = self.fixtures(Path(directory))
            result = self.run_finalize(paths)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["total_correct"], 3)
            self.assertEqual(result["overall_accuracy"], 0.6)
            self.assertEqual(result["free_form"]["total_cases"], 3)
            self.assertEqual(result["free_form"]["unique_semantic_cache_keys"], 1)
            self.assertEqual(result["free_form"]["passed_prediction_cases_judged"], 2)
            self.assertEqual(result["free_form"]["correct"], 2)
            self.assertEqual(result["semantic_judge"]["unique_judgment_records"], 2)
            self.assertEqual(result["semantic_judge"]["used_unique_judgments"], 1)
            self.assertEqual(
                result["semantic_judge"][
                    "excluded_historical_judgments_for_failed_predictions"
                ],
                1,
            )
            self.assertEqual(
                result["semantic_judge"]["usage_totals_over_unique_judgments"][
                    "total_tokens"
                ],
                15,
            )
            self.assertTrue(paths["semantic_metrics.json"].is_file())

    def test_missing_required_cache_key_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, _ = self.fixtures(Path(directory))
            write_jsonl(paths["judgments.jsonl"], [])
            with self.assertRaises(IncompleteEvaluationError):
                self.run_finalize(paths)
            self.assertFalse(paths["semantic_metrics.json"].exists())

    def test_served_model_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, judgments = self.fixtures(Path(directory))
            judgments[0]["served_model"] = "redirected-model"
            write_jsonl(paths["judgments.jsonl"], judgments)
            with self.assertRaisesRegex(ValueError, "served model mismatch"):
                self.run_finalize(paths)


if __name__ == "__main__":
    unittest.main()
