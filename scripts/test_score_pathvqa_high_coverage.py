#!/usr/bin/env python3
from __future__ import annotations

import unittest

from score_pathvqa_high_coverage import combine


class ScorePathvqaHighCoverageTests(unittest.TestCase):
    def test_verified_recovery_and_rejection(self) -> None:
        predictions = [
            {"index": 0, "answer": "yes", "completion": "Yes", "source_record_sha256": "a"},
            {"index": 1, "answer": "no", "completion": "finding absent", "source_record_sha256": "b"},
            {"index": 2, "answer": "yes", "completion": "conflict", "source_record_sha256": "c"},
        ]
        judgments = [
            {"index": 1, "judgment": {"answer": "no", "evidence_span": "absent", "reason_code": "entailed_conclusion"}},
            {"index": 2, "judgment": {"answer": "yes", "evidence_span": "conflict", "reason_code": "entailed_conclusion"}},
        ]
        verifications = [
            {"index": 1, "verdict": {"verdict": "accept", "contradiction_span": "", "reason_code": "consistent"}},
            {"index": 2, "verdict": {"verdict": "reject", "contradiction_span": "conflict", "reason_code": "material_contradiction"}},
        ]
        rows, metrics = combine(predictions, judgments, verifications)
        self.assertEqual(metrics["resolved_count"], 2)
        self.assertEqual(metrics["correct"], 2)
        self.assertFalse(rows[2]["resolved"])


if __name__ == "__main__":
    unittest.main()
