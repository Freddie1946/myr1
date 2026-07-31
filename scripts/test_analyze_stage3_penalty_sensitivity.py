#!/usr/bin/env python3
"""Unit tests for the offline Stage3 penalty sensitivity analysis."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import analyze_stage3_penalty_sensitivity as analysis


class SensitivityTest(unittest.TestCase):
    def test_score_clipping(self) -> None:
        self.assertEqual(analysis.process_score(0, 0, 0.4), 1.0)
        self.assertEqual(analysis.process_score(3, 3, 0.4), 0.0)
        self.assertAlmostEqual(analysis.process_score(1, 2, 0.3), 0.55)

    def test_spearman_ties_and_constant(self) -> None:
        self.assertAlmostEqual(analysis.spearman([0, 1, 1], [0, 2, 2]), 1.0)
        self.assertIsNone(analysis.spearman([1, 1], [0, 1]))

    def test_load_excludes_smoke_and_group_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache" / "aa"
            root.mkdir(parents=True)
            for index, (missing, errors, smoke) in enumerate(
                [(0, 0, False), (1, 0, False), (2, 1, True)]
            ):
                value = {
                    "cache_key": f"key-{index}",
                    "scores": {
                        "missing_integrity_count": missing,
                        "present_knowledge_error_count": errors,
                    },
                    "source": {
                        "smoke": smoke,
                        "training_segment": "segment",
                        "record_index": 7,
                        "problem_sha256": "problem",
                    },
                }
                (root / f"{index}.json").write_text(json.dumps(value), encoding="utf-8")
            records, smoke_count = analysis.load_records(root.parent)
            self.assertEqual(len(records), 2)
            self.assertEqual(smoke_count, 1)
            values = [analysis.process_score(row["missing"], row["errors"], 0.3) for row in records]
            summary = analysis.summarize_groups(records, values, values)
            self.assertEqual(summary["eligible_group_count_size_ge_2"], 1)
            self.assertEqual(summary["centered_advantage_sign_agreement_with_reference"], 1.0)
            self.assertEqual(summary["mean_top_reward_set_jaccard_with_reference"], 1.0)


if __name__ == "__main__":
    unittest.main()
