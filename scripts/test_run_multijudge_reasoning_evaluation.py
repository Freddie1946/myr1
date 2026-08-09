#!/usr/bin/env python3

import json
import unittest
from pathlib import Path

from run_multijudge_reasoning_evaluation import freeze_panel, parse_scores, response_cost


class MultiJudgeReasoningEvaluationTests(unittest.TestCase):
    def test_parse_scores(self):
        value = parse_scores(json.dumps({
            "r_acc": 0.8, "k_acc": 0.7, "rigor": 0.6, "professionalism": 1,
            "clarity": 0.9, "conciseness": 0.75, "overall_reason": "Consistent reasoning."
        }))
        self.assertEqual(value["r_acc"], 0.8)

    def test_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            parse_scores(json.dumps({
                "r_acc": 1.2, "k_acc": 0.7, "rigor": 0.6, "professionalism": 1,
                "clarity": 0.9, "conciseness": 0.75, "overall_reason": "Bad score."
            }))

    def test_panel_is_model_output_independent(self):
        rows_a = [{"index": i, "source_record_sha256": f"h{i}", "completion": "a"} for i in range(20)]
        rows_b = [{"index": i, "source_record_sha256": f"h{i}", "completion": "different"} for i in range(20)]
        panel = freeze_panel({"a": rows_a, "b": rows_b}, count=10, repeat_count=3, seed=42)
        self.assertEqual(len(panel["indices"]), 10)
        self.assertTrue(set(panel["repeat_indices"]).issubset(panel["indices"]))

    def test_cost(self):
        self.assertAlmostEqual(
            response_cost({"prompt_tokens": 1000, "completion_tokens": 100}, "gpt-4o-2024-08-06"),
            0.0035,
        )


if __name__ == "__main__":
    unittest.main()
