#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_aigcbest_hosted_baseline_smoke import (
    SmokeFailure, load_fixed_cases, make_payload, score_case
)


class HostedBaselineSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_fixed_cases()

    def test_fixed_cases_cover_all_behavioral_paths(self):
        self.assertEqual([row["case_id"] for row in self.cases], [
            "pathmmu_0", "pathvqa_1", "pathvqa_3", "omnimedvqa_0"
        ])
        self.assertEqual(self.cases[1]["record"]["answer_type"], "free_form")
        self.assertEqual(self.cases[2]["record"]["answer_type"], "yes_no")

    def test_payload_contains_image_and_frozen_prompt(self):
        payload = make_payload("example-model", self.cases[0])
        self.assertEqual(payload["model"], "example-model")
        self.assertEqual(payload["temperature"], 0)
        content = payload["messages"][0]["content"]
        self.assertEqual([item["type"] for item in content], ["image_url", "text"])
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/"))
        self.assertEqual(content[1]["text"], self.cases[0]["prompt"])

    def test_all_local_scoring_paths_execute(self):
        completions = {
            "pathmmu": "<think>brief</think><answer>C</answer>",
            "pathvqa": "positively charged",
            "omnimedvqa": "D",
        }
        for case in self.cases:
            score = score_case(case, completions[case["task"]])
            self.assertIsInstance(score, dict)
        free_form = score_case(self.cases[1], completions["pathvqa"])
        self.assertTrue(free_form["semantic_judge_required_for_full_score"])

    def test_failure_evidence_is_attached(self):
        failure = SmokeFailure("terminal", {"http_status": 404, "raw_response": {"error": {}}})
        self.assertEqual(failure.evidence["http_status"], 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
