#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_aigcbest_hosted_baseline_full import aggregate, build_case


class HostedBaselineFullTests(unittest.TestCase):
    def test_build_case_uses_frozen_task_caps(self):
        record = {
            "image": "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/"
            "external_vqa_contract_v1_20260729/pathvqa_images_by_sha256/"
            "697364a3492fbb1473afc982242084714cedb2f0f2b1c2077af3e80663797c4e.jpg",
            "question": "how are histones charged?",
            "answer": "positively charged",
            "answer_type": "free_form",
            "index": 0,
        }
        case = build_case("pathvqa", 0, record)
        self.assertEqual(case["case_id"], "pathvqa_0")
        self.assertEqual(case["max_tokens"], 128)

    def test_aggregate_keeps_legacy_and_aligned_omni_metrics(self):
        rows = [{
            "status": "passed",
            "score": {
                "official_most_similar_correct": False,
                "contract_aligned_correct": True,
            },
        }]
        metrics = aggregate("omnimedvqa", rows, 1)
        self.assertEqual(metrics["legacy_official_correct"], 0)
        self.assertEqual(metrics["contract_aligned_correct"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
