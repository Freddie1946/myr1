#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dual_model_roi_consensus_panel import box_overlap_coefficient, consensus


def ann(boxes):
    return {"regions": [{"box": b, "role": "supports_reference", "importance": .9} for b in boxes],
            "coverage_complete": True, "visual_answerability": "high", "confidence": .9}


class TestConsensus(unittest.TestCase):
    def test_containment_overlap(self):
        self.assertEqual(box_overlap_coefficient([0, 0, 100, 100], [20, 20, 80, 80]), 1.0)

    def test_bidirectional_multiregion_consensus(self):
        result = consensus(ann([[0, 0, 100, 100], [200, 200, 300, 300]]),
                           ann([[10, 10, 90, 90], [210, 210, 290, 290]]))
        self.assertTrue(result["eligible"])
        self.assertEqual(len(result["primary_consensus_regions"]), 2)

    def test_rejects_unmatched(self):
        self.assertFalse(consensus(ann([[0, 0, 100, 100]]), ann([[500, 500, 600, 600]]))["eligible"])

    def test_accepts_one_corroborated_region_despite_different_decomposition(self):
        result = consensus(
            ann([[0, 0, 100, 100], [200, 200, 300, 300], [400, 400, 500, 500]]),
            ann([[10, 10, 90, 90], [700, 700, 800, 800], [850, 850, 900, 900]]),
        )
        self.assertTrue(result["eligible"])
        self.assertEqual(len(result["primary_consensus_regions"]), 1)


if __name__ == "__main__":
    unittest.main()
