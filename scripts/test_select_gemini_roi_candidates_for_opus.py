#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_gemini_roi_candidates_for_opus import candidate_metrics


class TestGeminiRoiSelection(unittest.TestCase):
    def test_eligible_multifocal(self):
        annotation = {"regions": [
            {"box": [10, 10, 100, 100], "role": "supports_reference", "importance": .9},
            {"box": [200, 200, 300, 300], "role": "supports_reference", "importance": .8}],
            "coverage_complete": True, "visual_answerability": "high", "confidence": .9,
            "evidence_type": "multifocal"}
        result = candidate_metrics(annotation)
        self.assertTrue(result["eligible"])
        self.assertEqual(result["direct_region_count"], 2)

    def test_excludes_overbroad_mask(self):
        annotation = {"regions": [
            {"box": [0, 0, 700, 700], "role": "supports_reference", "importance": .9}],
            "coverage_complete": True, "visual_answerability": "high", "confidence": .9,
            "evidence_type": "focal"}
        self.assertFalse(candidate_metrics(annotation)["eligible"])


if __name__ == "__main__":
    unittest.main()
