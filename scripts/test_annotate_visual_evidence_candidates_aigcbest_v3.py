#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotate_visual_evidence_candidates_aigcbest_v3 import parse_json_object, validate


class TestExhaustiveAnnotation(unittest.TestCase):
    def test_multifocal_annotation(self):
        value = {
            "evidence_type": "multifocal",
            "regions": [
                {"box": [1, 2, 10, 20], "feature": "focus one", "role": "supports_reference", "importance": .9},
                {"box": [30, 40, 50, 60], "feature": "focus two", "role": "opposes_distractor", "importance": .7},
            ],
            "diffuse_evidence": [], "visual_answerability": "high", "strongest_distractor": "B",
            "coverage_complete": True, "coverage_note": "whole field reviewed", "confidence": .9,
            "rationale": "Both foci matter.",
        }
        self.assertEqual(len(validate(value)["regions"]), 2)

    def test_type_is_normalized_from_content(self):
        value = {
            "evidence_type": "mixed", "regions": [], "diffuse_evidence": ["architecture"],
            "visual_answerability": "high", "strongest_distractor": "B", "coverage_complete": True,
            "coverage_note": "reviewed", "confidence": .9, "rationale": "mixed",
        }
        result = validate(value)
        self.assertEqual(result["evidence_type"], "diffuse")
        self.assertTrue(result["evidence_type_normalized"])

    def test_fenced_json(self):
        self.assertEqual(parse_json_object('```json\n{"x": 1}\n```'), {"x": 1})

    def test_near_whole_image_box_is_rejected(self):
        value = {
            "evidence_type": "focal",
            "regions": [{"box": [0, 0, 999, 999], "feature": "diffuse field",
                         "role": "context", "importance": .5}],
            "diffuse_evidence": [], "visual_answerability": "high", "strongest_distractor": "B",
            "coverage_complete": True, "coverage_note": "reviewed", "confidence": .9,
            "rationale": "field-wide evidence",
        }
        with self.assertRaises(ValueError):
            validate(value)

    def test_float_coordinates_are_canonicalized(self):
        value = {
            "evidence_type": "focal",
            "regions": [{"box": [10.2, 20.0, 40.8, 60.1], "feature": "focus",
                         "role": "supports_reference", "importance": .9}],
            "diffuse_evidence": [], "visual_answerability": "high", "strongest_distractor": "B",
            "coverage_complete": True, "coverage_note": "reviewed", "confidence": .9,
            "rationale": "localized evidence",
        }
        self.assertEqual(validate(value)["regions"][0]["box"], [10, 20, 41, 60])


if __name__ == "__main__":
    unittest.main()
