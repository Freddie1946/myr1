#!/usr/bin/env python3

import unittest

from run_direct_bbox_self_report import (
    SYSTEM_PROMPT,
    answer_from_completion,
    extract_json,
    overlap_metrics,
    validate_annotation,
)


class DirectBboxTests(unittest.TestCase):
    def test_parse_fenced_json(self):
        value = extract_json('```json\n{"answer":"A","boxes":[]}\n```')
        self.assertEqual(value["answer"], "A")

    def test_valid_focal(self):
        value = validate_annotation({
            "answer": "b", "evidence_type": "focal",
            "boxes": [{"x_min": 10, "y_min": 20, "x_max": 100, "y_max": 200}],
            "reason": "cells",
        })
        self.assertEqual(value["answer"], "B")

    def test_diffuse_forbids_boxes(self):
        with self.assertRaises(ValueError):
            validate_annotation({
                "answer": "A", "evidence_type": "diffuse",
                "boxes": [{"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 100}],
                "reason": "distributed",
            })

    def test_overlap_identity(self):
        box = [{"x_min": 100, "y_min": 200, "x_max": 500, "y_max": 600}]
        metrics = overlap_metrics(box, box)
        self.assertEqual(metrics["union_iou"], 1.0)
        self.assertEqual(metrics["predicted_area_precision"], 1.0)
        self.assertEqual(metrics["reference_area_recall"], 1.0)

    def test_system_prompt_forbids_training_xml(self):
        self.assertIn("Never emit think/answer XML tags", SYSTEM_PROMPT)

    def test_trailing_xml_does_not_hide_json_object(self):
        value = extract_json(
            '{"answer":"B","evidence_type":"focal","boxes":[],"reason":"x"}'
            '<answer>B</answer>'
        )
        self.assertEqual(value["answer"], "B")

    def test_extract_frozen_answer(self):
        self.assertEqual(answer_from_completion("<answer>C) text</answer>"), "C")


if __name__ == "__main__":
    unittest.main()
