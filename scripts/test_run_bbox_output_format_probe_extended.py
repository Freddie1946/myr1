#!/usr/bin/env python3

import unittest

from run_bbox_output_format_probe_extended import extract_answer, extract_payload, valid_payload


class ExtendedFormatProbeTests(unittest.TestCase):
    def test_bbox_attributes(self):
        text = '<evidence type="focal"><bbox x_min="1" y_min="2" x_max="30" y_max="40"/></evidence>'
        kind, boxes = extract_payload(text)
        self.assertEqual(kind, "focal")
        self.assertTrue(valid_payload(kind, boxes))

    def test_answer_attributes(self):
        self.assertEqual(extract_answer('<answer choice="C"/>', None), "C")

    def test_nested_answer(self):
        self.assertEqual(extract_answer('<answer><choice>B</choice></answer>', None), "B")

    def test_plain_coordinates(self):
        kind, boxes = extract_payload('VISUAL_EVIDENCE: multifocal; boxes=[[1,2,30,40]]')
        self.assertEqual(kind, "multifocal")
        self.assertEqual(len(boxes), 1)

    def test_plain_coordinates_extract_every_box(self):
        kind, boxes = extract_payload(
            'VISUAL_EVIDENCE: focal; boxes=[[345,167,398,219],[100,167,153,219]]'
        )
        self.assertEqual(kind, "focal")
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[1]["x_min"], 100.0)

    def test_followup_fallback(self):
        self.assertEqual(extract_answer('<bbox>none</bbox>', 'A'), 'A')


if __name__ == "__main__":
    unittest.main()
