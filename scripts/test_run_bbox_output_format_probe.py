#!/usr/bin/env python3

import unittest

from run_bbox_output_format_probe import extract_boxes, valid_payload


class BboxFormatProbeTests(unittest.TestCase):
    def test_xml(self):
        text = '<evidence type="focal"><bbox x_min="1" y_min="2" x_max="30" y_max="40" label="x"/></evidence>'
        kind, boxes = extract_boxes(text, "xml_bbox_in_think")
        self.assertEqual(kind, "focal")
        self.assertTrue(valid_payload(kind, boxes))

    def test_qwen_box(self):
        text = 'evidence_type=multifocal <ref>x</ref><box>(1,2),(30,40)</box>'
        kind, boxes = extract_boxes(text, "qwen_ref_box_in_think")
        self.assertEqual(kind, "multifocal")
        self.assertEqual(boxes[0]["label"], "x")

    def test_plain_nonspatial(self):
        kind, boxes = extract_boxes(
            'VISUAL_EVIDENCE: diffuse; boxes=[]', "plain_bbox_in_think"
        )
        self.assertTrue(valid_payload(kind, boxes))

    def test_json(self):
        text = '<evidence_json>{"evidence_type":"focal","boxes":[{"x_min":1,"y_min":2,"x_max":30,"y_max":40,"label":"x"}]}</evidence_json>'
        kind, boxes = extract_boxes(text, "evidence_json_in_think")
        self.assertEqual(kind, "focal")
        self.assertTrue(valid_payload(kind, boxes))


if __name__ == "__main__":
    unittest.main()
