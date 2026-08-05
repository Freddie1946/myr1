import unittest

from compute_external_roi_overlap import mask_area, overlap
from run_external_roi_annotations import parse_json_object, validate_annotation
from run_external_heatmap_critique import validate as validate_critique
from run_attention_visualization_pilot import normalize_attention, option_margin
from run_external_attention_method_critique import validate as validate_method_critique


class ExternalRoiToolsTest(unittest.TestCase):
    def test_validate_box_annotation(self):
        value = validate_annotation(
            {
                "evidence_type": "focal",
                "boxes": [
                    {"x_min": 10, "y_min": 20, "x_max": 300, "y_max": 400, "label": "vessel"}
                ],
                "confidence": 0.8,
                "reason": "Visible vessel.",
            }
        )
        self.assertEqual(value["boxes"][0]["label"], "vessel")

    def test_reject_diffuse_with_box(self):
        with self.assertRaises(ValueError):
            validate_annotation(
                {
                    "evidence_type": "diffuse",
                    "boxes": [
                        {"x_min": 0, "y_min": 0, "x_max": 1000, "y_max": 1000, "label": "all"}
                    ],
                    "confidence": 0.5,
                    "reason": "Diffuse.",
                }
            )

    def test_parse_fenced_json(self):
        self.assertEqual(parse_json_object('```json\n{"a": 1}\n```'), {"a": 1})

    def test_parse_json_after_explanation(self):
        self.assertEqual(parse_json_object('Result follows:\n{"a": 1}\nDone.'), {"a": 1})

    def test_mask_area_handles_overlap(self):
        self.assertAlmostEqual(mask_area([(0, 0, 0.5, 1), (0.25, 0, 0.75, 1)]), 0.75)

    def test_exact_patch_overlap(self):
        result = overlap([(0, 0, 1 / 6, 1 / 6)], [0])
        self.assertAlmostEqual(result["iou"], 1.0)
        self.assertTrue(result["pointing_game_hit"])

    def test_validate_critique(self):
        value = validate_critique(
            {
                "assessment": "partially_reasonable",
                "alignment_rating": 2,
                "direct_visual_evidence": "vessels",
                "heatmap_alignment": "partial",
                "patch_resolution_comment": "coarse",
                "artifact_risk": "moderate",
                "decision_fidelity_caveat": "target only",
            }
        )
        self.assertEqual(value["alignment_rating"], 2)

    def test_normalize_attention(self):
        values = normalize_attention(__import__("numpy").array([1.0, 3.0]))
        self.assertAlmostEqual(float(values.sum()), 1.0)
        self.assertAlmostEqual(float(values[1]), 0.75)

    def test_option_margin(self):
        import torch

        logits = torch.tensor([0.0, 2.0, 0.0, 0.0])
        self.assertAlmostEqual(float(option_margin(logits, 1)), 2.0 - __import__("math").log(3.0))

    def test_validate_method_critique(self):
        value = validate_method_critique(
            {
                "direct_visual_evidence": "two vessels",
                "methods": {
                    name: {"alignment_rating": 2, "assessment": "partial"}
                    for name in ("occlusion_6x6", "raw_attention", "gradient_attention", "relevance_rollout")
                },
                "best_method": "raw_attention",
                "overall_verdict": "mixed",
                "limitations": "coarse maps",
            }
        )
        self.assertEqual(value["best_method"], "raw_attention")

    def test_validate_method_critique_with_blog_attention(self):
        methods = (
            "occlusion_6x6",
            "raw_attention",
            "blog_text_attention",
            "gradient_attention",
            "relevance_rollout",
        )
        value = validate_method_critique(
            {
                "direct_visual_evidence": "a vessel containing red blood cells",
                "methods": {
                    name: {"alignment_rating": 2, "assessment": "partial"}
                    for name in methods
                },
                "best_method": "blog_text_attention",
                "overall_verdict": "mixed",
                "limitations": "external visual review is subjective",
            },
            methods,
        )
        self.assertEqual(value["best_method"], "blog_text_attention")


if __name__ == "__main__":
    unittest.main()
