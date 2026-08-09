#!/usr/bin/env python3

import json
import unittest

from judge_stage3_badcases_claude import parse


class BadcaseJudgeTests(unittest.TestCase):
    def test_parse(self):
        value = parse(json.dumps({
            "effect": "regression", "before_error": "none",
            "after_error": "visual_feature_misread", "reference_ambiguous": False,
            "image_evidence": "The lesion is gland-forming.", "explanation": "After misreads it."
        }))
        self.assertEqual(value["effect"], "regression")


if __name__ == "__main__":
    unittest.main()
