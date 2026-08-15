#!/usr/bin/env python3

import unittest

from generate_bilingual_expert_review_translations import parse_translation


class TranslationValidationTest(unittest.TestCase):
    def test_valid_preserves_tags_and_options(self):
        source = {"x": "Question\nA) One\nB) Two", "y": "<think>Reason</think><answer>A)</answer>"}
        translated = "<<<TRANSLATION:x>>>\n问题\nA) 一\nB) 二\n<<<END:x>>>\n<<<TRANSLATION:y>>>\n<think>理由</think><answer>A)</answer>\n<<<END:y>>>"
        self.assertEqual(parse_translation(translated, source)["x"], "问题\nA) 一\nB) 二")

    def test_rejects_missing_tag(self):
        source = {"x": "<think>Reason</think>"}
        with self.assertRaises(ValueError):
            parse_translation("<<<TRANSLATION:x>>>\n理由\n<<<END:x>>>", source)


if __name__ == "__main__":
    unittest.main()
