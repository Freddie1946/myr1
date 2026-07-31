#!/usr/bin/env python3

import unittest

from pathvqa_llm_judge import cache_key, parse_judgment


class PathVQALLMJudgeTests(unittest.TestCase):
    def test_valid_correct_and_error(self):
        self.assertTrue(parse_judgment(
            '{"correct":true,"reason":"Equivalent.","error_type":"correct"}'
        )["correct"])
        self.assertFalse(parse_judgment(
            '{"correct":false,"reason":"Does not answer.","error_type":"omission"}'
        )["correct"])

    def test_fenced_json_is_accepted_but_extra_fields_are_not(self):
        value = parse_judgment(
            '```json\n{"correct":false,"reason":"Related only.",'
            '"error_type":"related_but_not_answer"}\n```'
        )
        self.assertFalse(value["correct"])
        with self.assertRaises(ValueError):
            parse_judgment(
                '{"correct":true,"reason":"x","error_type":"correct","score":1}'
            )

    def test_boolean_and_error_type_must_agree(self):
        with self.assertRaises(ValueError):
            parse_judgment(
                '{"correct":true,"reason":"x","error_type":"unclear"}'
            )

    def test_cache_key_covers_semantic_inputs(self):
        base = cache_key("q", "a", "c", "m")
        self.assertEqual(base, cache_key("q", "a", "c", "m"))
        self.assertNotEqual(base, cache_key("q2", "a", "c", "m"))
        self.assertNotEqual(base, cache_key("q", "a2", "c", "m"))
        self.assertNotEqual(base, cache_key("q", "a", "c2", "m"))
        self.assertNotEqual(base, cache_key("q", "a", "c", "m2"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
