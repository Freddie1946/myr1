#!/usr/bin/env python3

import argparse
import unittest
from pathlib import Path

from rescore_pathvqa_yesno_v3 import (
    legacy_v2_yes_no_correct,
    parse_named_path,
    select_yes_no,
)


class RescorePathVqaYesNoV3Tests(unittest.TestCase):
    def test_named_path(self):
        self.assertEqual(parse_named_path("model=/tmp/p.jsonl"), ("model", Path("/tmp/p.jsonl")))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_named_path("missing_separator")

    def test_exact_yes_no_count_is_required(self):
        rows = [
            {"answer_type": "yes_no", "answer": "yes"} for _ in range(3362)
        ] + [{"answer_type": "free_form", "answer": "liver"}]
        self.assertEqual(len(select_yes_no(rows)), 3362)
        with self.assertRaisesRegex(ValueError, "expected 3362"):
            select_yes_no(rows[:-2])

    def test_legacy_parser_does_not_decode_choice_prefix_inside_tag(self):
        self.assertFalse(legacy_v2_yes_no_correct("<answer>B) No</answer>", "no"))
        self.assertTrue(legacy_v2_yes_no_correct("The answer is no.", "no"))


if __name__ == "__main__":
    unittest.main()
