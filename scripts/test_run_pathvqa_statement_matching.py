#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from run_pathvqa_statement_matching import candidate_statements, select_yes_no


class PathVQAStatementMatchingTest(unittest.TestCase):
    def test_answer_precedes_question(self):
        result = candidate_statements("Does the image show necrosis?")
        self.assertEqual(result["yes"], "Answer: Yes. Question: Does the image show necrosis?")
        self.assertEqual(result["no"], "Answer: No. Question: Does the image show necrosis?")
        self.assertLess(result["yes"].index("Yes"), result["yes"].index("Does"))

    def test_filter_and_validate(self):
        with tempfile.NamedTemporaryFile(suffix=".png") as image:
            rows = [
                {"index": 1, "answer_type": "yes_no", "answer": "Yes", "image": image.name},
                {"index": 2, "answer_type": "open_ended", "answer": "tumor", "image": image.name},
            ]
            self.assertEqual([row["index"] for row in select_yes_no(rows)], [1])


if __name__ == "__main__":
    unittest.main()
