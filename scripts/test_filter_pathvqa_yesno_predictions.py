#!/usr/bin/env python3
from __future__ import annotations

import unittest

from filter_pathvqa_yesno_predictions import select


class FilterPathvqaYesNoPredictionsTests(unittest.TestCase):
    def test_selects_and_reindexes(self) -> None:
        rows = [{"index": 0, "answer_type": "free_form", "answer": "cell"}]
        rows += [{"index": index + 1, "answer_type": "yes_no", "answer": "yes"} for index in range(3362)]
        selected = select(rows)
        self.assertEqual(selected[0]["index"], 0)
        self.assertEqual(selected[0]["source_index"], 1)
        self.assertEqual(selected[-1]["index"], 3361)


if __name__ == "__main__":
    unittest.main()
