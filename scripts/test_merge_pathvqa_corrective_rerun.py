#!/usr/bin/env python3
from __future__ import annotations

import unittest

from merge_pathvqa_corrective_rerun import merge_rows


class MergePathvqaCorrectiveRerunTests(unittest.TestCase):
    def test_merge_requires_prefix_and_cap(self) -> None:
        original = [
            {
                "index": index,
                "source_record_sha256": str(index),
                "completion": "old" if index == 3 else "done",
                "generated_token_count": 64,
                "reached_generation_cap": index == 3,
            }
            for index in range(3362)
        ]
        correction = [{
            "index": 0,
            "source_record_sha256": "3",
            "completion": "old continued",
            "generated_token_count": 80,
            "reached_generation_cap": False,
        }]
        merged = merge_rows(original, correction)
        self.assertEqual(merged[3]["index"], 3)
        self.assertTrue(merged[3]["corrective_rerun"]["applied"])
        self.assertEqual(merged[3]["completion"], "old continued")

    def test_prefix_mismatch_fails(self) -> None:
        original = [
            {"index": index, "source_record_sha256": str(index), "completion": "old", "generated_token_count": 64, "reached_generation_cap": index == 3}
            for index in range(3362)
        ]
        with self.assertRaisesRegex(ValueError, "prefix mismatch"):
            merge_rows(original, [{"source_record_sha256": "3", "completion": "different"}])
        merged = merge_rows(
            original, [{"source_record_sha256": "3", "completion": "different", "generated_token_count": 2}],
            require_prefix=False,
        )
        self.assertFalse(merged[3]["corrective_rerun"]["old_completion_is_prefix"])


if __name__ == "__main__":
    unittest.main()
