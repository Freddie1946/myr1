#!/usr/bin/env python3

import unittest

from freeze_pathvqa_yesno_calibration_panel import select_panel


class FreezePathvqaYesNoCalibrationPanelTests(unittest.TestCase):
    def test_balanced_unique_and_deterministic(self):
        records = []
        for answer in ("yes", "no"):
            for index in range(8):
                records.append({
                    "answer": answer,
                    "image_sha256": f"{answer}-{index}",
                    "source_index": len(records),
                })
        first = select_panel(records, per_answer=4, seed=42)
        second = select_panel(records, per_answer=4, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(sum(row["answer"] == "yes" for row in first), 4)
        self.assertEqual(sum(row["answer"] == "no" for row in first), 4)
        self.assertEqual(len({row["image_sha256"] for row in first}), 8)


if __name__ == "__main__":
    unittest.main()
