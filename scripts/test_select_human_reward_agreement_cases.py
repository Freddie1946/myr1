#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_human_reward_agreement_cases import process_band, stable_key


class TestHumanRewardSelection(unittest.TestCase):
    def test_process_bands(self):
        self.assertEqual(process_band(1.0), "high_1p0")
        self.assertEqual(process_band(.8), "mid_0p8")
        self.assertEqual(process_band(.6), "low_le_0p6")

    def test_stable_key(self):
        row = {"rank": 1, "call_index": 2, "item_index": 0, "record_index": 3,
               "completion_sha256": "abc"}
        self.assertEqual(stable_key(7, row), stable_key(7, row))
        self.assertNotEqual(stable_key(7, row), stable_key(8, row))


if __name__ == "__main__":
    unittest.main()
