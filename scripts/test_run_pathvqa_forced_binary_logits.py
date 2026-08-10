#!/usr/bin/env python3
from __future__ import annotations

import unittest

from run_pathvqa_forced_binary_logits import binary_result, cyclic_image_map


class ForcedBinaryLogitTests(unittest.TestCase):
    def test_yes_and_no_decisions(self) -> None:
        yes = binary_result(2.0, 1.0, "yes")
        no = binary_result(-2.0, 1.0, "no")
        self.assertEqual(yes["forced_binary_answer"], "yes")
        self.assertTrue(yes["forced_binary_correct"])
        self.assertEqual(no["forced_binary_answer"], "no")
        self.assertTrue(no["forced_binary_correct"])
        self.assertGreater(yes["target_margin"], 0)
        self.assertGreater(no["target_margin"], 0)

    def test_cyclic_mismatch_never_preserves_an_image(self) -> None:
        mapping = cyclic_image_map([{"image": "b"}, {"image": "a"}, {"image": "b"}])
        self.assertEqual(mapping, {"a": "b", "b": "a"})


if __name__ == "__main__":
    unittest.main()
