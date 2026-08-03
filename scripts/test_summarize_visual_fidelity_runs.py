#!/usr/bin/env python3
from __future__ import annotations

import unittest

from summarize_visual_fidelity_runs import ARM_ORDER, validate_arm_labels


class SummarizeVisualFidelityRunsTests(unittest.TestCase):
    def test_accepts_exact_five_arm_set_in_any_order(self) -> None:
        validate_arm_labels(list(reversed(ARM_ORDER)))

    def test_rejects_missing_or_duplicate_arm(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_arm_labels(list(ARM_ORDER[:-1]))
        with self.assertRaisesRegex(ValueError, "duplicated"):
            validate_arm_labels([*ARM_ORDER[:-1], ARM_ORDER[0]])


if __name__ == "__main__":
    unittest.main()
