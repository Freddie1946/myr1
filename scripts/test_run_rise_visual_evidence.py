#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np

from run_rise_visual_evidence import exact_area_mask, rise_soft_masks, shifted_controls


class RiseVisualEvidenceTests(unittest.TestCase):
    def test_antithetic_masks_are_bounded_and_complementary(self) -> None:
        masks = rise_soft_masks(31, 23, count=10, cells=5, probability=0.5, seed=7)
        self.assertEqual(masks.shape, (10, 23, 31))
        self.assertTrue(np.all((masks >= 0) & (masks <= 1)))
        for index in range(0, len(masks), 2):
            self.assertTrue(np.allclose(masks[index] + masks[index + 1], 1.0, atol=1e-6))

    def test_exact_area_high_and_low(self) -> None:
        values = np.arange(100).reshape(10, 10)
        high = exact_area_mask(values, 0.25, high=True)
        low = exact_area_mask(values, 0.25, high=False)
        self.assertEqual(int(high.sum()), 25)
        self.assertEqual(int(low.sum()), 25)
        self.assertFalse(np.any(high & low))

    def test_shifted_controls_preserve_area_and_shape(self) -> None:
        mask = np.zeros((11, 13), dtype=bool)
        mask[2:6, 4:9] = True
        controls = shifted_controls(mask, 5, seed=3)
        self.assertEqual(len(controls), 5)
        self.assertTrue(all(int(row.sum()) == int(mask.sum()) for row in controls))


if __name__ == "__main__":
    unittest.main()
