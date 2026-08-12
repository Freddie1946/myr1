#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np
import torch

from run_integrated_gradients_visual_evidence import reconstruct_patch_attribution


class IntegratedGradientsVisualEvidenceTests(unittest.TestCase):
    def test_reconstruct_preserves_values_and_spatial_quadrants(self) -> None:
        # grid 4x4 raw patches, merge 2, patch 1, one channel/time.
        # Processor order groups the four micro-patches of each merged token.
        values = torch.arange(16, dtype=torch.float32).reshape(16, 1)
        image = reconstruct_patch_attribution(
            values, grid_h=4, grid_w=4, merge=2, patch=1, temporal=1
        )
        expected = np.array(
            [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]],
            dtype=np.float32,
        )
        self.assertTrue(np.array_equal(image, expected))
        self.assertAlmostEqual(float(image.sum()), float(values.sum()))


if __name__ == "__main__":
    unittest.main()
