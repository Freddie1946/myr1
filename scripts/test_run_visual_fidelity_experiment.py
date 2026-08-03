#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from run_visual_fidelity_experiment import (
    grid_boxes,
    perturb_image,
    selected_prefix,
    trapezoid_auc,
)


class VisualFidelityExperimentTests(unittest.TestCase):
    def test_grid_covers_nondivisible_image_without_overlap_gaps(self) -> None:
        boxes = grid_boxes(7, 5, 2, 3)
        coverage = np.zeros((5, 7), dtype=np.int32)
        for left, top, right, bottom in boxes:
            coverage[top:bottom, left:right] += 1
        self.assertTrue(np.all(coverage == 1))

    def test_deletion_and_insertion_are_complements(self) -> None:
        image = Image.new("RGB", (4, 2), (10, 20, 30))
        image.paste((200, 100, 50), (0, 0, 2, 2))
        boxes = grid_boxes(4, 2, 1, 2)
        deletion = np.asarray(
            perturb_image(image, boxes, [0], mode="deletion", fill=(1, 2, 3))
        )
        insertion = np.asarray(
            perturb_image(image, boxes, [0], mode="insertion", fill=(1, 2, 3))
        )
        self.assertTrue(np.all(deletion[:, :2] == (1, 2, 3)))
        self.assertTrue(np.all(deletion[:, 2:] == (10, 20, 30)))
        self.assertTrue(np.all(insertion[:, :2] == (200, 100, 50)))
        self.assertTrue(np.all(insertion[:, 2:] == (1, 2, 3)))

    def test_fraction_rounding_and_auc(self) -> None:
        order = list(range(36))
        self.assertEqual(len(selected_prefix(order, 0.25)), 9)
        self.assertEqual(selected_prefix(order, 0.0), [])
        self.assertEqual(selected_prefix(order, 1.0), order)
        self.assertAlmostEqual(trapezoid_auc([0.0, 0.5, 1.0], [1.0, 0.5, 0.0]), 0.5)
        with self.assertRaises(ValueError):
            trapezoid_auc([0.0, 0.5], [1.0, 0.5])


if __name__ == "__main__":
    unittest.main()

