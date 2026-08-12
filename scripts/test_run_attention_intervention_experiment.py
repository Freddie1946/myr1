#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from run_attention_intervention_experiment import (
    attention_orders,
    deterministic_random_orders,
    intervene,
    parse_layer_spec,
    perturbation_base,
)
from run_visual_fidelity_experiment import grid_boxes


class AttentionInterventionTests(unittest.TestCase):
    def test_layer_specs_and_rank_orders(self) -> None:
        maps = []
        for layer in range(28):
            values = np.ones((2, 2), dtype=float)
            values.flat[layer % 4] = 9.0
            maps.append(values.tolist())
        orders, aggregated = attention_orders(maps, ["layer14", "layer16", "layers14_17"])
        self.assertEqual(parse_layer_spec("layers14_17", 28), [14, 15, 16, 17])
        self.assertEqual(orders["layer14"][0], 2)
        self.assertEqual(orders["layer16"][0], 0)
        self.assertEqual(orders["low_attention_layers14_17"], list(reversed(orders["layers14_17"])))
        self.assertAlmostEqual(sum(aggregated["layers14_17"]), 1.0)

    def test_mean_deletion_and_retention_are_complements(self) -> None:
        image = Image.new("RGB", (4, 2), (10, 20, 30))
        image.paste((210, 110, 60), (0, 0, 2, 2))
        boxes = grid_boxes(4, 2, 1, 2)
        base = perturbation_base(image, "mean")
        deletion = np.asarray(intervene(image, base, boxes, [0], "deletion"))
        retention = np.asarray(intervene(image, base, boxes, [0], "retention"))
        source = np.asarray(image)
        baseline = np.asarray(base)
        self.assertTrue(np.all(deletion[:, :2] == baseline[:, :2]))
        self.assertTrue(np.all(deletion[:, 2:] == source[:, 2:]))
        self.assertTrue(np.all(retention[:, :2] == source[:, :2]))
        self.assertTrue(np.all(retention[:, 2:] == baseline[:, 2:]))

    def test_random_orders_are_reproducible_and_complete(self) -> None:
        left = deterministic_random_orders("abc", 12, 3)
        right = deterministic_random_orders("abc", 12, 3)
        self.assertEqual(left, right)
        self.assertEqual(len({tuple(row) for row in left}), 3)
        self.assertTrue(all(sorted(row) == list(range(12)) for row in left))


if __name__ == "__main__":
    unittest.main()
