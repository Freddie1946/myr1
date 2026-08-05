#!/usr/bin/env python3
"""Focused tests for attention query selection and layer-band summaries."""

import unittest

import torch

from analyze_attention_layers import summarize_layer_bands
from run_attention_visualization_pilot import (
    PROMPT_SUFFIX,
    find_unique_subsequence,
    question_option_query_positions,
)


class FakeTokenizer:
    all_special_ids = [90, 91]

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == PROMPT_SUFFIX:
            return [70, 71]
        raise AssertionError(text)

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return {11: "Question", 12: " ", 13: "A", 90: "<special>"}.get(ids[0], "x")


class AttentionQueryTests(unittest.TestCase):
    def test_unique_subsequence(self):
        self.assertEqual(find_unique_subsequence([1, 2, 3, 4], [2, 3]), 1)
        with self.assertRaises(ValueError):
            find_unique_subsequence([1, 2, 1, 2], [1, 2])

    def test_question_positions_exclude_special_blank_and_suffix(self):
        positions, tokens = question_option_query_positions(
            FakeTokenizer(),
            torch.tensor([90, 99, 11, 12, 90, 13, 70, 71, 91]),
            last_image_position=1,
        )
        self.assertEqual(positions.tolist(), [2, 5])
        self.assertEqual(tokens, ["Question", "A"])

    def test_layer_band_summary(self):
        layers = [
            {"iou": index / 100, "pointing_game_hit": index % 2 == 0,
             "roi_attention_enrichment": 1 + index / 100}
            for index in range(28)
        ]
        records = [
            {"method": method, "layers": layers}
            for method in ("last_query", "blog_text_query", "question_options_query")
        ]
        summaries = summarize_layer_bands(records)
        self.assertEqual(len(summaries), 9)
        self.assertEqual(summaries[0]["layers"], list(range(8)))
        self.assertAlmostEqual(summaries[0]["mean"]["iou"], 0.035)


if __name__ == "__main__":
    unittest.main()
