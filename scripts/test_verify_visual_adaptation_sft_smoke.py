#!/usr/bin/env python3

import unittest

from verify_visual_adaptation_sft_smoke import canonical_lora_module


class VisualAdaptationSmokeVerifierTests(unittest.TestCase):
    def test_canonical_language_module(self):
        key = "base_model.model.model.layers.3.self_attn.q_proj.lora_B.weight"
        self.assertEqual(
            canonical_lora_module(key),
            ("model.layers.3.self_attn.q_proj", "lora_B"),
        )

    def test_canonical_vision_module(self):
        key = "base_model.model.visual.blocks.31.attn.qkv.lora_A.weight"
        self.assertEqual(
            canonical_lora_module(key),
            ("visual.blocks.31.attn.qkv", "lora_A"),
        )

    def test_non_lora_key_is_ignored(self):
        self.assertIsNone(
            canonical_lora_module("base_model.model.visual.merger.mlp.0.weight")
        )


if __name__ == "__main__":
    unittest.main()
