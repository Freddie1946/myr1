#!/usr/bin/env python3

import argparse
import unittest
from pathlib import Path

from prepare_visual_adaptation_sft_pilot import build_config, target_modules


class VisualAdaptationPilotConfigTests(unittest.TestCase):
    def args(self, arm: str):
        return argparse.Namespace(
            arm=arm, mode="formal", model=Path("/model"), dataset_dir=Path("/data"),
            deepspeed=Path("/ds.json"), output_dir=Path("/out"), config=Path("/c"),
            manifest=Path("/m"), lora_micro_batch=4, full_micro_batch=1,
            world_size=8 if arm == "c0" else 2, smoke_steps=12,
            num_train_epochs=2.0, save_strategy="steps", save_steps=63,
            full_learning_rate=2.0e-5, lora_learning_rate=5.0e-5,
        )

    def test_a_freezes_vision_and_trains_merger(self):
        value = build_config(self.args("a"))
        self.assertTrue(value["freeze_vision_tower"])
        self.assertFalse(value["freeze_multi_modal_projector"])
        self.assertEqual(value["additional_target"], ["visual.merger"])
        self.assertFalse(any(name.startswith("visual.blocks") for name in value["lora_target"]))

    def test_l_freezes_vision_and_merger_and_trains_only_language_lora(self):
        value = build_config(self.args("l"))
        self.assertTrue(value["freeze_vision_tower"])
        self.assertTrue(value["freeze_multi_modal_projector"])
        self.assertNotIn("additional_target", value)
        self.assertFalse(any(name.startswith("visual.blocks") for name in value["lora_target"]))

    def test_b2_and_b4_have_only_requested_visual_blocks(self):
        b2 = target_modules("b2")
        b4 = target_modules("b4")
        self.assertTrue(any(name.startswith("visual.blocks.30") for name in b2))
        self.assertFalse(any(name.startswith("visual.blocks.29") for name in b2))
        self.assertTrue(any(name.startswith("visual.blocks.28") for name in b4))
        self.assertFalse(any(name.startswith("visual.blocks.27") for name in b4))
        self.assertEqual(len(b4) - len(b2), 10)

    def test_matched_global_batch(self):
        c0 = build_config(self.args("c0"))
        a = build_config(self.args("a"))
        self.assertEqual(c0["per_device_train_batch_size"] * 8, 8)
        self.assertEqual(a["per_device_train_batch_size"] * 2, 8)

    def test_throughput_smoke_uses_real_data_and_bounded_steps(self):
        args = self.args("c0")
        args.mode = "throughput_smoke"
        value = build_config(args)
        self.assertEqual(value["dataset"], "pathvlm_sft_n0500")
        self.assertEqual(value["max_samples"], 500)
        self.assertEqual(value["max_steps"], 12)
        self.assertEqual(value["save_strategy"], "no")
        self.assertNotIn("save_steps", value)
        self.assertNotIn("num_train_epochs", value)


if __name__ == "__main__":
    unittest.main()
