#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("prepare_lora_sft4000_control.py")
SPEC = importlib.util.spec_from_file_location("prepare_lora_sft4000_control", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class LoraSft4000ControlTests(unittest.TestCase):
    def test_config_continues_same_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = MODULE.build_config(
                root / "model",
                root / "parent",
                root / "data",
                root / "output",
                world_size=8,
                micro_batch=12,
                gradient_accumulation=1,
                epochs=2.0,
                max_steps=None,
            )
        self.assertEqual(config["finetuning_type"], "lora")
        self.assertFalse(config["create_new_adapter"])
        self.assertTrue(config["freeze_vision_tower"])
        self.assertTrue(config["freeze_multi_modal_projector"])
        self.assertEqual(config["per_device_train_batch_size"], 12)
        self.assertEqual(config["num_train_epochs"], 2.0)
        self.assertEqual(config["learning_rate"], 5e-5)
        self.assertEqual(config["cutoff_len"], 1024)
        self.assertEqual(config["save_strategy"], "epoch")

    def test_smoke_disables_checkpoint_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = MODULE.build_config(
                root / "model",
                root / "parent",
                root / "data",
                root / "output",
                world_size=1,
                micro_batch=1,
                gradient_accumulation=1,
                epochs=2.0,
                max_steps=2,
            )
        self.assertEqual(config["max_steps"], 2)
        self.assertEqual(config["save_strategy"], "no")


if __name__ == "__main__":
    unittest.main()
