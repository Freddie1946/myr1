#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "formal_machine/run_sft4000_control.py"
sys.path.insert(0, str(RUNNER.parent))
SPEC = importlib.util.spec_from_file_location("run_sft4000_control", RUNNER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SFT4000ControlTests(unittest.TestCase):
    def test_resolved_config_is_two_epoch_full_sft(self):
        config = MODULE.resolved_config(
            Path("/parent"), Path("/data"), Path("/output"), Path("/ds.json")
        )
        self.assertEqual(config["num_train_epochs"], 2)
        self.assertEqual(config["max_samples"], 1000)
        self.assertEqual(config["learning_rate"], 2e-5)
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["stage"], "sft")
        self.assertEqual(config["finetuning_type"], "full")
        self.assertTrue(config["freeze_vision_tower"])
        self.assertTrue(config["freeze_multi_modal_projector"])
        self.assertFalse(config["freeze_language_model"])
        self.assertNotIn("test", json.dumps(config).lower())

    def test_checkpoint_identity_rejects_wrong_manifest_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "config.json",
                "model.safetensors.index.json",
                "snapshot_manifest.json",
                "trainer_state.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "preprocessor_config.json",
            ):
                (root / name).write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest SHA-256"):
                MODULE.verify_checkpoint_identity(root)


if __name__ == "__main__":
    unittest.main()
