#!/usr/bin/env python3
from __future__ import annotations

import ast
import unittest
from pathlib import Path


class DeepSeekVL2ExternalRunnerStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path = Path(__file__).with_name("run_external_vqa_deepseek_vl2.py")
        cls.source = cls.path.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_official_native_interface_is_used(self) -> None:
        self.assertIn("DeepseekVLV2Processor", self.source)
        self.assertIn("prepare_inputs_embeds", self.source)
        self.assertIn("model.language.generate", self.source)

    def test_frozen_external_contract_is_used(self) -> None:
        self.assertIn("prompt_for_record", self.source)
        self.assertIn("score_record", self.source)
        self.assertIn("max_new_tokens != 64", self.source)
        self.assertIn('choices=("adapter_smoke", "external_test")', self.source)

    def test_resume_and_source_identity_are_delegated_to_shared_utils(self) -> None:
        self.assertIn("load_existing(predictions_path, records, args.task)", self.source)
        self.assertIn("append_row(predictions_path, row)", self.source)
        self.assertIn("result_row(", self.source)

    def test_no_sampling_or_quantization(self) -> None:
        self.assertIn('"quantization": "none"', self.source)
        self.assertIn('"do_sample": False', self.source)
        self.assertIn("do_sample=False", self.source)


if __name__ == "__main__":
    unittest.main()

