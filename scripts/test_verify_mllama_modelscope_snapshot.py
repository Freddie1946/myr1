#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from verify_mllama_modelscope_snapshot import verify


class VerifyMllamaModelScopeSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        for name in [
            "LICENSE.txt",
            "USE_POLICY.md",
            "chat_template.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]:
            (self.root / name).write_text("{}")
        (self.root / "config.json").write_text(
            json.dumps({"architectures": ["MllamaForConditionalGeneration"]})
        )
        shards = [f"model-{i:05d}-of-00005.safetensors" for i in range(1, 6)]
        for name in shards:
            (self.root / name).write_bytes(b"x")
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {f"w{i}": name for i, name in enumerate(shards)}})
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_valid_structure_and_frozen_size(self) -> None:
        with patch(
            "verify_mllama_modelscope_snapshot.FROZEN_MODELS",
            {"11b": {"shards": 5, "weight_bytes": 5}},
        ):
            result = verify(self.root, "11b")
        self.assertEqual(result["status"], "passed")

    def test_wrong_architecture_fails(self) -> None:
        (self.root / "config.json").write_text(json.dumps({"architectures": ["Other"]}))
        with self.assertRaisesRegex(ValueError, "unexpected architecture"):
            verify(self.root, "11b")

    def test_missing_shard_fails(self) -> None:
        (self.root / "model-00005-of-00005.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "weight shards are missing"):
            verify(self.root, "11b")


if __name__ == "__main__":
    unittest.main()

