#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from verify_checkpoint_snapshot import verify_snapshot


class VerifyCheckpointSnapshotTests(unittest.TestCase):
    def make_snapshot(self) -> Path:
        root = Path(self.tempdir.name) / "checkpoint-500"
        root.mkdir()
        files = {
            "chat_template.json": b"{}\n",
            "config.json": b"{}\n",
            "model-00001-of-00001.safetensors": b"weights",
            "model.safetensors.index.json": json.dumps(
                {"weight_map": {"x": "model-00001-of-00001.safetensors"}}
            ).encode(),
            "trainer_state.json": json.dumps({"global_step": 500, "epoch": 1.0}).encode(),
        }
        for name, data in files.items():
            (root / name).write_bytes(data)
        manifest = {
            "schema_version": 1,
            "global_step": 500,
            "epoch": 1.0,
            "model_only": True,
            "resumable": False,
            "files": [
                {
                    "name": name,
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
                for name, data in sorted(files.items())
            ],
        }
        (root / "snapshot_manifest.json").write_text(json.dumps(manifest))
        return root

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_valid_snapshot(self) -> None:
        root = self.make_snapshot()
        result = verify_snapshot(root, expected_step=500, expected_epoch=1.0)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["model_shard_count"], 1)

    def test_tampered_file_fails(self) -> None:
        root = self.make_snapshot()
        (root / "config.json").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "size mismatch|SHA-256 mismatch"):
            verify_snapshot(root)

    def test_unexpected_file_fails(self) -> None:
        root = self.make_snapshot()
        (root / "extra.bin").write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "file set mismatch"):
            verify_snapshot(root)

    def test_trainer_state_identity_fails(self) -> None:
        root = self.make_snapshot()
        state = root / "trainer_state.json"
        state.write_text(json.dumps({"global_step": 499, "epoch": 1.0}))
        manifest = json.loads((root / "snapshot_manifest.json").read_text())
        row = next(x for x in manifest["files"] if x["name"] == state.name)
        data = state.read_bytes()
        row["size_bytes"] = len(data)
        row["sha256"] = hashlib.sha256(data).hexdigest()
        (root / "snapshot_manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "trainer_state global_step"):
            verify_snapshot(root)


if __name__ == "__main__":
    unittest.main()

