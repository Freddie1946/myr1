#!/usr/bin/env python3
"""Regression tests for audited two-tier checkpoint retention."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from safetensors.torch import save_file
import torch

from formal_machine.checkpoint_retention import (
    TwoTierCheckpointArchiver,
    archive_checkpoint,
    validate_snapshot,
)


def make_checkpoint(root: Path, step: int, epoch: float) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    save_file({"weight": torch.arange(8, dtype=torch.bfloat16)}, checkpoint / "model.safetensors")
    index = {
        "metadata": {"total_size": 16},
        "weight_map": {"weight": "model.safetensors"},
    }
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(index) + "\n", encoding="utf-8"
    )
    (checkpoint / "config.json").write_text('{"model_type":"unit_test"}\n', encoding="utf-8")
    for name in (
        "merges.txt", "preprocessor_config.json", "special_tokens_map.json",
        "tokenizer.json", "tokenizer_config.json", "vocab.json",
    ):
        (checkpoint / name).write_text("{}\n", encoding="utf-8")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "epoch": epoch}) + "\n", encoding="utf-8"
    )
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer-state-must-not-be-archived")
    return checkpoint


class RetentionTests(unittest.TestCase):
    def test_model_snapshot_survives_source_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = make_checkpoint(root / "output", 10, 1.0)
            snapshot_root = root / "epoch_snapshots"
            payload = archive_checkpoint(checkpoint, snapshot_root)
            snapshot = snapshot_root / "checkpoint-10"
            self.assertEqual(payload["global_step"], 10)
            self.assertFalse((snapshot / "optimizer.pt").exists())
            self.assertEqual(
                (checkpoint / "model.safetensors").stat().st_ino,
                (snapshot / "model.safetensors").stat().st_ino,
            )
            for path in checkpoint.iterdir():
                path.unlink()
            checkpoint.rmdir()
            self.assertEqual(validate_snapshot(snapshot)["epoch"], 1.0)

    def test_archiver_preserves_all_snapshots_while_full_checkpoints_rotate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            snapshots = root / "epoch_snapshots"
            archiver = TwoTierCheckpointArchiver(
                output, snapshots, root / "events.jsonl", minimum_free_bytes=0,
                poll_seconds=0.01,
            )
            for step in (10, 20, 30):
                make_checkpoint(output, step, step / 10)
                archiver.scan_once()
                checkpoints = sorted(output.glob("checkpoint-*"))
                if len(checkpoints) > 2:
                    oldest = checkpoints[0]
                    for path in oldest.iterdir():
                        path.unlink()
                    oldest.rmdir()
            summary = archiver.stop_and_validate()
            self.assertEqual(summary["count"], 3)
            self.assertEqual(len(list(output.glob("checkpoint-*"))), 2)
            self.assertEqual(len(list(snapshots.glob("checkpoint-*"))), 3)

    def test_incomplete_checkpoint_is_not_archived(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "output/checkpoint-1"
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
            archiver = TwoTierCheckpointArchiver(
                root / "output", root / "snapshots", root / "events.jsonl",
                minimum_free_bytes=0,
            )
            archiver.scan_once()
            self.assertEqual(archiver.stop_and_validate()["count"], 0)

    def test_transient_empty_tokenizer_file_is_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = make_checkpoint(root / "output", 1, 1.0)
            (checkpoint / "vocab.json").write_bytes(b"")
            archiver = TwoTierCheckpointArchiver(
                root / "output", root / "snapshots", root / "events.jsonl",
                minimum_free_bytes=0,
            )
            archiver.scan_once()
            self.assertFalse((root / "snapshots/checkpoint-1").exists())
            (checkpoint / "vocab.json").write_text("{}\n", encoding="utf-8")
            archiver.scan_once()
            summary = archiver.stop_and_validate()
            self.assertEqual(summary["count"], 1)
            self.assertEqual(summary["missing_observed_snapshot_steps"], [])


if __name__ == "__main__":
    unittest.main()
