#!/usr/bin/env python3
"""CUDA-free tests for bounded Stage3 recovery helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stage3_checkpoint_recovery import (
    latest_complete_checkpoint,
    settle_unresolved,
    validate_complete_checkpoint,
)
from stage3_openrouter_judge import BudgetLedger


def make_checkpoint(output: Path, step: int, *, world_size: int = 2) -> Path:
    checkpoint = output / f"checkpoint-{step}"
    state_dir = checkpoint / f"global_step{step}"
    state_dir.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "epoch": step / 500}) + "\n",
        encoding="utf-8",
    )
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model-00001.safetensors"}}) + "\n",
        encoding="utf-8",
    )
    for name in ("scheduler.pt", "training_args.bin", "model-00001.safetensors"):
        (checkpoint / name).write_bytes(b"x")
    (checkpoint / "latest").write_text(f"global_step{step}\n", encoding="utf-8")
    for rank in range(world_size):
        (state_dir / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt").write_bytes(b"x")
        (state_dir / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt").write_bytes(b"x")
        (checkpoint / f"rng_state_{rank}.pth").write_bytes(b"x")
    return checkpoint


class CheckpointTests(unittest.TestCase):
    def test_latest_skips_newer_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            complete = make_checkpoint(output, 100)
            incomplete = make_checkpoint(output, 200)
            (incomplete / "global_step200" / "bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt").unlink()
            result = latest_complete_checkpoint(output, world_size=2)
            self.assertIsNotNone(result)
            self.assertEqual(result["checkpoint"], str(complete.resolve()))
            self.assertEqual(len(result["rejected_newer_candidates"]), 1)

    def test_validator_rejects_step_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = make_checkpoint(Path(directory) / "output", 100)
            (checkpoint / "trainer_state.json").write_text(
                '{"global_step": 99}\n', encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                validate_complete_checkpoint(checkpoint, world_size=2)


class SettlementTests(unittest.TestCase):
    def test_all_unresolved_are_charged_at_full_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            ledger = BudgetLedger(
                path, limit_usd=1.0, reserve_usd=0.05, max_unique_requests=10
            )
            ledger.reserve("one", {})
            ledger.retain_failed_reservation("one", error="ambiguous", attempts=1)
            ledger.reserve("two", {})
            result = settle_unresolved(
                path,
                limit_usd=1.0,
                reserve_usd=0.05,
                max_unique_requests=10,
                reason="unit test",
            )
            self.assertEqual(result["settled_count"], 2)
            self.assertAlmostEqual(result["settled_usd"], 0.1)
            snapshot = ledger.snapshot()
            self.assertFalse(snapshot["reservations"])
            self.assertAlmostEqual(snapshot["committed_spend_usd"], 0.1)
            self.assertEqual(
                {row["status"] for row in snapshot["history"]},
                {"conservatively_settled_after_training_failure"},
            )

    def test_settlement_is_idempotent_after_first_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            ledger = BudgetLedger(
                path, limit_usd=1.0, reserve_usd=0.05, max_unique_requests=10
            )
            ledger.reserve("one", {})
            first = settle_unresolved(
                path, limit_usd=1.0, reserve_usd=0.05,
                max_unique_requests=10, reason="first",
            )
            second = settle_unresolved(
                path, limit_usd=1.0, reserve_usd=0.05,
                max_unique_requests=10, reason="second",
            )
            self.assertEqual(first["settled_count"], 1)
            self.assertEqual(second["settled_count"], 0)
            self.assertAlmostEqual(second["committed_spend_usd"], 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
