#!/usr/bin/env python3
"""CUDA-free tests for bounded Stage3 recovery helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stage3_checkpoint_recovery import (
    classify_training_failure,
    latest_complete_checkpoint,
    raise_request_cap,
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


class RequestCapAmendmentTests(unittest.TestCase):
    def test_settled_ledger_cap_can_only_be_raised_with_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            old = BudgetLedger(
                path, limit_usd=1.0, reserve_usd=0.05, max_unique_requests=10
            )
            old.reserve("one", {})
            old.commit("one", 0.01, {"status": "completed"})
            result = raise_request_cap(
                path,
                limit_usd=1.0,
                reserve_usd=0.05,
                old_max_unique_requests=10,
                new_max_unique_requests=15,
                reason="approved recovery allowance",
            )
            self.assertEqual(result["status"], "request_cap_raised")
            amended = BudgetLedger(
                path, limit_usd=1.0, reserve_usd=0.05, max_unique_requests=15
            ).snapshot()
            self.assertEqual(amended["completed_unique_requests"], 1)
            self.assertEqual(amended["contract_amendments"][-1]["old_value"], 10)
            self.assertEqual(amended["contract_amendments"][-1]["new_value"], 15)

    def test_cap_amendment_rejects_unresolved_reservations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.json"
            ledger = BudgetLedger(
                path, limit_usd=1.0, reserve_usd=0.05, max_unique_requests=10
            )
            ledger.reserve("one", {})
            with self.assertRaisesRegex(ValueError, "reservations are unresolved"):
                raise_request_cap(
                    path,
                    limit_usd=1.0,
                    reserve_usd=0.05,
                    old_max_unique_requests=10,
                    new_max_unique_requests=15,
                    reason="must fail",
                )


class FailureClassificationTests(unittest.TestCase):
    def classify(self, text: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text(text, encoding="utf-8")
            return classify_training_failure(path)

    def test_consecutive_judge_outage_is_recoverable(self):
        result = self.classify(
            "RuleFallbackLimitExceeded: rule fallback consecutive limit reached: 4"
        )
        self.assertEqual(result["action"], "recover")
        self.assertEqual(result["category"], "consecutive_judge_outage")

    def test_incomplete_read_is_recoverable(self):
        result = self.classify("http.client.IncompleteRead: IncompleteRead(0 bytes read)")
        self.assertEqual(result["action"], "recover")

    def test_openrouter_http_520_is_recoverable(self):
        result = self.classify(
            "stage3_openrouter_judge.TransportFailure: OpenRouter HTTP 520"
        )
        self.assertEqual(result["action"], "recover")
        self.assertEqual(result["category"], "transient_judge_transport")

    def test_interrupt_oom_budget_and_identity_stop(self):
        for message, category in (
            ("KeyboardInterrupt", "user_interrupt"),
            ("torch.OutOfMemoryError: CUDA out of memory", "resource_or_numeric_failure"),
            ("BudgetError: hard cap refuses next reservation", "budget_or_request_gate"),
            ("served model mismatch: expected x", "judge_identity_or_schema_mismatch"),
            (
                "_pickle.UnpicklingError: Weights only load failed.",
                "checkpoint_deserialization_compatibility",
            ),
        ):
            with self.subTest(message=message):
                result = self.classify(message)
                self.assertEqual(result["action"], "stop")
                self.assertEqual(result["category"], category)

    def test_total_fallback_limit_and_unknown_failure_stop(self):
        total = self.classify("rule fallback total limit reached: 24")
        self.assertEqual(total["action"], "stop")
        self.assertEqual(total["category"], "total_rule_fallback_limit")
        unknown = self.classify("some new distributed failure")
        self.assertEqual(unknown["action"], "stop")
        self.assertEqual(unknown["category"], "unknown_failure_fail_closed")

    def test_missing_log_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            result = classify_training_failure(Path(directory) / "missing.log")
        self.assertEqual(result["action"], "stop")
        self.assertEqual(result["category"], "missing_training_log")


if __name__ == "__main__":
    unittest.main(verbosity=2)
