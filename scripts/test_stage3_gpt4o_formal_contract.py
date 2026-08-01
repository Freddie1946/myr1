#!/usr/bin/env python3
"""Static regression tests for the formal GPT-4o Stage3 shell contract."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/launch_stage3_gpt4o_formal.sh"
SUPERVISOR = ROOT / "scripts/supervise_stage3_gpt4o_formal.sh"


class FormalContractTests(unittest.TestCase):
    def test_shell_syntax(self):
        subprocess.run(
            ["bash", "-n", str(LAUNCHER), str(SUPERVISOR)],
            check=True,
        )

    def test_frozen_scientific_contract(self):
        text = LAUNCHER.read_text(encoding="utf-8")
        for fragment in (
            'PENALTY="0.4"',
            "MAX_HTTP_ATTEMPTS=12360",
            'ACCOUNTING_CAPACITY_USD="247.20"',
            "--num_generations 4",
            "--per_device_train_batch_size 1",
            "--gradient_accumulation_steps 1",
            "--learning_rate 1.0e-6",
            "--beta 0.04",
            "--max_steps 1500",
            "--save_steps 100",
            "--save_total_limit 2",
            "--seed 42",
            "--data_seed 42",
            'PATHVLM_EPOCH_SNAPSHOT_STEPS="500,1000,1500"',
            'PATHVLM_STAGE3_RULE_FALLBACK_TOTAL_LIMIT="24"',
            'PATHVLM_STAGE3_RULE_FALLBACK_CONSECUTIVE_LIMIT="4"',
        ):
            self.assertIn(fragment, text)

    def test_no_paid_launch_in_preflight_only_mode(self):
        text = LAUNCHER.read_text(encoding="utf-8")
        gate = text.index('PATHVLM_STAGE3_PREFLIGHT_ONLY:-false')
        training = text.index('Starting formal GPT-4o Stage3')
        self.assertLess(gate, training)

    def test_supervisor_is_classification_gated_and_bounded(self):
        text = SUPERVISOR.read_text(encoding="utf-8")
        self.assertIn('PATHVLM_STAGE3_MAX_RECOVERIES:-3', text)
        self.assertIn('classify --log "$train_log"', text)
        self.assertIn('"event":"failure_classifier_error"', text)
        self.assertIn('if [[ "$recovery_action" != "recover" ]]', text)
        self.assertIn('--world-size 8', text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
