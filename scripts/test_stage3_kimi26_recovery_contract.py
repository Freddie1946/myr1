#!/usr/bin/env python3
"""Static regression tests for the approved Kimi Stage3 recovery contract."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/launch_stage3_kimi26_formal.sh"
SUPERVISOR = ROOT / "scripts/supervise_stage3_kimi26_formal.sh"


class KimiRecoveryContractTests(unittest.TestCase):
    def test_shell_syntax(self):
        subprocess.run(["bash", "-n", str(LAUNCHER), str(SUPERVISOR)], check=True)

    def test_recovery_request_cap_is_identical_and_amendable_upward(self):
        launcher = LAUNCHER.read_text(encoding="utf-8")
        supervisor = SUPERVISOR.read_text(encoding="utf-8")
        self.assertIn('PATHVLM_STAGE3_MAX_UNIQUE_REQUESTS:-12976', launcher)
        self.assertIn('PATHVLM_STAGE3_MAX_UNIQUE_REQUESTS:-12976', supervisor)
        self.assertIn("max_unique_requests < 12976", launcher)
        self.assertNotIn("MAX_UNIQUE_REQUESTS=12361", launcher)
        self.assertNotIn("MAX_UNIQUE_REQUESTS=12361", supervisor)

    def test_scientific_and_bounded_recovery_contract_is_unchanged(self):
        launcher = LAUNCHER.read_text(encoding="utf-8")
        supervisor = SUPERVISOR.read_text(encoding="utf-8")
        for fragment in (
            'PATHVLM_OPENROUTER_MODEL_ID="moonshotai/kimi-k2.6"',
            'PATHVLM_OPENROUTER_PROVIDER_ONLY="inceptron"',
            'PATHVLM_OPENROUTER_RETRY_DELAYS_SECONDS="15,45,90"',
            'PATHVLM_OPENROUTER_MIN_REQUEST_INTERVAL_SECONDS="$MIN_REQUEST_INTERVAL_SECONDS"',
            'PATHVLM_STAGE3_RULE_FALLBACK_TOTAL_LIMIT="$FALLBACK_TOTAL_LIMIT"',
            'PATHVLM_STAGE3_RULE_FALLBACK_CONSECUTIVE_LIMIT="$FALLBACK_CONSECUTIVE_LIMIT"',
            'PATHVLM_EPOCH_SNAPSHOT_STEPS="500,1000,1500"',
            'SAVE_STEPS="${PATHVLM_STAGE3_SAVE_STEPS:-100}"',
            "--max_steps 1500",
            '--save_steps "$SAVE_STEPS"',
            "--seed 42",
            "--data_seed 42",
        ):
            self.assertIn(fragment, launcher)
        self.assertIn('PATHVLM_STAGE3_MAX_RECOVERIES:-3', supervisor)
        self.assertIn('MAX_RECOVERIES > 6', supervisor)
        self.assertIn('--world-size 8', supervisor)
        self.assertIn('export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1"', launcher)
        self.assertIn('unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD || true', launcher)
        self.assertIn('PATHVLM_STAGE3_ALLOW_SHARED_GPUS:-false', launcher)
        self.assertIn('PATHVLM_STAGE3_SHARED_GPU_MIN_FREE_MIB:-30720', launcher)
        self.assertIn('PATHVLM_STAGE3_MIN_REQUEST_INTERVAL_SECONDS:-4', launcher)
        self.assertIn('Kimi rate-limit contract mismatch', launcher)


if __name__ == "__main__":
    unittest.main(verbosity=2)
