#!/usr/bin/env python3
"""Static gates for optional Stage3/shared-GPU evaluation admission."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "scripts/run_stage3_selected_full_evaluations.sh"
VISUAL = ROOT / "scripts/run_stage3_gpt4o_visual_fidelity.sh"


class ParallelGpuAdmissionContractTests(unittest.TestCase):
    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(EVALUATION), str(VISUAL)], check=True)

    def test_shared_mode_is_explicit_and_bounded(self) -> None:
        for path in (EVALUATION, VISUAL):
            text = path.read_text(encoding="utf-8")
            self.assertIn("PATHVLM_ALLOW_SHARED_GPU_WITH_STAGE3", text)
            self.assertIn("PATHVLM_SHARED_GPU_OWNER_RUN_DIR", text)
            self.assertIn("PATHVLM_SHARED_GPU_MAX_INITIAL_USED_MIB:-30000", text)
            self.assertIn("PATHVLM_SHARED_GPU_MIN_FREE_MIB:-45000", text)
            self.assertIn("rewards/audited_process_reward", text)
            self.assertIn('"shared_with_stage3"', text)

    def test_default_idle_gate_is_retained(self) -> None:
        self.assertIn("used <= 10", EVALUATION.read_text(encoding="utf-8"))
        self.assertIn("used <= 10", VISUAL.read_text(encoding="utf-8"))

    def test_full_evaluation_rechecks_owner_after_smoke(self) -> None:
        text = EVALUATION.read_text(encoding="utf-8")
        smoke_passed = text.index('record_event "smoke_phase_passed"')
        owner_check = text.index("verify_shared_owner ||", smoke_passed)
        full_started = text.index('record_event "full_phase_started"')
        self.assertLess(smoke_passed, owner_check)
        self.assertLess(owner_check, full_started)


if __name__ == "__main__":
    unittest.main(verbosity=2)
