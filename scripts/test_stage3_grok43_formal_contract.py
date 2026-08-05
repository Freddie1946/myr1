#!/usr/bin/env python3
"""Static gates for the Grok 4.3 AIGCBest Stage3 arm."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/launch_stage3_grok43_formal.sh"
SUPERVISOR = ROOT / "scripts/supervise_stage3_grok43_formal.sh"


class GrokFormalContractTests(unittest.TestCase):
    def test_shell_syntax(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER), str(SUPERVISOR)], check=True)

    def test_identity_pricing_budget_and_stability_are_frozen(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        for fragment in (
            'PATHVLM_AIGCBEST_MODEL_ID="grok-4.3"',
            'PATHVLM_AIGCBEST_MODEL_RATIO="0.625"',
            'PATHVLM_AIGCBEST_COMPLETION_RATIO="2"',
            'PATHVLM_AIGCBEST_INPUT_USD_PER_MILLION="1.25"',
            'PATHVLM_AIGCBEST_OUTPUT_USD_PER_MILLION="2.5"',
            'PATHVLM_AIGCBEST_LIMIT_USD="50"',
            'grok-4.3-stability-gate.json',
            'formal_stage3_grok43_via_aigcbest_seed42',
        ):
            self.assertIn(fragment, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
