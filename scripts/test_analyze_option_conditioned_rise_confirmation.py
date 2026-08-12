from __future__ import annotations

from analyze_option_conditioned_rise_confirmation import case_advantages


def test_case_advantages_use_correct_auc_direction() -> None:
    row = {
        "curves": {
            "deletion": {
                "rise_high": {"target_margin_auc": 1.0},
                "rise_low": {"target_margin_auc": 2.0},
                "random_shifted_shape_matched": [
                    {"target_margin_auc": 1.5}, {"target_margin_auc": 2.5},
                ],
            },
            "retention": {
                "rise_high": {"target_margin_auc": 3.0},
                "rise_low": {"target_margin_auc": 2.0},
                "random_shifted_shape_matched": [
                    {"target_margin_auc": 1.5}, {"target_margin_auc": 2.5},
                ],
            },
        }
    }
    assert case_advantages(row, "deletion") == {
        "high_vs_random": 1.0, "high_vs_low": 1.0, "low_vs_random": 0.0,
    }
    assert case_advantages(row, "retention") == {
        "high_vs_random": 1.0, "high_vs_low": 1.0, "low_vs_random": 0.0,
    }
