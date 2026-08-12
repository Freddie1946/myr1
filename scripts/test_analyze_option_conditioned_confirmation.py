from __future__ import annotations

from analyze_option_conditioned_confirmation import holm_adjust, selectivity, sign_flip_p

import numpy as np


def test_selectivity_compares_target_with_other_options() -> None:
    assert selectivity(np.array([4.0, 1.0, 1.0, 1.0]), 0) == 3.0


def test_holm_adjust_is_monotone_and_preserves_missing() -> None:
    result = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2, "missing": None})
    assert result == {"a": 0.03, "b": 0.06, "c": 0.2, "missing": None}


def test_sign_flip_detects_consistently_positive_values() -> None:
    assert sign_flip_p([1.0] * 12, "positive", 10000) < 0.001
    assert sign_flip_p([], "empty", 100) is None
