import numpy as np

from run_decision_score_agreement import parsed as parsed_agreement
from run_reference_evidence_behavior import controls, parsed as parsed_behavior


def test_answer_parsers_accept_reasoning_and_single_letter():
    for parser in (parsed_agreement, parsed_behavior):
        assert parser("<think>x</think><answer>C</answer>") == "C"
        assert parser("A") == "A"
        assert parser("no valid answer") is None


def test_matched_controls_preserve_area_and_avoid_reference():
    mask = np.zeros((100, 120), dtype=bool)
    mask[20:40, 30:50] = True
    neighbor, randoms = controls(mask, seed=42, count=3)
    assert int(neighbor.sum()) == int(mask.sum())
    assert len(randoms) == 3
    assert all(int(item.sum()) == int(mask.sum()) for item in randoms)
    assert not np.array_equal(neighbor, mask)
