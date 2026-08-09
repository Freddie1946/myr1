import importlib.util
from pathlib import Path


PATH = Path(__file__).with_name("analyze_paired_closed_benchmark.py")
SPEC = importlib.util.spec_from_file_location("paired_closed", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_paired_counts_and_mcnemar():
    left = {
        "a": {"image": "/x/1.png", "ok": True},
        "b": {"image": "/x/2.png", "ok": True},
        "c": {"image": "/x/3.png", "ok": False},
        "d": {"image": "/x/4.png", "ok": False},
    }
    right = {
        "a": {"image": "/x/1.png", "ok": True},
        "b": {"image": "/x/2.png", "ok": False},
        "c": {"image": "/x/3.png", "ok": True},
        "d": {"image": "/x/4.png", "ok": True},
    }
    result = MODULE.analyze(
        left, right, correctness_field="ok", expected_count=4, replicates=1000, seed=1
    )
    assert result["left_correct"] == 2
    assert result["right_correct"] == 3
    assert result["transition_counts"]["left_only"] == 1
    assert result["transition_counts"]["right_only"] == 2
    assert result["mcnemar_exact_two_sided_p"] == 1.0


def test_missing_field_without_rescorer_fails():
    try:
        MODULE.is_correct({}, "missing", None)
    except ValueError as error:
        assert "missing correctness field" in str(error)
    else:
        raise AssertionError("expected ValueError")
