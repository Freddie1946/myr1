import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("analyze_multijudge_reasoning_evaluation.py")
SPEC = importlib.util.spec_from_file_location("analyze_multijudge", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(judge, label, index, run, value):
    return {
        "judge": judge, "candidate_label": label, "index": index, "run": run,
        "scores": {metric: value for metric in MODULE.METRICS},
    }


def test_analysis_has_paired_delta_and_reliability():
    rows = []
    for judge in ("j1", "j2"):
        for index in (1, 2):
            for run in (0, 1, 2):
                rows.append(row(judge, "Stage2", index, run, 0.2 + 0.1 * index))
                rows.append(row(judge, "Stage3", index, run, 0.3 + 0.1 * index))
    result = MODULE.analyze(
        rows, stage2_label="Stage2", stage3_label="Stage3", expected_per_judge=12,
        bootstrap_replicates=100, seed=7,
    )
    assert result["complete"]
    assert abs(result["paired_stage3_vs_stage2"]["j1"]["stage3_minus_stage2"]["r_acc"]["mean_delta"] - 0.1) < 1e-9
    assert result["three_run_reliability"]["j1"]["three_run_units"] == 4
    assert result["three_run_reliability"]["j1"]["metrics"]["r_acc"]["mean_pairwise_absolute_difference"] == 0
    assert result["interjudge_agreement"]["j1 vs j2"]["n"] == 4


def test_percentile_interpolates():
    assert MODULE.percentile([0.0, 1.0], 0.5) == 0.5
