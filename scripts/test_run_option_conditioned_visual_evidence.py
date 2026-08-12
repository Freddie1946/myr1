from __future__ import annotations

import numpy as np

from run_option_conditioned_visual_evidence import (
    aggregate,
    canonical_json,
    load_partial_records,
    margins,
    shifted_cell_masks,
    top_cell_mask,
)


def test_load_partial_records_validates_frozen_identity(tmp_path) -> None:
    cases = [
        {"panel_index": 0, "image_sha256": "image-0", "target_choice": "A"},
        {"panel_index": 1, "image_sha256": "image-1", "target_choice": "B"},
    ]
    path = tmp_path / "case_results.jsonl"
    path.write_text(canonical_json({
        "panel_index": 1, "image_sha256": "image-1", "target_choice": "B",
    }) + "\n", encoding="utf-8")
    assert [row["panel_index"] for row in load_partial_records(path, cases)] == [1]


def test_load_partial_records_rejects_duplicate_index(tmp_path) -> None:
    cases = [{"panel_index": 0, "image_sha256": "image-0", "target_choice": "A"}]
    row = canonical_json({"panel_index": 0, "image_sha256": "image-0", "target_choice": "A"})
    path = tmp_path / "case_results.jsonl"
    path.write_text(row + "\n" + row + "\n", encoding="utf-8")
    try:
        load_partial_records(path, cases)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate partial record should be rejected")


def test_top_and_low_masks_are_area_matched() -> None:
    values = np.arange(36, dtype=float).reshape(6, 6)
    high = top_cell_mask(values, 0.25, True)
    low = top_cell_mask(values, 0.25, False)
    assert high.sum() == low.sum() == 9
    assert np.array_equal(np.flatnonzero(high), np.arange(27, 36))
    assert np.array_equal(np.flatnonzero(low), np.arange(9))


def test_shifted_controls_preserve_shape_and_area() -> None:
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:3, 2:5] = True
    controls = shifted_cell_masks(mask, 2, 42)
    assert len(controls) == 2
    assert all(x.sum() == mask.sum() for x in controls)
    assert all(x.shape == mask.shape for x in controls)
    assert all(not np.array_equal(x, mask) for x in controls)


def test_option_margin_has_positive_target_when_probability_is_largest() -> None:
    values = margins([0.7, 0.1, 0.1, 0.1])
    assert values[0] > max(values[1:])


def test_aggregate_compares_random_controls_per_case() -> None:
    letters = "ABCD"
    methods = {}
    for method in ("positive", "low_positive", "negative", "random_0", "random_1"):
        methods[method] = {
            "deletion": {"option_margins": {x: (1.0 if x == "A" else 0.0) for x in letters}},
            "retention": {"option_margins": {x: (0.5 if x == "A" else 0.0) for x in letters}},
        }
    row = {
        "baseline_correct": True,
        "target_choice": "A",
        "baseline_predicted_choice": "A",
        "baseline_option_margins": {x: (2.0 if x == "A" else 0.0) for x in letters},
        "attribution": {"baseline_option_margins": {x: (2.0 if x == "A" else 0.0) for x in letters}},
        "interventions": {"0": {x: {"methods": methods} for x in letters}},
    }
    summary = aggregate([row], [0])["by_layer_option"]["0"]["A"]
    assert summary["positive_beats_random_rate"] == 0.0
    assert summary["positive_mean_margin_drop"] == summary["random_mean_margin_drop"]


def test_aggregate_excludes_unavailable_positive_map() -> None:
    letters = "ABCD"
    only_raw = {
        "raw_attention": {
            "deletion": {"option_margins": {x: 0.0 for x in letters}},
            "retention": {"option_margins": {x: 0.0 for x in letters}},
        }
    }
    row = {
        "baseline_correct": True,
        "target_choice": "A",
        "baseline_predicted_choice": "A",
        "baseline_option_margins": {x: 0.0 for x in letters},
        "attribution": {"baseline_option_margins": {x: 0.0 for x in letters}},
        "interventions": {"0": {x: {"positive_available": False, "methods": only_raw} for x in letters}},
    }
    result = aggregate([row], [0])
    option = result["by_layer_option"]["0"]["A"]
    assert option["positive_available_case_count"] == 0
    assert option["positive_mean_margin_drop"] is None
    assert result["by_layer_role"]["0"]["ground_truth"]["available_case_count"] == 0
