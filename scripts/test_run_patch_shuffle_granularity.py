from __future__ import annotations

import numpy as np
from PIL import Image

from run_patch_shuffle_granularity import aggregate, pixel_multiset_digest
from run_visual_understanding_counterfactual import patch_shuffle


def test_patch_shuffle_preserves_exact_pixel_multiset() -> None:
    values = np.arange(24 * 24 * 3, dtype=np.uint16).reshape(24, 24, 3)
    image = Image.fromarray((values % 251).astype(np.uint8), mode="RGB")
    shuffled = patch_shuffle(image, 8, 73)
    assert pixel_multiset_digest(image) == pixel_multiset_digest(shuffled)
    assert np.any(np.asarray(image) != np.asarray(shuffled))


def test_aggregate_uses_ground_truth_margin_and_paired_accuracy() -> None:
    def row(original_correct: bool, shuffled_correct: bool, original_margin: float, shuffled_margin: float):
        return {
            "conditions": {
                "original": {"correct": original_correct, "target_margin": original_margin, "target_probability": 0.7},
                "patch_shuffle_4x4": {"correct": shuffled_correct, "target_margin": shuffled_margin, "target_probability": 0.5},
            },
            "transform_audit": {"patch_shuffle_4x4": {"pixel_location_value_changed_rate": 0.9}},
        }
    result = aggregate([
        row(True, False, 2.0, 1.0),
        row(False, False, 0.0, -0.5),
    ], [4])
    metric = result["by_granularity"]["patch_shuffle_4x4"]
    assert result["original_accuracy"] == 0.5
    assert metric["counterfactual_accuracy"] == 0.0
    assert metric["paired_accuracy_gain"] == 0.5
    assert metric["mean_target_margin_gain"] == 0.75
    assert metric["mean_pixel_location_value_changed_rate"] == 0.9
