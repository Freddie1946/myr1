from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from run_layerwise_visual_activation_patching import (
    aggregate,
    decoder_hidden,
    geometry_match,
    replace_decoder_hidden,
)


def test_geometry_match_uses_exact_reference_size() -> None:
    image = Image.fromarray(np.zeros((20, 40, 3), dtype=np.uint8), mode="RGB")
    reference = Image.fromarray(np.zeros((31, 17, 3), dtype=np.uint8), mode="RGB")
    assert geometry_match(image, reference).size == reference.size


def test_decoder_output_compatibility_preserves_container() -> None:
    hidden = torch.zeros(1, 3, 4)
    replacement = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    tensor_output = replace_decoder_hidden(hidden, replacement, 2)
    assert isinstance(tensor_output, torch.Tensor)
    assert torch.equal(tensor_output[:, -1, :], replacement)
    tuple_output = replace_decoder_hidden((hidden, "attention", "cache"), replacement, 2)
    assert isinstance(tuple_output, tuple)
    assert tuple_output[1:] == ("attention", "cache")
    assert torch.equal(decoder_hidden(tuple_output, 2)[:, -1, :], replacement)


def test_aggregate_measures_directional_recovery() -> None:
    letters = "ABCD"
    def state(target_margin: float, probabilities: list[float], correct: bool):
        return {
            "option_margins": {letter: (target_margin if letter == "A" else 0.0) for letter in letters},
            "option_probabilities": dict(zip(letters, probabilities)),
            "correct": correct,
        }
    row = {
        "target_choice": "A",
        "original": state(2.0, [0.7, 0.1, 0.1, 0.1], True),
        "mismatch_baseline": state(0.0, [0.25, 0.25, 0.25, 0.25], False),
        "patched_by_layer": {
            "27": {
                "correct_direction": state(2.0, [0.7, 0.1, 0.1, 0.1], True),
                "opposite_direction": state(-1.0, [0.1, 0.3, 0.3, 0.3], False),
                "permuted_direction_controls": [
                    {"condition": state(0.5, [0.4, 0.2, 0.2, 0.2], True)},
                    {"condition": state(-0.5, [0.2, 0.3, 0.3, 0.2], False)},
                ],
            }
        },
    }
    result = aggregate([row], [27])
    layer = result["by_layer"]["27"]
    assert layer["mean_correct_patch_target_margin_recovery"] == 2.0
    assert layer["mean_opposite_patch_target_margin_effect"] == -1.0
    assert layer["mean_correct_minus_opposite_directional_contrast"] == 3.0
    assert layer["aggregate_recovery_fraction"] == 1.0
    assert layer["max_correct_patch_vs_original_probability_error"] == 0.0
    assert layer["mean_permuted_direction_target_margin_effect"] == 0.0
    assert layer["mean_correct_minus_permuted_direction"] == 2.0
    assert layer["correct_patch_beats_permuted_rate"] == 1.0
