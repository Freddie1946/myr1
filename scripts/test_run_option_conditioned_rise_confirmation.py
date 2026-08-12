from __future__ import annotations

import numpy as np

from run_option_conditioned_rise_confirmation import (
    analysis_size,
    normalized_saliency,
    option_saliency_maps,
)


def test_analysis_size_preserves_aspect_and_caps_long_edge() -> None:
    assert analysis_size(1600, 800, 256) == (256, 128)
    assert analysis_size(100, 50, 256) == (100, 50)


def test_normalized_saliency_tracks_mask_conditioned_score() -> None:
    masks = np.asarray([
        [[1, 0], [0, 0]],
        [[0, 1], [1, 1]],
    ], dtype=np.float32)
    result = normalized_saliency(np.asarray([2.0, -1.0]), masks)
    assert result[0, 0] == 1.0
    assert np.all(result[1:, :] == 0.0)


def test_option_maps_include_all_four_options() -> None:
    masks = np.asarray([
        [[1, 0], [0, 1]],
        [[0, 1], [1, 0]],
    ], dtype=np.float32)
    maps = option_saliency_maps([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]], masks)
    assert set(maps) == set("ABCD")
    assert all(value.shape == (2, 2) for value in maps.values())
