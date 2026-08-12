from __future__ import annotations

import numpy as np

from PIL import Image

from render_option_conditioned_layer_profiles import display_map, overlay


def test_display_map_rejects_zero_and_invalid_maps() -> None:
    assert display_map(np.zeros((3, 4))) is None
    assert display_map(np.array([[1.0, np.nan]])) is None


def test_display_map_normalizes_positive_map() -> None:
    result = display_map(np.arange(1, 13, dtype=float).reshape(3, 4))
    assert result is not None
    assert result.shape == (3, 4)
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_overlay_preserves_image_geometry() -> None:
    image = Image.fromarray(np.full((31, 47, 3), 128, dtype=np.uint8), mode="RGB")
    result = overlay(image, np.arange(12, dtype=float).reshape(3, 4), 0.25)
    assert result.size == image.size
