from __future__ import annotations

import numpy as np
from PIL import Image

from run_visual_understanding_counterfactual import exact_mcnemar_p, patch_shuffle


def test_patch_shuffle_preserves_pixels_and_changes_layout() -> None:
    values = np.arange(24 * 32 * 3, dtype=np.uint16).reshape(24, 32, 3) % 256
    image = Image.fromarray(values.astype(np.uint8), mode="RGB")
    shuffled = np.asarray(patch_shuffle(image, 8, 42))
    assert shuffled.shape == values.shape
    assert not np.array_equal(shuffled, values.astype(np.uint8))
    assert np.array_equal(
        np.sort(shuffled.reshape(-1, 3), axis=0),
        np.sort(values.astype(np.uint8).reshape(-1, 3), axis=0),
    )


def test_exact_mcnemar_is_symmetric_and_bounded() -> None:
    assert exact_mcnemar_p(5, 1) == exact_mcnemar_p(1, 5)
    assert 0.0 <= exact_mcnemar_p(12, 1) <= 1.0
    assert exact_mcnemar_p(0, 0) == 1.0
