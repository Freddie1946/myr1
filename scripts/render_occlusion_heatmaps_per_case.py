#!/usr/bin/env python3
"""Render stored 6x6 occlusion values with an explicit per-case color scale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-index", type=int, action="append", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = {
        row["panel_index"]: row
        for row in (json.loads(line) for line in args.case_results.open())
    }
    for panel_index in args.panel_index:
        row = rows[panel_index]
        image = Image.open(row["image"]).convert("RGB")
        values = np.asarray(row["patch_importance_log_target_probability_drop"], dtype=np.float32).reshape(6, 6)
        heat = np.asarray(
            Image.fromarray(values).resize(image.size, resample=Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        vmax = float(np.max(np.abs(values))) or 1.0
        figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[1].imshow(heat, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axes[1].set_title(f"6x6 occlusion, per-case scale (+/-{vmax:.4f})")
        axes[2].imshow(image)
        overlay = axes[2].imshow(
            heat, cmap="coolwarm", vmin=-vmax, vmax=vmax, alpha=0.5
        )
        axes[2].set_title("Image + occlusion importance")
        for axis in axes:
            axis.axis("off")
        figure.colorbar(
            overlay,
            ax=axes,
            shrink=0.75,
            label="log normalized target-option probability drop",
        )
        figure.savefig(args.output_dir / f"case_{panel_index:02d}_heatmap.png", dpi=180)
        plt.close(figure)


if __name__ == "__main__":
    main()
