#!/usr/bin/env python3
"""Render fixed-case raw and option-conditioned attribution depth profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from run_option_conditioned_visual_evidence import sha256_file, top_cell_mask


def display_map(values: Any) -> np.ndarray | None:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)) or float(array.sum()) <= 0:
        return None
    maximum = float(np.quantile(array[array > 0], 0.99))
    return np.clip(array / max(maximum, 1e-12), 0.0, 1.0)


def overlay(image: Image.Image, values: Any, fraction: float) -> Image.Image:
    shown = display_map(values)
    if shown is None:
        result = image.copy()
        draw = ImageDraw.Draw(result)
        draw.rectangle((4, 4, 94, 24), fill=(0, 0, 0))
        draw.text((8, 8), "unavailable", fill=(255, 255, 255))
        return result
    heat = np.stack(
        [shown, np.square(shown), 0.15 * (1.0 - shown)], axis=-1
    )
    heat_image = Image.fromarray(np.uint8(np.clip(heat, 0, 1) * 255), mode="RGB")
    heat_image = heat_image.resize(image.size, Image.Resampling.BILINEAR)
    result = Image.blend(image.convert("RGB"), heat_image, 0.52)
    selected = top_cell_mask(np.asarray(values, dtype=np.float64), fraction, True)
    binary = Image.fromarray(np.uint8(selected) * 255, mode="L").resize(
        image.size, Image.Resampling.NEAREST
    )
    mask = np.asarray(binary) > 0
    boundary = np.zeros_like(mask)
    boundary[1:, :] |= mask[1:, :] != mask[:-1, :]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    output = np.asarray(result).copy()
    output[boundary] = np.array([0, 255, 255], dtype=np.uint8)
    return Image.fromarray(output, mode="RGB")


def tile(image: Image.Image, title: str, size: tuple[int, int] = (300, 260)) -> Image.Image:
    title_height = 34
    canvas = Image.new("RGB", (size[0], size[1] + title_height), (255, 255, 255))
    contained = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    left = (size[0] - contained.width) // 2
    top = title_height + (size[1] - contained.height) // 2
    canvas.paste(contained, (left, top))
    ImageDraw.Draw(canvas).text((6, 10), title, fill=(0, 0, 0))
    return canvas


def render(row: dict[str, Any], output: Path) -> None:
    image_path = Path(row["image"])
    if sha256_file(image_path) != row["image_sha256"]:
        raise ValueError(f"image hash mismatch: {row['panel_index']}")
    image = Image.open(image_path).convert("RGB")
    layers = row["layers_evaluated"]
    target = row["target_choice"]
    fraction = float(row["fraction"])
    columns = len(layers) + 1
    tiles: list[list[Image.Image]] = [[], []]
    tiles[0].append(tile(
        image,
        f"case {row['panel_index']} target {target} pred {row['baseline_predicted_choice']}",
    ))
    tiles[1].append(tile(image, "original; cyan = top-area mask"))
    for layer in layers:
        tiles[0].append(tile(
            overlay(image, row["attribution"]["raw_attention_layers"][layer], fraction),
            f"raw attention L{layer}",
        ))
        tiles[1].append(tile(
            overlay(
                image,
                row["attribution"]["positive_option_layers"][target][layer],
                fraction,
            ),
            f"target {target} grad x attention L{layer}",
        ))
    tile_width, tile_height = tiles[0][0].size
    canvas = Image.new("RGB", (tile_width * columns, tile_height * 2), (255, 255, 255))
    for row_index, row_tiles in enumerate(tiles):
        for column, panel in enumerate(row_tiles):
            canvas.paste(panel, (column * tile_width, row_index * tile_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-index", type=int, action="append", required=True)
    args = parser.parse_args()
    requested = list(dict.fromkeys(args.panel_index))
    rows: dict[int, dict[str, Any]] = {}
    with args.case_results.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["panel_index"] in requested:
                    rows[row["panel_index"]] = row
    missing = sorted(set(requested) - set(rows))
    if missing:
        raise ValueError(f"requested cases are absent: {missing}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to mix rendered outputs: {args.output_dir}")
    for panel_index in requested:
        render(rows[panel_index], args.output_dir / f"case_{panel_index:02d}_layer_profile.png")
    print(json.dumps({
        "status": "completed", "case_count": len(requested),
        "output_dir": str(args.output_dir.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
