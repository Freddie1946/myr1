#!/usr/bin/env python3
"""Compare external pseudo-reference boxes with raw patch-occlusion rankings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def rectangle_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    return rectangle_area(
        (max(left[0], right[0]), max(left[1], right[1]), min(left[2], right[2]), min(left[3], right[3]))
    )


def mask_area(boxes: list[tuple[float, float, float, float]]) -> float:
    xs = sorted({coordinate for box in boxes for coordinate in (box[0], box[2])})
    if len(xs) < 2:
        return 0.0
    area = 0.0
    for x0, x1 in zip(xs, xs[1:]):
        intervals = sorted(
            (box[1], box[3]) for box in boxes if box[0] < x1 and box[2] > x0
        )
        covered = 0.0
        start = end = None
        for y0, y1 in intervals:
            if start is None:
                start, end = y0, y1
            elif y0 > end:
                covered += end - start
                start, end = y0, y1
            else:
                end = max(end, y1)
        if start is not None:
            covered += end - start
        area += (x1 - x0) * covered
    return area


def grid_box(index: int, rows: int = 6, columns: int = 6) -> tuple[float, float, float, float]:
    row, column = divmod(index, columns)
    return (column / columns, row / rows, (column + 1) / columns, (row + 1) / rows)


def reference_boxes(annotation: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    return [
        (box["x_min"] / 1000, box["y_min"] / 1000, box["x_max"] / 1000, box["y_max"] / 1000)
        for box in annotation["boxes"]
    ]


def overlap(
    reference: list[tuple[float, float, float, float]],
    top_indices: list[int],
    *,
    rows: int = 6,
    columns: int = 6,
) -> dict[str, float | bool]:
    selected = [grid_box(index, rows, columns) for index in top_indices]
    ref_area = mask_area(reference)
    selected_area = mask_area(selected)
    intersections = []
    for ref in reference:
        for patch in selected:
            area = intersection_area(ref, patch)
            if area:
                intersections.append((max(ref[0], patch[0]), max(ref[1], patch[1]), min(ref[2], patch[2]), min(ref[3], patch[3])))
    intersection = mask_area(intersections)
    union = ref_area + selected_area - intersection
    top_patch = selected[0]
    center = ((top_patch[0] + top_patch[2]) / 2, (top_patch[1] + top_patch[3]) / 2)
    return {
        "iou": intersection / union if union else 0.0,
        "roi_coverage": intersection / ref_area if ref_area else 0.0,
        "selected_precision": intersection / selected_area if selected_area else 0.0,
        "pointing_game_hit": any(
            box[0] <= center[0] <= box[2] and box[1] <= center[1] <= box[3]
            for box in reference
        ),
        "reference_area_fraction": ref_area,
        "selected_area_fraction": selected_area,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--case-results", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations = json.loads(args.annotations.read_text())
    case_rows = {
        row["panel_index"]: row
        for row in (json.loads(line) for line in args.case_results.open())
    }
    output = []
    for row in annotations["results"]:
        annotation = row["annotation"]
        result = {
            "panel_index": row["panel_index"],
            "requested_model": row["requested_model"],
            "served_model": row["served_model"],
            "annotation_status": row.get("status"),
        }
        if not isinstance(annotation, dict):
            result["overlap_status"] = "not_applicable_invalid_annotation"
            output.append(result)
            continue
        result["evidence_type"] = annotation["evidence_type"]
        if annotation["evidence_type"] in {"focal", "multifocal"}:
            boxes = reference_boxes(annotation)
            order = case_rows[row["panel_index"]]["top_patch_order"]
            result["top_10pct"] = overlap(boxes, order[:4])
            result["top_25pct"] = overlap(boxes, order[:9])
        else:
            result["overlap_status"] = "not_applicable_without_spatial_boxes"
        output.append(result)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
