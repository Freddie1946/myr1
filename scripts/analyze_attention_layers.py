#!/usr/bin/env python3
"""Render and quantify saved text-to-vision attention maps layer by layer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from compute_external_roi_overlap import (
    grid_box,
    intersection_area,
    mask_area,
    overlap,
    reference_boxes,
)


METHODS = {
    "last_query": "last_query_layer_maps",
    "blog_text_query": "blog_text_query_layer_maps",
    "question_options_query": "question_options_query_layer_maps",
}

LAYER_BANDS = {
    "early_0_7": (0, 8),
    "middle_8_17": (8, 18),
    "late_18_27": (18, 28),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-run", type=Path, action="append", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--annotation-model", default="claude-sonnet-4-6")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resized(values: np.ndarray, image: Image.Image) -> np.ndarray:
    return np.asarray(
        Image.fromarray(values.astype(np.float32), mode="F").resize(
            image.size, resample=Image.Resampling.BILINEAR
        )
    )


def roi_attention_enrichment(
    values: np.ndarray, reference: list[tuple[float, float, float, float]]
) -> float:
    rows, columns = values.shape
    patch_area = 1.0 / (rows * columns)
    reference_area = mask_area(reference)
    if reference_area <= 0:
        return 0.0
    roi_mass = 0.0
    normalized = values / values.sum()
    for index, value in enumerate(normalized.reshape(-1)):
        patch = grid_box(index, rows, columns)
        intersections = []
        for box in reference:
            if intersection_area(patch, box) > 0:
                intersections.append(
                    (
                        max(patch[0], box[0]),
                        max(patch[1], box[1]),
                        min(patch[2], box[2]),
                        min(patch[3], box[3]),
                    )
                )
        fraction_inside = mask_area(intersections) / patch_area
        roi_mass += float(value) * fraction_inside
    return roi_mass / reference_area


def layer_metrics(
    maps: list[np.ndarray],
    reference: list[tuple[float, float, float, float]],
    top_fraction: float,
) -> list[dict[str, Any]]:
    output = []
    for layer, values in enumerate(maps):
        rows, columns = values.shape
        top_count = max(1, math.ceil(values.size * top_fraction))
        order = np.argsort(-values.reshape(-1)).tolist()
        metrics = overlap(
            reference, order[:top_count], rows=rows, columns=columns
        )
        metrics["roi_attention_enrichment"] = roi_attention_enrichment(values, reference)
        metrics["layer"] = layer
        metrics["top_patch_count"] = top_count
        output.append(metrics)
    return output


def render_layer_grid(
    path: Path,
    image: Image.Image,
    maps: list[np.ndarray],
    metrics: list[dict[str, Any]],
    *,
    title: str,
    shared_scale: bool,
) -> None:
    columns = 4
    rows = math.ceil(len(maps) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16, 4 * rows), constrained_layout=True)
    flat_axes = np.asarray(axes).reshape(-1)
    rendered = [resized(values, image) for values in maps]
    shared_vmax = max(float(values.max()) for values in rendered)
    for layer, (axis, heat, row) in enumerate(zip(flat_axes, rendered, metrics)):
        axis.imshow(image)
        axis.imshow(
            heat,
            cmap="inferno",
            alpha=0.52,
            vmin=0,
            vmax=shared_vmax if shared_scale else float(heat.max()) or 1.0,
        )
        hit = "Y" if row["pointing_game_hit"] else "N"
        axis.set_title(
            f"Layer {layer} | IoU {row['iou']:.3f} | hit {hit} | enrich {row['roi_attention_enrichment']:.2f}x",
            fontsize=9,
        )
        axis.axis("off")
    for axis in flat_axes[len(maps) :]:
        axis.axis("off")
    scale = "shared scale" if shared_scale else "per-layer scale"
    figure.suptitle(f"{title} ({scale})", fontsize=15)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def render_metric_curves(path: Path, records: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    metric_specs = (
        ("iou", "Top-area IoU", (0, None)),
        ("pointing_game_hit", "Pointing-game hit rate", (0, 1.05)),
        ("roi_attention_enrichment", "ROI attention enrichment", (0, None)),
    )
    colors = {
        "last_query": "#3366cc",
        "blog_text_query": "#cc5500",
        "question_options_query": "#138a63",
    }
    for method in METHODS:
        method_rows = [row for row in records if row["method"] == method]
        for axis, (metric, label, limits) in zip(axes, metric_specs):
            values = np.asarray(
                [[float(layer[metric]) for layer in row["layers"]] for row in method_rows]
            )
            layers = np.arange(values.shape[1])
            for case_values in values:
                axis.plot(layers, case_values, color=colors[method], alpha=0.18, linewidth=1)
            axis.plot(
                layers,
                values.mean(axis=0),
                color=colors[method],
                linewidth=2.5,
                label=f"{method} mean",
            )
            axis.set_title(label)
            axis.set_xlabel("Language-model layer (zero-based)")
            axis.set_xlim(0, values.shape[1] - 1)
            axis.set_ylim(*limits)
            axis.grid(alpha=0.25)
    for axis in axes:
        axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def summarize_layer_bands(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    metrics = ("iou", "pointing_game_hit", "roi_attention_enrichment")
    for method in METHODS:
        method_rows = [row for row in records if row["method"] == method]
        if not method_rows:
            continue
        layer_count = len(method_rows[0]["layers"])
        for band, (start, stop) in LAYER_BANDS.items():
            if stop > layer_count:
                raise ValueError(
                    f"layer band {band} requires {stop} layers, found {layer_count}"
                )
            case_means = {
                metric: [
                    float(np.mean([layer[metric] for layer in row["layers"][start:stop]]))
                    for row in method_rows
                ]
                for metric in metrics
            }
            summaries.append(
                {
                    "method": method,
                    "band": band,
                    "layers": list(range(start, stop)),
                    "case_count": len(method_rows),
                    "case_mean_values": case_means,
                    "mean": {
                        metric: float(np.mean(values))
                        for metric, values in case_means.items()
                    },
                    "sample_std": {
                        metric: float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                        for metric, values in case_means.items()
                    },
                    "cases_with_roi_enrichment_above_one": int(
                        sum(value > 1.0 for value in case_means["roi_attention_enrichment"])
                    ),
                }
            )
    return summaries


def main() -> None:
    args = parse_args()
    if not 0 < args.top_fraction <= 1:
        raise ValueError("top-fraction must be in (0, 1]")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    attention_records: dict[int, dict[str, Any]] = {}
    attention_inputs = []
    for run in args.attention_run:
        source = run / "attention_results.json"
        payload = json.loads(source.read_text())
        attention_inputs.append({"path": str(source.resolve()), "sha256": sha256_file(source)})
        for row in payload["records"]:
            attention_records[row["panel_index"]] = row

    annotation_payload = json.loads(args.annotations.read_text())
    validated_annotations = {
        row["panel_index"]: row["annotation"]
        for row in annotation_payload["results"]
        if row.get("requested_model") == args.annotation_model
        and row.get("status") == "validated"
        and isinstance(row.get("annotation"), dict)
    }
    missing = sorted(set(attention_records) - set(validated_annotations))
    if missing:
        raise ValueError(f"missing validated annotations for cases: {missing}")

    output_records = []
    excluded_spatial_cases = []
    for panel_index in sorted(attention_records):
        row = attention_records[panel_index]
        annotation = validated_annotations[panel_index]
        if annotation.get("evidence_type") not in {"focal", "multifocal"}:
            excluded_spatial_cases.append(
                {
                    "panel_index": panel_index,
                    "evidence_type": annotation.get("evidence_type"),
                    "reason": annotation.get("reason"),
                }
            )
            continue
        image = Image.open(row["image"]).convert("RGB")
        boxes = reference_boxes(annotation)
        for method, field in METHODS.items():
            maps = [np.asarray(values, dtype=np.float64) for values in row[field]]
            metrics = layer_metrics(maps, boxes, args.top_fraction)
            stem = f"case_{panel_index:02d}_{method}_layers"
            render_layer_grid(
                args.output_dir / f"{stem}_shared_scale.png",
                image,
                maps,
                metrics,
                title=f"Case {panel_index:02d}: {method}",
                shared_scale=True,
            )
            render_layer_grid(
                args.output_dir / f"{stem}_per_layer_scale.png",
                image,
                maps,
                metrics,
                title=f"Case {panel_index:02d}: {method}",
                shared_scale=False,
            )
            output_records.append(
                {
                    "panel_index": panel_index,
                    "method": method,
                    "grid_rows": row["grid_rows"],
                    "grid_columns": row["grid_columns"],
                    "layers": metrics,
                }
            )

    if not output_records:
        raise ValueError("no focal or multifocal cases are spatially evaluable")
    render_metric_curves(args.output_dir / "layer_metric_curves.png", output_records)
    layer_band_summary = summarize_layer_bands(output_records)
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_pilot",
        "formal_result": False,
        "selection_policy": "no_primary_layer_selected",
        "layer_band_policy": {
            "kind": "predefined_descriptive_bands_not_layer_selection",
            "bands": {name: list(bounds) for name, bounds in LAYER_BANDS.items()},
        },
        "top_fraction": args.top_fraction,
        "annotation_kind": "external_model_pseudo_reference_roi_not_expert_ground_truth",
        "annotation_model": args.annotation_model,
        "input_case_count": len(attention_records),
        "spatially_evaluable_case_count": len(attention_records) - len(excluded_spatial_cases),
        "excluded_spatial_cases": excluded_spatial_cases,
        "annotations": {
            "path": str(args.annotations.resolve()),
            "sha256": sha256_file(args.annotations),
        },
        "attention_inputs": attention_inputs,
        "records": output_records,
        "layer_band_summary": layer_band_summary,
    }
    (args.output_dir / "layer_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "cases": len(attention_records),
                "spatially_evaluable_cases": result["spatially_evaluable_case_count"],
                "methods": len(METHODS),
                "output": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
