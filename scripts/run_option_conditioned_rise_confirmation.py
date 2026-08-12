#!/usr/bin/env python3
"""Confirm ground-truth-option spatial support with black-box RISE."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from PIL import Image, ImageFilter

from run_option_conditioned_visual_evidence import (
    OPTION_LETTERS,
    OptionEvidenceModel,
    canonical_json,
    margins,
    model_prompt,
    sha256_file,
)
from run_visual_understanding_counterfactual import bootstrap_ci


FRACTIONS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)


def trapezoid_auc(xs: list[float], ys: list[float]) -> float:
    return float(np.trapezoid(np.asarray(ys, dtype=np.float64), np.asarray(xs, dtype=np.float64)))


def rise_soft_masks(
    width: int, height: int, *, count: int, cells: int,
    probability: float, seed: int,
) -> np.ndarray:
    if count <= 0 or count % 2 or cells <= 1 or not 0 < probability < 1:
        raise ValueError("RISE requires positive even count, cells > 1, probability in (0,1)")
    rng = np.random.default_rng(seed)
    cell_w, cell_h = math.ceil(width / cells), math.ceil(height / cells)
    up_w, up_h = (cells + 1) * cell_w, (cells + 1) * cell_h
    output = []
    for _ in range(count // 2):
        coarse = (rng.random((cells, cells)) < probability).astype(np.float32)
        offset_x, offset_y = int(rng.integers(0, cell_w)), int(rng.integers(0, cell_h))
        for values in (coarse, 1.0 - coarse):
            enlarged = Image.fromarray(values, mode="F").resize((up_w, up_h), Image.Resampling.BILINEAR)
            output.append(np.asarray(
                enlarged.crop((offset_x, offset_y, offset_x + width, offset_y + height)),
                dtype=np.float32,
            ))
    return np.stack(output)


def blend(source: np.ndarray, base: np.ndarray, mask: np.ndarray) -> Image.Image:
    values = source * mask[..., None] + base * (1.0 - mask[..., None])
    return Image.fromarray(np.clip(np.rint(values), 0, 255).astype(np.uint8), mode="RGB")


def exact_area_mask(values: np.ndarray, fraction: float, *, high: bool) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    count = min(len(flat), max(0, int(round(fraction * len(flat)))))
    selected = np.zeros(len(flat), dtype=bool)
    if count:
        order = np.argsort(-flat if high else flat, kind="stable")
        selected[order[:count]] = True
    return selected.reshape(values.shape)


def shifted_controls(mask: np.ndarray, count: int, seed: int) -> list[np.ndarray]:
    rng = random.Random(seed)
    height, width = mask.shape
    controls, seen = [], set()
    while len(controls) < count:
        shift_y, shift_x = rng.randrange(height), rng.randrange(width)
        if (shift_y, shift_x) in seen or (shift_y == 0 and shift_x == 0):
            continue
        seen.add((shift_y, shift_x))
        controls.append(np.roll(mask, (shift_y, shift_x), axis=(0, 1)))
    return controls


def intervene(source: np.ndarray, base: np.ndarray, mask: np.ndarray, mode: str) -> Image.Image:
    if mode == "deletion":
        return blend(source, base, (~mask).astype(np.float32))
    if mode == "retention":
        return blend(source, base, mask.astype(np.float32))
    raise ValueError(mode)


def case_seed(case_sha: str, suffix: str) -> int:
    return int(hashlib.sha256(f"option-rise-v1:{case_sha}:{suffix}".encode()).hexdigest()[:16], 16)


def analysis_size(width: int, height: int, long_edge: int) -> tuple[int, int]:
    scale = min(1.0, long_edge / max(width, height))
    return max(8, int(round(width * scale))), max(8, int(round(height * scale)))


def normalized_saliency(scores: np.ndarray, masks: np.ndarray) -> np.ndarray:
    weighted = np.tensordot(np.asarray(scores, dtype=np.float64), masks, axes=(0, 0))
    values = weighted / masks.sum(axis=0).clip(min=1e-6)
    span = float(values.max() - values.min())
    return ((values - values.min()) / max(span, 1e-12)).astype(np.float32)


def option_saliency_maps(
    probabilities: list[list[float]], masks: np.ndarray,
) -> dict[str, np.ndarray]:
    option_margins = np.asarray([margins(row) for row in probabilities], dtype=np.float64)
    return {
        letter: normalized_saliency(option_margins[:, index], masks)
        for index, letter in enumerate(OPTION_LETTERS)
    }


def upsample_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        Image.fromarray(mask.astype(np.float32), mode="F").resize(size, Image.Resampling.BILINEAR),
        dtype=np.float32,
    )


def score_soft_masks(
    scorer: OptionEvidenceModel,
    prompt: str,
    source: np.ndarray,
    blur: np.ndarray,
    masks: np.ndarray,
    size: tuple[int, int],
    chunk_size: int = 8,
) -> list[list[float]]:
    output: list[list[float]] = []
    for start in range(0, len(masks), chunk_size):
        images = [
            blend(source, blur, upsample_mask(mask, size))
            for mask in masks[start:start + chunk_size]
        ]
        output.extend(scorer.score(prompt, images))
    return output


def curve_record(probabilities: list[list[float]], target: int) -> dict[str, Any]:
    option_margins = [margins(row) for row in probabilities]
    target_margins = [row[target] for row in option_margins]
    target_probabilities = [row[target] for row in probabilities]
    predictions = [int(np.argmax(row)) for row in probabilities]
    return {
        "option_probabilities": probabilities,
        "option_margins": [dict(zip(OPTION_LETTERS, row)) for row in option_margins],
        "target_margins": target_margins,
        "target_probabilities": target_probabilities,
        "predicted_choices": [OPTION_LETTERS[index] for index in predictions],
        "correct": [index == target for index in predictions],
        "target_margin_auc": trapezoid_auc(list(FRACTIONS), target_margins),
        "target_probability_auc": trapezoid_auc(list(FRACTIONS), target_probabilities),
    }


def evaluate_case(
    scorer: OptionEvidenceModel,
    case: dict[str, Any],
    *,
    rise_count: int,
    rise_cells: int,
    random_controls: int,
    attribution_long_edge: int,
) -> dict[str, Any]:
    path = Path(case["image"])
    if sha256_file(path) != case["image_sha256"]:
        raise ValueError(f"image hash mismatch: {case['panel_index']}")
    source_image = Image.open(path).convert("RGB")
    source = np.asarray(source_image, dtype=np.float32)
    blur_image = source_image.filter(
        ImageFilter.GaussianBlur(radius=max(2.0, min(source_image.size) * 0.035))
    )
    blur = np.asarray(blur_image, dtype=np.float32)
    mean = np.broadcast_to(source.mean(axis=(0, 1)), source.shape).astype(np.float32)
    prompt = model_prompt(scorer.processor, case["problem"])
    baseline, baseline_repeat = scorer.score(prompt, [source_image, source_image])
    target = OPTION_LETTERS.index(case["target_choice"])
    map_size = analysis_size(source_image.width, source_image.height, attribution_long_edge)
    masks = rise_soft_masks(
        map_size[0], map_size[1], count=rise_count, cells=rise_cells,
        probability=0.5, seed=case_seed(case["source_record_sha256"], "option-maps"),
    )
    masked_probabilities = score_soft_masks(
        scorer, prompt, source, blur, masks, source_image.size,
    )
    maps = option_saliency_maps(masked_probabilities, masks)
    target_saliency = upsample_mask(maps[case["target_choice"]], source_image.size)
    curves: dict[str, Any] = {}
    sentinel_intervention_diff: float | None = None
    for mode in ("deletion", "retention"):
        curves[mode] = {}
        for strategy, high in (("rise_high", True), ("rise_low", False)):
            binary = [exact_area_mask(target_saliency, fraction, high=high) for fraction in FRACTIONS]
            images = [intervene(source, mean, mask, mode) for mask in binary]
            probabilities = scorer.score(prompt, images)
            curves[mode][strategy] = curve_record(probabilities, target)
            if mode == "deletion" and strategy == "rise_high":
                repeated = scorer.score(prompt, [images[2]])[0]
                sentinel_intervention_diff = float(np.max(
                    np.abs(np.asarray(probabilities[2]) - np.asarray(repeated))
                ))
        random_rows = []
        shifted_by_fraction = {
            fraction: shifted_controls(
                exact_area_mask(target_saliency, fraction, high=True),
                random_controls,
                case_seed(case["source_record_sha256"], f"option-shift:{fraction}"),
            )
            for fraction in FRACTIONS
        }
        for control in range(random_controls):
            images = [
                intervene(source, mean, shifted_by_fraction[fraction][control], mode)
                for fraction in FRACTIONS
            ]
            random_rows.append(curve_record(scorer.score(prompt, images), target))
        curves[mode]["random_shifted_shape_matched"] = random_rows
    masked_repeat = scorer.score(
        prompt, [blend(source, blur, upsample_mask(masks[0], source_image.size))]
    )[0]
    return {
        "panel_index": case["panel_index"],
        "source_index": case["index"],
        "source_record_sha256": case["source_record_sha256"],
        "image": str(path),
        "image_sha256": case["image_sha256"],
        "target_choice": case["target_choice"],
        "baseline_option_probabilities": dict(zip(OPTION_LETTERS, baseline)),
        "baseline_option_margins": dict(zip(OPTION_LETTERS, margins(baseline))),
        "baseline_predicted_choice": OPTION_LETTERS[int(np.argmax(baseline))],
        "baseline_correct": int(np.argmax(baseline)) == target,
        "baseline_repeatability_max_abs_probability_diff": float(np.max(
            np.abs(np.asarray(baseline) - np.asarray(baseline_repeat))
        )),
        "masked_sentinel_repeatability_max_abs_probability_diff": float(np.max(
            np.abs(np.asarray(masked_probabilities[0]) - np.asarray(masked_repeat))
        )),
        "intervention_sentinel_repeatability_max_abs_probability_diff": sentinel_intervention_diff,
        "rise_mask_count": rise_count,
        "rise_cells": rise_cells,
        "attribution_map_size": list(map_size),
        "option_saliency_maps": {letter: values.tolist() for letter, values in maps.items()},
        "fractions": list(FRACTIONS),
        "curves": curves,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "case_count": len(records),
        "baseline_accuracy": float(np.mean([row["baseline_correct"] for row in records])),
        "strategy_fidelity": {},
    }
    for mode in ("deletion", "retention"):
        result["strategy_fidelity"][mode] = {}
        for endpoint in ("target_margin_auc", "target_probability_auc"):
            random_values = [
                float(np.mean([
                    control[endpoint]
                    for control in row["curves"][mode]["random_shifted_shape_matched"]
                ]))
                for row in records
            ]
            result["strategy_fidelity"][mode][endpoint] = {}
            for strategy in ("rise_high", "rise_low"):
                observed = [row["curves"][mode][strategy][endpoint] for row in records]
                advantage = [
                    random - value if mode == "deletion" else value - random
                    for value, random in zip(observed, random_values)
                ]
                result["strategy_fidelity"][mode][endpoint][strategy] = {
                    "mean_auc": float(np.mean(observed)),
                    "mean_random_auc": float(np.mean(random_values)),
                    "mean_fidelity_advantage": float(np.mean(advantage)),
                    "fidelity_advantage_bootstrap_95ci": bootstrap_ci(
                        advantage, f"option-rise:{mode}:{endpoint}:{strategy}"
                    ),
                    "positive_rate": float(np.mean(np.asarray(advantage) > 0)),
                }
    deletion = []
    retention = []
    for row in records:
        for mode, output in (("deletion", deletion), ("retention", retention)):
            random_auc = float(np.mean([
                control["target_margin_auc"]
                for control in row["curves"][mode]["random_shifted_shape_matched"]
            ]))
            observed = row["curves"][mode]["rise_high"]["target_margin_auc"]
            output.append(random_auc - observed if mode == "deletion" else observed - random_auc)
    minimum = np.minimum(deletion, retention)
    result["bidirectional_target_margin"] = {
        "both_positive_rate": float(np.mean((np.asarray(deletion) > 0) & (np.asarray(retention) > 0))),
        "mean_minimum_direction_advantage": float(np.mean(minimum)),
        "minimum_direction_bootstrap_95ci": bootstrap_ci(
            minimum.tolist(), "option-rise:bidirectional-minimum"
        ),
    }
    return result


def load_partial(path: Path, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected = {row["panel_index"]: row for row in cases}
    records = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    seen: set[int] = set()
    for row in records:
        index = row.get("panel_index")
        if index not in expected or index in seen:
            raise ValueError(f"unexpected or duplicate partial case: {index}")
        if row.get("image_sha256") != expected[index]["image_sha256"]:
            raise ValueError(f"partial image identity mismatch: {index}")
        seen.add(index)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--expected-panel-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-index", type=int, action="append")
    parser.add_argument("--rise-count", type=int, default=256)
    parser.add_argument("--rise-cells", type=int, default=7)
    parser.add_argument("--random-controls", type=int, default=5)
    parser.add_argument("--attribution-long-edge", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    if args.resume and not args.output_dir.exists():
        raise FileNotFoundError(f"resume directory missing: {args.output_dir}")
    if args.batch_size != 1:
        raise ValueError("formal option-conditioned RISE requires batch-size 1")
    panel_sha = sha256_file(args.panel)
    if panel_sha != args.expected_panel_sha256:
        raise ValueError("frozen panel SHA-256 mismatch")
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    if panel.get("case_count") != 96 or panel.get("selection_uses_model_outputs") is not False:
        raise ValueError("requires the frozen model-blind 96-case panel")
    selected = set(args.panel_index or range(96))
    cases = [row for row in panel["cases"] if row["panel_index"] in selected]
    if len(cases) != len(selected):
        raise ValueError("requested panel index missing")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    case_path = args.output_dir / "case_results.jsonl"
    metrics_path = args.output_dir / "metrics.json"
    if args.resume and metrics_path.exists():
        raise FileExistsError("run already contains final metrics")
    records = load_partial(case_path, cases) if args.resume else []
    complete = {row["panel_index"] for row in records}
    scorer = OptionEvidenceModel(args.model, batch_size=1)
    with case_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for case in cases:
            if case["panel_index"] in complete:
                continue
            row = evaluate_case(
                scorer, case, rise_count=args.rise_count, rise_cells=args.rise_cells,
                random_controls=args.random_controls,
                attribution_long_edge=args.attribution_long_edge,
            )
            records.append(row)
            handle.write(canonical_json(row) + "\n")
            handle.flush()
            print(canonical_json({
                "completed": len(records), "total": len(cases),
                "panel_index": case["panel_index"],
            }), flush=True)
    records.sort(key=lambda row: row["panel_index"])
    metrics = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_confirmation" if len(cases) == 96 else "completed_smoke",
        "formal_result": False,
        "method": "ground_truth_option_margin_rise_blur_generation_mean_fill_validation_v1",
        "attribution_target": "each_ABCD_option_margin",
        "causal_validation_target": "ground_truth_option_margin",
        "model_label": args.model_label,
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "panel_path": str(args.panel.resolve()),
        "panel_sha256": panel_sha,
        "panel_indices": [row["panel_index"] for row in records],
        "rise_mask_count": args.rise_count,
        "rise_cells": args.rise_cells,
        "rise_visibility_probability": 0.5,
        "attribution_long_edge": args.attribution_long_edge,
        "validation_perturbation": "hard_mean_fill",
        "random_control": "circular_shift_preserves_binary_mask_area_and_shape",
        "random_controls": args.random_controls,
        "score_batch_size": 1,
        "software_environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "attention_implementation": "eager",
        },
        "max_baseline_repeatability_abs_probability_diff": max(
            row["baseline_repeatability_max_abs_probability_diff"] for row in records
        ),
        "max_masked_sentinel_repeatability_abs_probability_diff": max(
            row["masked_sentinel_repeatability_max_abs_probability_diff"] for row in records
        ),
        "max_intervention_sentinel_repeatability_abs_probability_diff": max(
            row["intervention_sentinel_repeatability_max_abs_probability_diff"] for row in records
        ),
        "case_results": str(case_path.resolve()),
        "case_results_sha256": sha256_file(case_path),
        **aggregate(records),
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(canonical_json(metrics), flush=True)


if __name__ == "__main__":
    main()
