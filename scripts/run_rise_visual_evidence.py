#!/usr/bin/env python3
"""Generate RISE visual evidence maps and independently test their fidelity.

RISE maps are estimated with blurred-background random soft masks.  Their
rankings are then evaluated with hard mean-fill deletion/retention, so the
attribution generator and its principal validation perturbation are distinct.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from run_attention_intervention_experiment import DecisionScorer
from run_visual_fidelity_experiment import (
    canonical_json,
    model_prompt,
    sha256_file,
    trapezoid_auc,
)


FRACTIONS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
LETTERS = "ABCD"


def rise_soft_masks(
    width: int,
    height: int,
    *,
    count: int,
    cells: int,
    probability: float,
    seed: int,
) -> np.ndarray:
    """Return antithetic RISE masks with random upsample/crop translations."""
    if count <= 0 or count % 2 or cells <= 1 or not 0 < probability < 1:
        raise ValueError("RISE requires a positive even count, cells > 1 and probability in (0,1)")
    rng = np.random.default_rng(seed)
    cell_w = math.ceil(width / cells)
    cell_h = math.ceil(height / cells)
    up_w = (cells + 1) * cell_w
    up_h = (cells + 1) * cell_h
    output = []
    for _ in range(count // 2):
        coarse = (rng.random((cells, cells)) < probability).astype(np.float32)
        offset_x = int(rng.integers(0, cell_w))
        offset_y = int(rng.integers(0, cell_h))
        for values in (coarse, 1.0 - coarse):
            enlarged = Image.fromarray(values, mode="F").resize(
                (up_w, up_h), resample=Image.Resampling.BILINEAR
            )
            output.append(
                np.asarray(
                    enlarged.crop((offset_x, offset_y, offset_x + width, offset_y + height)),
                    dtype=np.float32,
                )
            )
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
    controls = []
    seen = set()
    while len(controls) < count:
        shift_y = rng.randrange(height)
        shift_x = rng.randrange(width)
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


def score_images(scorer: DecisionScorer, prompt: str, images: list[Image.Image]) -> list[list[float]]:
    return scorer.score(prompt, images)


def curve_record(
    probabilities: list[list[float]], decision: int, target: int
) -> dict[str, Any]:
    predictions = [int(np.argmax(row)) for row in probabilities]
    values = [float(row[decision]) for row in probabilities]
    return {
        "option_probabilities": probabilities,
        "original_decision_probability": values,
        "predicted_choices": [LETTERS[index] for index in predictions],
        "answer_changed": [index != decision for index in predictions],
        "correct": [index == target for index in predictions],
        "auc": trapezoid_auc(list(FRACTIONS), values),
    }


def case_seed(case_sha: str, suffix: str) -> int:
    return int(hashlib.sha256(f"rise-v1:{case_sha}:{suffix}".encode()).hexdigest()[:16], 16)


def evaluate_case(
    scorer: DecisionScorer,
    case: dict[str, Any],
    mismatch_case: dict[str, Any],
    *,
    rise_count: int,
    rise_cells: int,
    random_controls: int,
) -> dict[str, Any]:
    path = Path(case["image"])
    if sha256_file(path) != case["image_sha256"]:
        raise ValueError(f"image hash mismatch for case {case['panel_index']}")
    source_image = Image.open(path).convert("RGB")
    source = np.asarray(source_image, dtype=np.float32)
    radius = max(2.0, min(source_image.size) * 0.035)
    blur_image = source_image.filter(ImageFilter.GaussianBlur(radius=radius))
    blur = np.asarray(blur_image, dtype=np.float32)
    mean_rgb = np.mean(source, axis=(0, 1))
    mean = np.broadcast_to(mean_rgb, source.shape).astype(np.float32)
    prompt = model_prompt(scorer.processor, case["problem"])
    baseline = score_images(scorer, prompt, [source_image])[0]
    decision = int(np.argmax(baseline))
    target = LETTERS.index(case["target_choice"])

    masks = rise_soft_masks(
        source_image.width,
        source_image.height,
        count=rise_count,
        cells=rise_cells,
        probability=0.5,
        seed=case_seed(case["source_record_sha256"], "masks"),
    )
    masked_probabilities = score_images(
        scorer, prompt, [blend(source, blur, values) for values in masks]
    )
    decision_scores = np.asarray([row[decision] for row in masked_probabilities], dtype=np.float64)
    # Conventional RISE conditional visibility score. Denominator handles edge
    # variation introduced by random translations.
    weighted = np.tensordot(decision_scores, masks, axes=(0, 0))
    denominator = masks.sum(axis=0).clip(min=1e-6)
    saliency = weighted / denominator
    saliency = (saliency - saliency.min()) / max(float(saliency.max() - saliency.min()), 1e-12)

    curves: dict[str, Any] = {}
    for mode in ("deletion", "retention"):
        curves[mode] = {}
        for strategy, high in (("rise_high", True), ("rise_low", False)):
            masks_by_fraction = [exact_area_mask(saliency, fraction, high=high) for fraction in FRACTIONS]
            probabilities = score_images(
                scorer,
                prompt,
                [intervene(source, mean, values, mode) for values in masks_by_fraction],
            )
            curves[mode][strategy] = curve_record(probabilities, decision, target)
        random_rows = []
        # Circularly shifting each high-saliency binary mask preserves its exact
        # area and spatial morphology while breaking image alignment.
        for control in range(random_controls):
            images = []
            for fraction in FRACTIONS:
                high_mask = exact_area_mask(saliency, fraction, high=True)
                shifted = shifted_controls(
                    high_mask,
                    random_controls,
                    case_seed(case["source_record_sha256"], f"shift:{fraction}"),
                )[control]
                images.append(intervene(source, mean, shifted, mode))
            random_rows.append(curve_record(score_images(scorer, prompt, images), decision, target))
        curves[mode]["random_shifted_shape_matched"] = random_rows

    mismatch = Image.open(mismatch_case["image"]).convert("RGB")
    controls = score_images(scorer, prompt, [blur_image, Image.fromarray(mean.astype(np.uint8)), mismatch])
    return {
        "panel_index": case["panel_index"],
        "source_index": case["index"],
        "image": case["image"],
        "image_sha256": case["image_sha256"],
        "target_choice": case["target_choice"],
        "baseline_option_probabilities": dict(zip(LETTERS, baseline)),
        "baseline_predicted_choice": LETTERS[decision],
        "baseline_correct": decision == target,
        "rise_mask_count": rise_count,
        "rise_cells": rise_cells,
        "rise_saliency_shape": list(saliency.shape),
        "rise_saliency": saliency.astype(np.float32).tolist(),
        "rise_masked_original_decision_probability_mean": float(decision_scores.mean()),
        "rise_masked_original_decision_probability_std": float(decision_scores.std(ddof=1)),
        "fractions": list(FRACTIONS),
        "curves": curves,
        "global_visual_dependency_controls": {
            "blur": dict(zip(LETTERS, controls[0])),
            "mean_image": dict(zip(LETTERS, controls[1])),
            "mismatched_image": dict(zip(LETTERS, controls[2])),
            "mismatched_panel_index": mismatch_case["panel_index"],
        },
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "case_count": len(records),
        "baseline_accuracy": float(np.mean([row["baseline_correct"] for row in records])),
        "strategy_fidelity": {},
        "global_visual_dependency": {},
    }
    for mode in ("deletion", "retention"):
        random_auc = [
            float(np.mean([x["auc"] for x in row["curves"][mode]["random_shifted_shape_matched"]]))
            for row in records
        ]
        result["strategy_fidelity"][mode] = {}
        for strategy in ("rise_high", "rise_low"):
            observed = [row["curves"][mode][strategy]["auc"] for row in records]
            advantage = [
                random - value if mode == "deletion" else value - random
                for value, random in zip(observed, random_auc)
            ]
            result["strategy_fidelity"][mode][strategy] = {
                "mean_auc": float(np.mean(observed)),
                "mean_random_auc": float(np.mean(random_auc)),
                "mean_fidelity_advantage": float(np.mean(advantage)),
                "sample_std_fidelity_advantage": float(np.std(advantage, ddof=1)) if len(advantage) > 1 else 0.0,
                "positive_cases": int(sum(value > 0 for value in advantage)),
            }
    for control in ("blur", "mean_image", "mismatched_image"):
        decision_drops = []
        changes = []
        for row in records:
            decision = LETTERS.index(row["baseline_predicted_choice"])
            baseline = row["baseline_option_probabilities"][LETTERS[decision]]
            values = row["global_visual_dependency_controls"][control]
            prediction = max(LETTERS, key=lambda letter: values[letter])
            decision_drops.append(baseline - values[LETTERS[decision]])
            changes.append(prediction != LETTERS[decision])
        result["global_visual_dependency"][control] = {
            "mean_original_decision_probability_drop": float(np.mean(decision_drops)),
            "answer_change_rate": float(np.mean(changes)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--expected-panel-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-index", type=int, action="append")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rise-count", type=int, default=256)
    parser.add_argument("--rise-cells", type=int, default=7)
    parser.add_argument("--random-controls", type=int, default=5)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run: {args.output_dir}")
    panel_sha = sha256_file(args.panel)
    if panel_sha != args.expected_panel_sha256:
        raise ValueError(f"panel hash mismatch: {panel_sha}")
    panel = json.loads(args.panel.read_text())
    if panel.get("case_count") != 24 or panel.get("selection_uses_model_outputs") is not False:
        raise ValueError("panel contract mismatch")
    selected = set(args.panel_index or range(24))
    cases = [row for row in panel["cases"] if row["panel_index"] in selected]
    if len(cases) != len(selected):
        raise ValueError("missing panel index")
    by_index = {row["panel_index"]: row for row in panel["cases"]}
    args.output_dir.mkdir(parents=True)
    scorer = DecisionScorer(args.model, args.batch_size)
    output_path = args.output_dir / "case_results.jsonl"
    records = []
    with output_path.open("w") as handle:
        for case in cases:
            mismatch = by_index[(case["panel_index"] + 1) % 24]
            row = evaluate_case(
                scorer,
                case,
                mismatch,
                rise_count=args.rise_count,
                rise_cells=args.rise_cells,
                random_controls=args.random_controls,
            )
            records.append(row)
            handle.write(canonical_json(row) + "\n")
            handle.flush()
            print(canonical_json({"completed": len(records), "total": len(cases), "panel_index": case["panel_index"]}), flush=True)
    metrics = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_exploratory" if len(cases) == 24 else "completed_smoke",
        "formal_result": False,
        "method": "rise_blur_generation_mean_fill_cross_perturbation_v1",
        "attribution_target": "model_original_decision_probability",
        "model_label": args.model_label,
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "panel_path": str(args.panel.resolve()),
        "panel_sha256": panel_sha,
        "panel_indices": [row["panel_index"] for row in cases],
        "rise_mask_count": args.rise_count,
        "rise_cells": args.rise_cells,
        "rise_visibility_probability": 0.5,
        "validation_perturbation": "hard_mean_fill",
        "random_control": "circular_shift_preserves_binary_mask_area_and_shape",
        "random_controls": args.random_controls,
        "case_results": str(output_path.resolve()),
        "case_results_sha256": sha256_file(output_path),
        **aggregate(records),
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(canonical_json(metrics))


if __name__ == "__main__":
    main()
