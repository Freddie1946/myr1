#!/usr/bin/env python3
"""Measure visual decision dependence across patch-shuffle granularities."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from run_option_conditioned_visual_evidence import (
    OPTION_LETTERS,
    OptionEvidenceModel,
    canonical_json,
    margins,
    model_prompt,
    sha256_file,
)
from run_visual_understanding_counterfactual import (
    bootstrap_ci,
    exact_mcnemar_p,
    patch_shuffle,
)


def pixel_multiset_digest(image: Image.Image) -> str:
    """Hash the sorted RGB-pixel multiset, ignoring spatial order."""
    array = np.asarray(image.convert("RGB"), dtype=np.uint32)
    packed = (array[..., 0] << 16) | (array[..., 1] << 8) | array[..., 2]
    return hashlib.sha256(np.sort(packed.reshape(-1)).tobytes()).hexdigest()


def evaluate_case(
    model: OptionEvidenceModel,
    case: dict[str, Any],
    cell_counts: list[int],
) -> dict[str, Any]:
    path = Path(case["image"])
    if sha256_file(path) != case["image_sha256"]:
        raise ValueError(f"image hash mismatch: {case['panel_index']}")
    original = Image.open(path).convert("RGB")
    original_array = np.asarray(original)
    original_multiset = pixel_multiset_digest(original)
    prompt = model_prompt(model.processor, case["problem"])
    first, repeated = model.score(prompt, [original, original])
    scores: dict[str, list[float]] = {"original": first}
    transform_audit: dict[str, Any] = {}
    for cells in cell_counts:
        seed = int(hashlib.sha256(
            f"patch-shuffle-granularity-v1:{case['source_record_sha256']}:{cells}".encode()
        ).hexdigest()[:16], 16)
        shuffled = patch_shuffle(original, cells, seed)
        if pixel_multiset_digest(shuffled) != original_multiset:
            raise ValueError(f"shuffle changed the pixel multiset: {case['panel_index']}:{cells}")
        condition = f"patch_shuffle_{cells}x{cells}"
        scores[condition] = model.score(prompt, [shuffled])[0]
        transform_audit[condition] = {
            "seed": seed,
            "pixel_multiset_preserved": True,
            "pixel_location_value_changed_rate": float(
                np.mean(np.any(np.asarray(shuffled) != original_array, axis=-1))
            ),
        }
    target = OPTION_LETTERS.index(case["target_choice"])
    conditions: dict[str, Any] = {}
    for condition, probabilities in scores.items():
        option_margins = margins(probabilities)
        prediction = int(np.argmax(probabilities))
        conditions[condition] = {
            "option_probabilities": dict(zip(OPTION_LETTERS, probabilities)),
            "option_margins": dict(zip(OPTION_LETTERS, option_margins)),
            "predicted_choice": OPTION_LETTERS[prediction],
            "correct": prediction == target,
            "target_probability": float(probabilities[target]),
            "target_margin": float(option_margins[target]),
        }
    return {
        "panel_index": case["panel_index"],
        "source_index": case["index"],
        "source_record_sha256": case["source_record_sha256"],
        "image": str(path),
        "image_sha256": case["image_sha256"],
        "target_choice": case["target_choice"],
        "original_repeatability_max_abs_probability_diff": float(
            np.max(np.abs(np.asarray(first) - np.asarray(repeated)))
        ),
        "transform_audit": transform_audit,
        "conditions": conditions,
    }


def aggregate(records: list[dict[str, Any]], cell_counts: list[int]) -> dict[str, Any]:
    original_accuracy = float(np.mean([
        row["conditions"]["original"]["correct"] for row in records
    ]))
    by_granularity: dict[str, Any] = {}
    for cells in cell_counts:
        condition = f"patch_shuffle_{cells}x{cells}"
        margin_gain = [
            row["conditions"]["original"]["target_margin"]
            - row["conditions"][condition]["target_margin"]
            for row in records
        ]
        probability_gain = [
            row["conditions"]["original"]["target_probability"]
            - row["conditions"][condition]["target_probability"]
            for row in records
        ]
        original_wins = sum(
            row["conditions"]["original"]["correct"]
            and not row["conditions"][condition]["correct"]
            for row in records
        )
        counterfactual_wins = sum(
            not row["conditions"]["original"]["correct"]
            and row["conditions"][condition]["correct"]
            for row in records
        )
        counterfactual_accuracy = float(np.mean([
            row["conditions"][condition]["correct"] for row in records
        ]))
        changed_rates = [
            row["transform_audit"][condition]["pixel_location_value_changed_rate"]
            for row in records
        ]
        by_granularity[condition] = {
            "counterfactual_accuracy": counterfactual_accuracy,
            "paired_accuracy_gain": original_accuracy - counterfactual_accuracy,
            "original_correct_counterfactual_wrong": int(original_wins),
            "original_wrong_counterfactual_correct": int(counterfactual_wins),
            "mcnemar_exact_two_sided_p": exact_mcnemar_p(original_wins, counterfactual_wins),
            "mean_target_margin_gain": float(np.mean(margin_gain)),
            "target_margin_gain_bootstrap_95ci": bootstrap_ci(margin_gain, f"shuffle-margin:{cells}"),
            "positive_target_margin_gain_rate": float(np.mean(np.asarray(margin_gain) > 0)),
            "mean_target_probability_gain": float(np.mean(probability_gain)),
            "target_probability_gain_bootstrap_95ci": bootstrap_ci(probability_gain, f"shuffle-prob:{cells}"),
            "mean_pixel_location_value_changed_rate": float(np.mean(changed_rates)),
            "min_pixel_location_value_changed_rate": float(np.min(changed_rates)),
        }
    return {
        "case_count": len(records),
        "original_accuracy": original_accuracy,
        "by_granularity": by_granularity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--expected-panel-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cells", type=int, action="append", default=[])
    args = parser.parse_args()
    cell_counts = sorted(set(args.cells or [4, 8, 16]))
    if any(c < 2 or c > 64 for c in cell_counts):
        raise ValueError("shuffle cell count must be in [2, 64]")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run: {args.output_dir}")
    panel_sha = sha256_file(args.panel)
    if panel_sha != args.expected_panel_sha256:
        raise ValueError(f"panel SHA-256 mismatch: {panel_sha}")
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    if (
        panel.get("case_count") != 96
        or panel.get("selection_uses_model_outputs") is not False
        or panel.get("status") != "frozen_before_model_visualization_outputs"
    ):
        raise ValueError("requires the frozen model-blind 96-case panel")
    args.output_dir.mkdir(parents=True)
    model = OptionEvidenceModel(args.model, batch_size=1)
    records: list[dict[str, Any]] = []
    path = args.output_dir / "case_results.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for case in panel["cases"]:
            row = evaluate_case(model, case, cell_counts)
            records.append(row)
            handle.write(canonical_json(row) + "\n")
            handle.flush()
            print(canonical_json({
                "completed": len(records), "total": len(panel["cases"]),
                "panel_index": case["panel_index"],
            }), flush=True)
    metrics = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_confirmation",
        "formal_result": False,
        "method": "exact_pixel_preserving_patch_shuffle_granularity_v1",
        "claim_boundary": "tests spatial-organization dependence while preserving the RGB pixel multiset",
        "model_label": args.model_label,
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "panel_path": str(args.panel.resolve()),
        "panel_sha256": panel_sha,
        "cell_counts": cell_counts,
        "score_batch_size": 1,
        "max_original_repeatability_abs_probability_diff": max(
            row["original_repeatability_max_abs_probability_diff"] for row in records
        ),
        "case_results": str(path.resolve()),
        "case_results_sha256": sha256_file(path),
        **aggregate(records, cell_counts),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(canonical_json(metrics), flush=True)


if __name__ == "__main__":
    main()
