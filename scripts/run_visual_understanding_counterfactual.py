#!/usr/bin/env python3
"""Test whether the correct decision specifically benefits from its paired image."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from run_option_conditioned_visual_evidence import (
    OPTION_LETTERS,
    OptionEvidenceModel,
    canonical_json,
    margins,
    model_prompt,
    sha256_file,
)


CONDITIONS = (
    "original", "nearest_same_target", "nearest_different_target",
    "patch_shuffle_8x8", "strong_blur", "mean_image",
)


def patch_shuffle(image: Image.Image, cells: int, seed: int) -> Image.Image:
    source = np.asarray(image.convert("RGB"))
    ys = np.linspace(0, source.shape[0], cells + 1, dtype=int)
    xs = np.linspace(0, source.shape[1], cells + 1, dtype=int)
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for row in range(cells):
        for column in range(cells):
            shape = (ys[row + 1] - ys[row], xs[column + 1] - xs[column])
            groups.setdefault(shape, []).append((row, column))
    rng = np.random.default_rng(seed)
    output = np.empty_like(source)
    for coordinates in groups.values():
        sources = list(coordinates)
        rng.shuffle(sources)
        if len(sources) > 1 and sources == coordinates:
            sources = sources[1:] + sources[:1]
        for (dst_r, dst_c), (src_r, src_c) in zip(coordinates, sources):
            output[ys[dst_r]:ys[dst_r + 1], xs[dst_c]:xs[dst_c + 1]] = source[ys[src_r]:ys[src_r + 1], xs[src_c]:xs[src_c + 1]]
    return Image.fromarray(output, mode="RGB")


def condition_images(case: dict[str, Any], pairing: dict[str, Any]) -> dict[str, Image.Image]:
    path = Path(case["image"])
    if sha256_file(path) != case["image_sha256"]:
        raise ValueError(f"source image hash mismatch: {case['panel_index']}")
    original = Image.open(path).convert("RGB")
    values = np.asarray(original, dtype=np.float32)
    mean_rgb = tuple(int(round(float(x))) for x in values.mean(axis=(0, 1)))
    output = {
        "original": original,
        "patch_shuffle_8x8": patch_shuffle(original, 8, pairing["patch_shuffle_seed"]),
        "strong_blur": original.filter(ImageFilter.GaussianBlur(radius=max(4.0, min(original.size) * 0.08))),
        "mean_image": Image.new("RGB", original.size, mean_rgb),
    }
    for name in ("nearest_same_target", "nearest_different_target"):
        mismatch = pairing[name]
        mismatch_path = Path(mismatch["image"])
        if sha256_file(mismatch_path) != mismatch["image_sha256"]:
            raise ValueError(f"mismatch image hash mismatch: {case['panel_index']}:{name}")
        output[name] = Image.open(mismatch_path).convert("RGB")
    return output


def score_record(model: OptionEvidenceModel, case: dict[str, Any], pairing: dict[str, Any]) -> dict[str, Any]:
    prompt = model_prompt(model.processor, case["problem"])
    images = condition_images(case, pairing)
    first, repeated = model.score(prompt, [images["original"], images["original"]])
    repeatability = float(np.max(np.abs(np.asarray(first) - np.asarray(repeated))))
    scores = {"original": first}
    for condition in CONDITIONS[1:]:
        scores[condition] = model.score(prompt, [images[condition]])[0]
    target = OPTION_LETTERS.index(case["target_choice"])
    results = {}
    for condition, probabilities in scores.items():
        option_margins = margins(probabilities)
        prediction = int(np.argmax(probabilities))
        results[condition] = {
            "option_probabilities": dict(zip(OPTION_LETTERS, probabilities)),
            "option_margins": dict(zip(OPTION_LETTERS, option_margins)),
            "predicted_choice": OPTION_LETTERS[prediction],
            "correct": prediction == target,
            "target_probability": float(probabilities[target]),
            "target_margin": float(option_margins[target]),
        }
    return {
        "panel_index": case["panel_index"], "source_index": case["index"],
        "source_record_sha256": case["source_record_sha256"], "image": case["image"],
        "image_sha256": case["image_sha256"], "target_choice": case["target_choice"],
        "original_repeatability_max_abs_probability_diff": repeatability,
        "pairing": pairing, "conditions": results,
    }


def bootstrap_ci(values: list[float], seed_text: str, repetitions: int = 10000) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return [float("nan"), float("nan")]
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(repetitions, len(array)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def exact_mcnemar_p(original_wins: int, counterfactual_wins: int) -> float:
    total = original_wins + counterfactual_wins
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(min(original_wins, counterfactual_wins) + 1)) / (2 ** total)
    return min(1.0, 2.0 * tail)


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "case_count": len(records),
        "original_accuracy": float(np.mean([r["conditions"]["original"]["correct"] for r in records])),
        "by_counterfactual": {},
    }
    for condition in CONDITIONS[1:]:
        margin_gain = [r["conditions"]["original"]["target_margin"] - r["conditions"][condition]["target_margin"] for r in records]
        probability_gain = [r["conditions"]["original"]["target_probability"] - r["conditions"][condition]["target_probability"] for r in records]
        original_wins = sum(r["conditions"]["original"]["correct"] and not r["conditions"][condition]["correct"] for r in records)
        counterfactual_wins = sum((not r["conditions"]["original"]["correct"]) and r["conditions"][condition]["correct"] for r in records)
        counterfactual_accuracy = float(np.mean([r["conditions"][condition]["correct"] for r in records]))
        output["by_counterfactual"][condition] = {
            "counterfactual_accuracy": counterfactual_accuracy,
            "paired_accuracy_gain": output["original_accuracy"] - counterfactual_accuracy,
            "original_correct_counterfactual_wrong": int(original_wins),
            "original_wrong_counterfactual_correct": int(counterfactual_wins),
            "mcnemar_exact_two_sided_p": exact_mcnemar_p(original_wins, counterfactual_wins),
            "mean_target_margin_gain": float(np.mean(margin_gain)),
            "target_margin_gain_bootstrap_95ci": bootstrap_ci(margin_gain, f"margin:{condition}"),
            "positive_target_margin_gain_rate": float(np.mean(np.asarray(margin_gain) > 0)),
            "mean_target_probability_gain": float(np.mean(probability_gain)),
            "target_probability_gain_bootstrap_95ci": bootstrap_ci(probability_gain, f"probability:{condition}"),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--expected-panel-sha256", required=True)
    parser.add_argument("--counterfactual-manifest", type=Path, required=True)
    parser.add_argument("--expected-counterfactual-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-index", type=int, action="append")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run: {args.output_dir}")
    panel_sha = sha256_file(args.panel)
    manifest_sha = sha256_file(args.counterfactual_manifest)
    if panel_sha != args.expected_panel_sha256 or manifest_sha != args.expected_counterfactual_sha256:
        raise ValueError("frozen input SHA-256 mismatch")
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    manifest = json.loads(args.counterfactual_manifest.read_text(encoding="utf-8"))
    if panel.get("case_count") != 96 or manifest.get("case_count") != 96 or manifest.get("panel_sha256") != panel_sha:
        raise ValueError("96-case frozen counterfactual contract mismatch")
    selected = set(args.panel_index or range(96))
    cases = [row for row in panel["cases"] if row["panel_index"] in selected]
    pairings = {row["panel_index"]: row for row in manifest["pairings"]}
    if len(cases) != len(selected):
        raise ValueError("requested panel index missing")
    args.output_dir.mkdir(parents=True)
    model = OptionEvidenceModel(args.model, batch_size=1)
    records = []
    path = args.output_dir / "case_results.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            row = score_record(model, case, pairings[case["panel_index"]])
            records.append(row); handle.write(canonical_json(row) + "\n"); handle.flush()
            print(canonical_json({"completed": len(records), "total": len(cases), "panel_index": case["panel_index"]}), flush=True)
    metrics = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_confirmation" if len(cases) == 96 else "completed_smoke", "formal_result": False,
        "method": "paired_image_specific_visual_understanding_counterfactual_v1",
        "primary_endpoint": "ground_truth_option_margin_original_minus_nearest_same_domain_mismatch",
        "model_label": args.model_label, "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "panel_path": str(args.panel.resolve()), "panel_sha256": panel_sha,
        "counterfactual_manifest_path": str(args.counterfactual_manifest.resolve()), "counterfactual_manifest_sha256": manifest_sha,
        "panel_indices": [r["panel_index"] for r in records], "conditions": list(CONDITIONS),
        "score_batch_size": 1,
        "max_original_repeatability_abs_probability_diff": max(r["original_repeatability_max_abs_probability_diff"] for r in records),
        "case_results": str(path.resolve()), "case_results_sha256": sha256_file(path),
        **aggregate(records),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(canonical_json(metrics), flush=True)


if __name__ == "__main__":
    main()
