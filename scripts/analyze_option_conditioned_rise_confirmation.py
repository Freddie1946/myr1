#!/usr/bin/env python3
"""Strict confirmatory statistics for option-conditioned RISE runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_option_conditioned_confirmation import holm_adjust, sign_flip_p
from run_option_conditioned_visual_evidence import canonical_json, sha256_file
from run_visual_understanding_counterfactual import bootstrap_ci


def case_advantages(row: dict[str, Any], mode: str) -> dict[str, float]:
    curves = row["curves"][mode]
    random_auc = float(np.mean([
        control["target_margin_auc"]
        for control in curves["random_shifted_shape_matched"]
    ]))
    high = float(curves["rise_high"]["target_margin_auc"])
    low = float(curves["rise_low"]["target_margin_auc"])
    if mode == "deletion":
        return {
            "high_vs_random": random_auc - high,
            "high_vs_low": low - high,
            "low_vs_random": random_auc - low,
        }
    if mode == "retention":
        return {
            "high_vs_random": high - random_auc,
            "high_vs_low": high - low,
            "low_vs_random": low - random_auc,
        }
    raise ValueError(f"unsupported mode: {mode}")


def summarize(values: list[float], seed: str, repetitions: int) -> dict[str, Any]:
    return {
        "mean": float(np.mean(values)),
        "bootstrap_95ci": bootstrap_ci(values, seed),
        "positive_rate": float(np.mean(np.asarray(values) > 0)),
        "sign_flip_one_sided_p": sign_flip_p(values, seed + ":sign-flip", repetitions),
    }


def analyze(path: Path, repetitions: int) -> dict[str, Any]:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if metrics.get("status") != "completed_confirmation" or metrics.get("case_count") != 96:
        raise ValueError(f"not a completed 96-case RISE run: {path}")
    case_path = Path(metrics["case_results"])
    if sha256_file(case_path) != metrics["case_results_sha256"]:
        raise ValueError(f"case-results hash mismatch: {case_path}")
    rows = [json.loads(line) for line in case_path.open(encoding="utf-8") if line.strip()]
    if len(rows) != 96 or len({row["panel_index"] for row in rows}) != 96:
        raise ValueError(f"case identity audit failed: {case_path}")
    by_mode: dict[str, Any] = {}
    per_case: dict[str, dict[int, dict[str, float]]] = {}
    for mode in ("deletion", "retention"):
        values = {row["panel_index"]: case_advantages(row, mode) for row in rows}
        per_case[mode] = values
        by_mode[mode] = {
            endpoint: summarize(
                [row[endpoint] for row in values.values()],
                f"option-rise-confirmation:{metrics['model_label']}:{mode}:{endpoint}",
                repetitions,
            )
            for endpoint in ("high_vs_random", "high_vs_low", "low_vs_random")
        }
    deletion = per_case["deletion"]
    retention = per_case["retention"]
    minima = [
        min(deletion[index]["high_vs_random"], retention[index]["high_vs_random"])
        for index in sorted(deletion)
    ]
    both_positive = [
        deletion[index]["high_vs_random"] > 0 and retention[index]["high_vs_random"] > 0
        for index in sorted(deletion)
    ]
    return {
        "model_label": metrics["model_label"],
        "metrics_path": str(path.resolve()),
        "metrics_sha256": sha256_file(path),
        "case_results_sha256": metrics["case_results_sha256"],
        "baseline_accuracy": metrics["baseline_accuracy"],
        "by_mode": by_mode,
        "bidirectional": {
            "both_positive_rate": float(np.mean(both_positive)),
            "minimum_direction": summarize(
                minima, f"option-rise-confirmation:{metrics['model_label']}:minimum", repetitions
            ),
        },
        "repeatability": {
            "baseline": metrics["max_baseline_repeatability_abs_probability_diff"],
            "masked_sentinel": metrics["max_masked_sentinel_repeatability_abs_probability_diff"],
            "intervention_sentinel": metrics["max_intervention_sentinel_repeatability_abs_probability_diff"],
        },
    }


def apply_global_holm(models: list[dict[str, Any]], endpoint: str) -> None:
    family = {
        f"{model['model_label']}:{mode}": model["by_mode"][mode][endpoint]["sign_flip_one_sided_p"]
        for model in models for mode in ("deletion", "retention")
    }
    adjusted = holm_adjust(family)
    for model in models:
        for mode in ("deletion", "retention"):
            key = f"{model['model_label']}:{mode}"
            model["by_mode"][mode][endpoint]["holm_adjusted_across_two_models_two_directions_p"] = adjusted[key]


def markdown(models: list[dict[str, Any]]) -> str:
    lines = [
        "# Option-conditioned RISE 96-case confirmation", "",
        "Primary tests compare the ground-truth-option RISE-high region with five area/shape-matched random shifts under hard mean-fill validation. One-sided sign-flip p-values are Holm-adjusted over two models x deletion/retention (four tests). RISE generation used blurred replacement; validation therefore uses a different perturbation.", "",
        "| Model | Mode | High vs random [95% CI] | Holm p | High vs low [95% CI] | Holm p | Positive cases |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        for mode in ("deletion", "retention"):
            hr = model["by_mode"][mode]["high_vs_random"]
            hl = model["by_mode"][mode]["high_vs_low"]
            hci, lci = hr["bootstrap_95ci"], hl["bootstrap_95ci"]
            lines.append(
                f"| {model['model_label']} | {mode} | {hr['mean']:+.4f} [{hci[0]:+.4f}, {hci[1]:+.4f}] | "
                f"{hr['holm_adjusted_across_two_models_two_directions_p']:.4g} | "
                f"{hl['mean']:+.4f} [{lci[0]:+.4f}, {lci[1]:+.4f}] | "
                f"{hl['holm_adjusted_across_two_models_two_directions_p']:.4g} | {hr['positive_rate']:.3f} |"
            )
    lines.extend(["", "## Bidirectional diagnostic", ""])
    for model in models:
        minimum = model["bidirectional"]["minimum_direction"]
        ci = minimum["bootstrap_95ci"]
        lines.append(
            f"- {model['model_label']}: both directions positive in "
            f"{model['bidirectional']['both_positive_rate']:.1%} of cases; mean per-case minimum "
            f"{minimum['mean']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]."
        )
    lines.extend(["", "## Reproducibility gates", ""])
    for model in models:
        gate = model["repeatability"]
        lines.append(
            f"- {model['model_label']}: baseline/masked/intervention maximum A/B/C/D probability differences "
            f"= {gate['baseline']}/{gate['masked_sentinel']}/{gate['intervention_sentinel']}."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--sign-flip-repetitions", type=int, default=100000)
    args = parser.parse_args()
    if args.output_json.exists() or args.output_markdown.exists():
        raise FileExistsError("refusing to overwrite RISE confirmation analysis")
    models = [analyze(path, args.sign_flip_repetitions) for path in args.metrics]
    apply_global_holm(models, "high_vs_random")
    apply_global_holm(models, "high_vs_low")
    result = {
        "schema_version": 1,
        "status": "completed_analysis",
        "method": "option_conditioned_rise_bidirectional_confirmation_v1",
        "sign_flip_repetitions": args.sign_flip_repetitions,
        "models": models,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_markdown.write_text(markdown(models), encoding="utf-8")
    print(canonical_json({
        "status": result["status"], "model_count": len(models),
        "output_json": str(args.output_json.resolve()),
        "output_markdown": str(args.output_markdown.resolve()),
    }))


if __name__ == "__main__":
    main()
