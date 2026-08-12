#!/usr/bin/env python3
"""Audit option-conditioned spatial support across layers and controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from run_option_conditioned_visual_evidence import OPTION_LETTERS, canonical_json, sha256_file
from run_visual_understanding_counterfactual import bootstrap_ci


def selectivity(values: np.ndarray, target: int) -> float:
    return float(values[target] - np.delete(values, target).mean())


def option_vector(record: dict[str, Any]) -> np.ndarray:
    return np.asarray([record[letter] for letter in OPTION_LETTERS], dtype=np.float64)


def sign_flip_p(values: list[float], seed_text: str, repetitions: int) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return None
    observed = float(array.mean())
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    while completed < repetitions:
        size = min(5000, repetitions - completed)
        signs = rng.integers(0, 2, size=(size, len(array)), dtype=np.int8) * 2 - 1
        means = signs.dot(array) / len(array)
        extreme += int(np.sum(means >= observed))
        completed += size
    return float((extreme + 1) / (repetitions + 1))


def holm_adjust(values: dict[str, float | None]) -> dict[str, float | None]:
    present = [(name, value) for name, value in values.items() if value is not None]
    present.sort(key=lambda item: item[1])
    output: dict[str, float | None] = {name: None for name in values}
    previous = 0.0
    count = len(present)
    for rank, (name, value) in enumerate(present):
        previous = max(previous, min(1.0, float(value) * (count - rank)))
        output[name] = previous
    return output


def layer_case_values(row: dict[str, Any], layer: int) -> dict[str, float] | None:
    target_letter = row["target_choice"]
    target = OPTION_LETTERS.index(target_letter)
    methods = row["interventions"][str(layer)][target_letter]["methods"]
    random_names = sorted(name for name in methods if name.startswith("random_"))
    if "positive" not in methods or not random_names:
        return None
    baseline = option_vector(row["baseline_option_margins"])
    positive_deletion = option_vector(methods["positive"]["deletion"]["option_margins"])
    positive_retention = option_vector(methods["positive"]["retention"]["option_margins"])
    random_deletions = [
        option_vector(methods[name]["deletion"]["option_margins"])
        for name in random_names
    ]
    random_retentions = [
        option_vector(methods[name]["retention"]["option_margins"])
        for name in random_names
    ]
    positive_effect = baseline - positive_deletion
    random_effects = [baseline - values for values in random_deletions]
    random_deletion_selectivity = float(np.mean([
        selectivity(values, target) for values in random_effects
    ]))
    random_retention_selectivity = float(np.mean([
        selectivity(values, target) for values in random_retentions
    ]))
    values = {
        "positive_deletion_target_drop_minus_random": float(
            positive_effect[target] - np.mean([effect[target] for effect in random_effects])
        ),
        "positive_deletion_selectivity_minus_random": float(
            selectivity(positive_effect, target) - random_deletion_selectivity
        ),
        "positive_retention_target_margin_minus_random": float(
            positive_retention[target] - np.mean([retention[target] for retention in random_retentions])
        ),
        "positive_retention_selectivity_minus_random": float(
            selectivity(positive_retention, target) - random_retention_selectivity
        ),
    }
    if "low_positive" in methods:
        low_deletion = option_vector(methods["low_positive"]["deletion"]["option_margins"])
        low_retention = option_vector(methods["low_positive"]["retention"]["option_margins"])
        low_effect = baseline - low_deletion
        values["positive_deletion_selectivity_minus_low"] = float(
            selectivity(positive_effect, target) - selectivity(low_effect, target)
        )
        values["positive_retention_selectivity_minus_low"] = float(
            selectivity(positive_retention, target) - selectivity(low_retention, target)
        )
    if "raw_attention" in methods:
        raw_deletion = option_vector(methods["raw_attention"]["deletion"]["option_margins"])
        raw_retention = option_vector(methods["raw_attention"]["retention"]["option_margins"])
        raw_effect = baseline - raw_deletion
        values["raw_deletion_selectivity_minus_random"] = float(
            selectivity(raw_effect, target) - random_deletion_selectivity
        )
        values["raw_retention_selectivity_minus_random"] = float(
            selectivity(raw_retention, target) - random_retention_selectivity
        )
    return values


def summarize_values(
    values: list[dict[str, float]], model_label: str, layer: int, repetitions: int,
) -> dict[str, Any]:
    names = sorted({name for row in values for name in row})
    result: dict[str, Any] = {"available_case_count": len(values)}
    for name in names:
        sample = [row[name] for row in values if name in row]
        result[name] = {
            "mean": float(np.mean(sample)),
            "bootstrap_95ci": bootstrap_ci(sample, f"option-confirmation:{model_label}:{layer}:{name}"),
            "positive_rate": float(np.mean(np.asarray(sample) > 0)),
            "sign_flip_one_sided_p": sign_flip_p(
                sample, f"option-confirmation-p:{model_label}:{layer}:{name}", repetitions,
            ),
        }
    deletion = [row["positive_deletion_selectivity_minus_random"] for row in values]
    retention = [row["positive_retention_selectivity_minus_random"] for row in values]
    result["bidirectional_positive_rate"] = float(np.mean(
        (np.asarray(deletion) > 0) & (np.asarray(retention) > 0)
    ))
    result["bidirectional_minimum_mean"] = float(np.mean(np.minimum(deletion, retention)))
    result["bidirectional_minimum_bootstrap_95ci"] = bootstrap_ci(
        np.minimum(deletion, retention).tolist(),
        f"option-confirmation-min:{model_label}:{layer}",
    )
    return result


def analyze_metrics(path: Path, repetitions: int) -> dict[str, Any]:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if metrics.get("case_count") != 96 or metrics.get("status") != "completed_confirmation":
        raise ValueError(f"not a completed 96-case confirmation: {path}")
    case_path = Path(metrics["case_results"])
    if sha256_file(case_path) != metrics["case_results_sha256"]:
        raise ValueError(f"case-results SHA-256 mismatch: {case_path}")
    rows = [json.loads(line) for line in case_path.open(encoding="utf-8") if line.strip()]
    if len(rows) != 96 or len({row["panel_index"] for row in rows}) != 96:
        raise ValueError(f"case-results identity audit failed: {case_path}")
    layers = metrics["layers_evaluated"]
    result = {
        "model_label": metrics["model_label"],
        "metrics_path": str(path.resolve()),
        "metrics_sha256": sha256_file(path),
        "case_results_sha256": metrics["case_results_sha256"],
        "baseline_accuracy": metrics["baseline_accuracy"],
        "repeatability": {
            "baseline": metrics["max_baseline_repeatability_abs_probability_diff"],
            "sentinel_intervention": metrics["max_sentinel_intervention_repeatability_abs_probability_diff"],
            "attribution_vs_causal_baseline": metrics["max_attribution_vs_causal_baseline_abs_probability_diff"],
        },
        "by_layer": {},
    }
    for layer in layers:
        values = [value for row in rows if (value := layer_case_values(row, layer)) is not None]
        result["by_layer"][str(layer)] = summarize_values(
            values, metrics["model_label"], layer, repetitions,
        )
    for endpoint in (
        "positive_deletion_selectivity_minus_random",
        "positive_retention_selectivity_minus_random",
    ):
        raw = {
            str(layer): result["by_layer"][str(layer)][endpoint]["sign_flip_one_sided_p"]
            for layer in layers
        }
        adjusted = holm_adjust(raw)
        for layer in layers:
            result["by_layer"][str(layer)][endpoint]["holm_adjusted_across_layers_p"] = adjusted[str(layer)]
    return result


def markdown_report(analyses: list[dict[str, Any]]) -> str:
    lines = [
        "# Option-conditioned 96-case spatial-support audit", "",
        "A spatial map is treated as support only when its target-option deletion selectivity and retention selectivity both exceed area-matched random controls. P-values are one-sided sign-flip tests with Holm correction across the eight inspected layers within each model.", "",
        "| Model | Layer | Cases | Deletion selectivity minus random [95% CI], Holm p | Retention selectivity minus random [95% CI], Holm p | Both positive | Minimum-direction mean [95% CI] | Raw deletion / retention selectivity minus random |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for analysis in analyses:
        for layer, row in analysis["by_layer"].items():
            deletion = row["positive_deletion_selectivity_minus_random"]
            retention = row["positive_retention_selectivity_minus_random"]
            raw_d = row.get("raw_deletion_selectivity_minus_random", {"mean": float("nan")})
            raw_r = row.get("raw_retention_selectivity_minus_random", {"mean": float("nan")})
            dci, rci = deletion["bootstrap_95ci"], retention["bootstrap_95ci"]
            mci = row["bidirectional_minimum_bootstrap_95ci"]
            lines.append(
                f"| {analysis['model_label']} | {layer} | {row['available_case_count']} | "
                f"{deletion['mean']:+.4f} [{dci[0]:+.4f}, {dci[1]:+.4f}], {deletion['holm_adjusted_across_layers_p']:.4g} | "
                f"{retention['mean']:+.4f} [{rci[0]:+.4f}, {rci[1]:+.4f}], {retention['holm_adjusted_across_layers_p']:.4g} | "
                f"{row['bidirectional_positive_rate']:.3f} | {row['bidirectional_minimum_mean']:+.4f} [{mci[0]:+.4f}, {mci[1]:+.4f}] | "
                f"{raw_d['mean']:+.4f} / {raw_r['mean']:+.4f} |"
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
        raise FileExistsError("refusing to overwrite option-confirmation analysis")
    analyses = [analyze_metrics(path, args.sign_flip_repetitions) for path in args.metrics]
    result = {
        "schema_version": 1,
        "status": "completed_analysis",
        "method": "bidirectional_option_conditioned_spatial_support_vs_random_v1",
        "sign_flip_repetitions": args.sign_flip_repetitions,
        "models": analyses,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_markdown.write_text(markdown_report(analyses), encoding="utf-8")
    print(canonical_json({
        "status": result["status"], "model_count": len(analyses),
        "output_json": str(args.output_json.resolve()),
        "output_markdown": str(args.output_markdown.resolve()),
    }))


if __name__ == "__main__":
    main()
