#!/usr/bin/env python3
"""Summarize paired behavior and dual-stream causal evidence experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import binomtest, wilcoxon


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def bootstrap_ci(values: list[float], seed: int = 20260813, draws: int = 10000) -> list[float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def paired_summary(values: list[float]) -> dict[str, Any]:
    nonzero = [value for value in values if abs(value) > 1e-12]
    positive = sum(value > 0 for value in nonzero)
    sign_p = float(binomtest(positive, len(nonzero), 0.5).pvalue) if nonzero else None
    try:
        signed_rank_p = float(wilcoxon(values, zero_method="wilcox").pvalue) if nonzero else None
    except ValueError:
        signed_rank_p = None
    return {
        "count": len(values),
        "mean": float(np.mean(values)) if values else None,
        "median": float(np.median(values)) if values else None,
        "mean_bootstrap_95ci": bootstrap_ci(values),
        "positive_count": sum(value > 0 for value in values),
        "negative_count": sum(value < 0 for value in values),
        "zero_count": sum(abs(value) <= 1e-12 for value in values),
        "two_sided_sign_test_p": sign_p,
        "two_sided_wilcoxon_p": signed_rank_p,
    }


def behavior_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    primary = [
        row for row in records
        if row["clean_reasoning_correct"] and row["clean_reasoning_forced_agreement"]
    ]
    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        effects = [float(row["reference_minus_random_extra_drop"]) for row in group]
        reference_flips = 0
        random_flips = 0
        random_trials = 0
        for row in group:
            target = row["target_choice"]
            reference_flips += row["conditions"]["reference_deleted"]["reasoning_choice"] != target
            for name, condition in row["conditions"].items():
                if name.startswith("random_deleted_"):
                    random_trials += 1
                    random_flips += condition["reasoning_choice"] != target
        return {
            "case_count": len(group),
            "reference_minus_random_extra_target_margin_drop": paired_summary(effects),
            "reference_deletion_reasoning_flip_count": reference_flips,
            "reference_deletion_reasoning_flip_rate": reference_flips / len(group) if group else None,
            "random_deletion_reasoning_flip_count": random_flips,
            "random_deletion_reasoning_flip_trials": random_trials,
            "random_deletion_reasoning_flip_rate": random_flips / random_trials if random_trials else None,
        }
    return {"all_cases": summarize(records), "strict_primary_cases": summarize(primary)}


def dual_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"case_count": 0, "by_layer": {}}
    layers = sorted(records[0]["patched_by_layer"], key=int)
    output: dict[str, Any] = {}
    for layer in layers:
        streams = {}
        for stream in ("visual_token_patch", "query_position_patch"):
            raw = [float(row["patched_by_layer"][layer][stream]["raw_margin_recovery"]) for row in records]
            fractions = [
                float(row["patched_by_layer"][layer][stream]["recovery_fraction"])
                for row in records
                if row["patched_by_layer"][layer][stream]["recovery_fraction"] is not None
            ]
            streams[stream] = {
                "raw_margin_recovery": paired_summary(raw),
                "positive_gap_recovery_fraction": paired_summary(fractions),
            }
        output[layer] = streams
    return {"case_count": len(records), "by_layer": output}


def markdown(payload: dict[str, Any]) -> str:
    behavior = payload["behavior"]
    lines = [
        "# Reference-evidence causal experiment summary", "",
        "## Behavioral deletion", "",
        "| Population | n | Ref−random margin effect | 95% bootstrap CI | Wilcoxon p | Ref flip | Random flip |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("all_cases", "All frozen cases"), ("strict_primary_cases", "Clean-correct + interface-agree")):
        row = behavior[key]
        effect = row["reference_minus_random_extra_target_margin_drop"]
        ci = effect["mean_bootstrap_95ci"]
        ci_text = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "NA"
        p = effect["two_sided_wilcoxon_p"]
        lines.append(
            f"| {label} | {row['case_count']} | {effect['mean']:.4f} | {ci_text} | "
            f"{p:.4g} | {row['reference_deletion_reasoning_flip_rate']:.3f} | "
            f"{row['random_deletion_reasoning_flip_rate']:.3f} |"
        )
    lines += ["", "## Dual-stream activation restoration", "",
              "| Layer | Visual recovery fraction | Query recovery fraction | Visual raw recovery | Query raw recovery |",
              "|---:|---:|---:|---:|---:|"]
    for layer, streams in payload["dual_stream"]["by_layer"].items():
        visual = streams["visual_token_patch"]
        query = streams["query_position_patch"]
        lines.append(
            f"| {layer} | {visual['positive_gap_recovery_fraction']['median']:.4f} | "
            f"{query['positive_gap_recovery_fraction']['median']:.4f} | "
            f"{visual['raw_margin_recovery']['median']:.4f} | "
            f"{query['raw_margin_recovery']['median']:.4f} |"
        )
    lines += [
        "", "Interpretation boundary: visual-token recovery at late layers is structurally limited by the "
        "lack of subsequent attention layers. The visual-token and query-position curves must be interpreted "
        "jointly. External boxes remain reference annotations pending blinded pathology-expert review.", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior-dir", type=Path, required=True)
    parser.add_argument("--dual-stream-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    behavior_records = rows(args.behavior_dir / "case_results.jsonl")
    dual_records = rows(args.dual_stream_dir / "case_results.jsonl")
    payload = {
        "schema_version": 1,
        "behavior_dir": str(args.behavior_dir.resolve()),
        "dual_stream_dir": str(args.dual_stream_dir.resolve()),
        "behavior": behavior_summary(behavior_records),
        "dual_stream": dual_summary(dual_records),
    }
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    args.output_root.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_root.with_suffix(".md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({
        "json": str(args.output_root.with_suffix('.json').resolve()),
        "markdown": str(args.output_root.with_suffix('.md').resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
