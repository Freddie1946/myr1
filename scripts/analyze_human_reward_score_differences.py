#!/usr/bin/env python3
"""Compare expert-derived process scores with GPT-4o and Claude scores."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


EVENTS = (
    "image_feature_analysis_present",
    "option_elimination_present",
    "medical_knowledge_support_present",
    "histological_definition_error",
    "logical_contradiction",
    "outdated_or_incorrect_pathology_criterion",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score(events: dict[str, bool]) -> float:
    if set(events) != set(EVENTS) or any(type(events[field]) is not bool for field in EVENTS):
        raise ValueError("invalid six-event record")
    missing = sum(not events[field] for field in EVENTS[:3])
    errors = sum(events[field] for field in EVENTS[3:])
    return (max(0.0, 1.0 - 0.4 * missing) + max(0.0, 1.0 - 0.4 * errors)) / 2.0


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize(pairs: list[dict[str, Any]], target: str, iterations: int = 10000) -> dict[str, Any]:
    differences = [row["human_score"] - row[f"{target}_score"] for row in pairs]
    absolute = [abs(value) for value in differences]
    rng = random.Random(20260815)
    boot_mean, boot_mae = [], []
    for _ in range(iterations):
        sample = [differences[rng.randrange(len(differences))] for _ in differences]
        boot_mean.append(statistics.fmean(sample))
        boot_mae.append(statistics.fmean(abs(value) for value in sample))
    return {
        "case_count": len(pairs),
        "human_score_mean": statistics.fmean(row["human_score"] for row in pairs),
        f"{target}_score_mean": statistics.fmean(row[f"{target}_score"] for row in pairs),
        "mean_signed_human_minus_target": statistics.fmean(differences),
        "mean_signed_difference_bootstrap_95ci": [percentile(boot_mean, 0.025), percentile(boot_mean, 0.975)],
        "mean_absolute_difference": statistics.fmean(absolute),
        "mean_absolute_difference_bootstrap_95ci": [percentile(boot_mae, 0.025), percentile(boot_mae, 0.975)],
        "median_absolute_difference": statistics.median(absolute),
        "root_mean_squared_difference": math.sqrt(statistics.fmean(value * value for value in differences)),
        "exact_score_agreement": sum(value < 1e-12 for value in absolute) / len(absolute),
        "within_0p2": sum(value <= 0.2 + 1e-12 for value in absolute) / len(absolute),
        "event_agreement_secondary": sum(
            row["human_events"][field] == row[f"{target}_events"][field]
            for row in pairs for field in EVENTS
        ) / (len(pairs) * len(EVENTS)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings-dir", type=Path, required=True)
    parser.add_argument("--gpt-key", type=Path, required=True)
    parser.add_argument("--claude-references", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    gpt = {row["case_id"]: row for row in load_jsonl(args.gpt_key)}
    claude = {row["case_id"]: row["reference"] for row in load_jsonl(args.claude_references)}
    reviewer_results = {}
    all_pair_rows = []
    for reviewer_dir in sorted(path for path in args.ratings_dir.iterdir() if path.is_dir()):
        pairs = []
        for path in sorted(reviewer_dir.glob("HRA-*.json")):
            human = json.loads(path.read_text(encoding="utf-8"))
            case_id = human["case_id"]
            human_events = human["events"]
            gpt_events = gpt[case_id]["events"]
            claude_events = {field: claude[case_id][field] for field in EVENTS}
            row = {
                "reviewer_id": reviewer_dir.name,
                "case_id": case_id,
                "human_events": human_events,
                "gpt_events": gpt_events,
                "claude_events": claude_events,
                "human_score": score(human_events),
                "gpt_score": score(gpt_events),
                "claude_score": score(claude_events),
            }
            if abs(float(gpt[case_id]["process_reward"]) - row["gpt_score"]) > 1e-9:
                raise ValueError(f"stored GPT reward mismatch: {case_id}")
            pairs.append(row)
            all_pair_rows.append(row)
        if pairs:
            reviewer_results[reviewer_dir.name] = {
                "gpt4o_training_reward": summarize(pairs, "gpt"),
                "claude_external_reference": summarize(pairs, "claude"),
            }
    if not reviewer_results:
        raise ValueError("no saved expert ratings found")
    payload = {
        "schema_version": 1,
        "primary_endpoint": "per-case process-score difference under the frozen 0.4 formula",
        "scientific_boundary": "reference-assisted expert ratings; not independent blinded ratings",
        "reviewers": reviewer_results,
        "case_rows": all_pair_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# 人工评分与奖励模型的最终过程分差",
        "",
        "> 主终点是按相同 0.4 公式换算后的逐病例过程分差；事件一致率仅作辅助。当前网页显示外部参考，因此属于 reference-assisted，不是独立盲评。",
        "",
        "| Reviewer | Target | N | Human mean | Target mean | Mean signed diff | 95% CI | MAE | Exact | Within 0.2 | Event agreement (secondary) |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for reviewer, targets in reviewer_results.items():
        for label, result in targets.items():
            target_mean_key = "gpt_score_mean" if label.startswith("gpt4o") else "claude_score_mean"
            ci = result["mean_signed_difference_bootstrap_95ci"]
            lines.append(
                f"| {reviewer} | {label} | {result['case_count']} | {result['human_score_mean']:.3f} | "
                f"{result[target_mean_key]:.3f} | {result['mean_signed_human_minus_target']:+.3f} | "
                f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | {result['mean_absolute_difference']:.3f} | "
                f"{result['exact_score_agreement']:.1%} | {result['within_0p2']:.1%} | {result['event_agreement_secondary']:.1%} |"
            )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "reviewers": list(reviewer_results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
