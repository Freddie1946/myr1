#!/usr/bin/env python3
"""Compare training-time GPT-4o reward events with Claude external references."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


EVENTS = (
    "image_feature_analysis_present",
    "option_elimination_present",
    "medical_knowledge_support_present",
    "histological_definition_error",
    "logical_contradiction",
    "outdated_or_incorrect_pathology_criterion",
)
POSITIVE_EVENTS = EVENTS[:3]
ERROR_EVENTS = EVENTS[3:]
LABELS = {
    "image_feature_analysis_present": "图像特征分析 / Image-feature analysis",
    "option_elimination_present": "选项排除 / Option elimination",
    "medical_knowledge_support_present": "医学知识支持 / Medical-knowledge support",
    "histological_definition_error": "组织学定义错误 / Histological-definition error",
    "logical_contradiction": "逻辑矛盾 / Logical contradiction",
    "outdated_or_incorrect_pathology_criterion": "错误或过时病理标准 / Incorrect/outdated criterion",
}


def kappa(left: list[bool], right: list[bool]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    observed = mean(a == b for a, b in zip(left, right))
    pa = mean(left)
    pb = mean(right)
    expected = pa * pb + (1 - pa) * (1 - pb)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2
        for offset in range(position, end):
            result[order[offset]] = average_rank
        position = end
    return result


def correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    lm, rm = mean(left), mean(right)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - lm) ** 2 for a in left) * sum((b - rm) ** 2 for b in right))
    return None if denominator == 0 else numerator / denominator


def spearman(left: list[float], right: list[float]) -> float | None:
    return correlation(ranks(left), ranks(right))


def process_reward(events: dict[str, bool]) -> float:
    missing = sum(not events[name] for name in POSITIVE_EVENTS)
    errors = sum(events[name] for name in ERROR_EVENTS)
    return (max(0.0, 1 - 0.4 * missing) + max(0.0, 1 - 0.4 * errors)) / 2


def wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def bootstrap(cases: list[dict[str, Any]], iterations: int, seed: int) -> dict[str, list[float]]:
    rng = random.Random(seed)
    values = {"micro_event_agreement": [], "reward_mae": [], "reward_spearman": []}
    for _ in range(iterations):
        sample = [cases[rng.randrange(len(cases))] for _ in cases]
        event_pairs = [(row["gpt"][event], row["claude"][event]) for row in sample for event in EVENTS]
        values["micro_event_agreement"].append(mean(a == b for a, b in event_pairs))
        gpt_rewards = [process_reward(row["gpt"]) for row in sample]
        claude_rewards = [process_reward(row["claude"]) for row in sample]
        values["reward_mae"].append(mean(abs(a - b) for a, b in zip(gpt_rewards, claude_rewards)))
        rho = spearman(gpt_rewards, claude_rewards)
        if rho is not None:
            values["reward_spearman"].append(rho)
    return {key: [percentile(series, 0.025), percentile(series, 0.975)] for key, series in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-key", type=Path, required=True)
    parser.add_argument("--claude-references", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()
    gpt_rows = {row["case_id"]: row for row in (json.loads(line) for line in args.gpt_key.read_text(encoding="utf-8").splitlines() if line)}
    claude_rows = {row["case_id"]: row for row in (json.loads(line) for line in args.claude_references.read_text(encoding="utf-8").splitlines() if line)}
    if set(gpt_rows) != set(claude_rows) or len(gpt_rows) != 60:
        raise ValueError("expected the same 60 cases")
    cases = []
    for case_id in sorted(gpt_rows):
        claude_events = {event: claude_rows[case_id]["reference"][event] for event in EVENTS}
        if any(value == "uncertain" for value in claude_events.values()):
            raise ValueError(f"uncertain external event in {case_id}; predeclare exclusion handling")
        cases.append({
            "case_id": case_id,
            "group": gpt_rows[case_id]["selection_group"],
            "gpt": {event: bool(gpt_rows[case_id]["events"][event]) for event in EVENTS},
            "claude": {event: bool(claude_events[event]) for event in EVENTS},
        })
    event_results = {}
    for event in EVENTS:
        left = [row["gpt"][event] for row in cases]
        right = [row["claude"][event] for row in cases]
        confusion = Counter((a, b) for a, b in zip(left, right))
        agreements = sum(a == b for a, b in zip(left, right))
        tp = confusion[(True, True)]
        precision = tp / sum(right) if sum(right) else None
        recall = tp / sum(left) if sum(left) else None
        f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
        event_results[event] = {
            "label": LABELS[event],
            "count": len(left),
            "agreement": agreements / len(left),
            "agreement_wilson_95ci": wilson(agreements, len(left)),
            "cohen_kappa": kappa(left, right),
            "gpt_positive": sum(left),
            "claude_positive": sum(right),
            "confusion_gpt_rows_claude_columns": {
                "true_true": confusion[(True, True)], "true_false": confusion[(True, False)],
                "false_true": confusion[(False, True)], "false_false": confusion[(False, False)],
            },
            "claude_vs_gpt_positive_precision": precision,
            "claude_vs_gpt_positive_recall": recall,
            "claude_vs_gpt_positive_f1": f1,
        }
    all_left = [row["gpt"][event] for row in cases for event in EVENTS]
    all_right = [row["claude"][event] for row in cases for event in EVENTS]
    gpt_rewards = [process_reward(row["gpt"]) for row in cases]
    claude_rewards = [process_reward(row["claude"]) for row in cases]
    group_results = {}
    for group in ("representative", "challenge"):
        subset = [row for row in cases if row["group"] == group]
        left = [row["gpt"][event] for row in subset for event in EVENTS]
        right = [row["claude"][event] for row in subset for event in EVENTS]
        group_results[group] = {"case_count": len(subset), "micro_event_agreement": mean(a == b for a, b in zip(left, right)), "micro_cohen_kappa": kappa(left, right)}
    ci = bootstrap(cases, args.bootstrap_iterations, 20260815)
    payload = {
        "schema_version": 1,
        "status": "completed",
        "scientific_boundary": "GPT-4o versus Claude Sonnet 4.6 machine-machine agreement on a challenge-enriched 60-case panel; not human validation",
        "case_count": 60,
        "event_decision_count": 360,
        "event_results": event_results,
        "overall": {
            "micro_event_agreement": mean(a == b for a, b in zip(all_left, all_right)),
            "micro_event_agreement_cluster_bootstrap_95ci": ci["micro_event_agreement"],
            "micro_cohen_kappa": kappa(all_left, all_right),
            "gpt_process_reward_mean": mean(gpt_rewards),
            "claude_process_reward_mean": mean(claude_rewards),
            "mean_signed_claude_minus_gpt": mean(b - a for a, b in zip(gpt_rewards, claude_rewards)),
            "reward_mae": mean(abs(a - b) for a, b in zip(gpt_rewards, claude_rewards)),
            "reward_mae_cluster_bootstrap_95ci": ci["reward_mae"],
            "reward_exact_agreement": mean(abs(a - b) < 1e-12 for a, b in zip(gpt_rewards, claude_rewards)),
            "reward_within_0p2": mean(abs(a - b) <= 0.2000001 for a, b in zip(gpt_rewards, claude_rewards)),
            "reward_pearson": correlation(gpt_rewards, claude_rewards),
            "reward_spearman": spearman(gpt_rewards, claude_rewards),
            "reward_spearman_cluster_bootstrap_95ci": ci["reward_spearman"],
        },
        "selection_groups": group_results,
        "bootstrap_iterations": args.bootstrap_iterations,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# GPT-4o 训练奖励与 Claude 外部参考一致性", "",
        "> 这是机器—机器一致性，不是专家验证。60 例包含 36 例 representative 与 24 例 challenge，不能当作总体随机样本。", "",
        "| 六事件 | Agreement | Cohen κ | GPT positive | Claude positive | Claude-vs-GPT F1 |", "|---|---:|---:|---:|---:|---:|",
    ]
    for event in EVENTS:
        row = event_results[event]
        kappa_text = "NA" if row["cohen_kappa"] is None else f"{row['cohen_kappa']:.3f}"
        f1_text = "NA" if row["claude_vs_gpt_positive_f1"] is None else f"{row['claude_vs_gpt_positive_f1']:.3f}"
        lines.append(f"| {row['label']} | {row['agreement']:.1%} | {kappa_text} | {row['gpt_positive']} | {row['claude_positive']} | {f1_text} |")
    overall = payload["overall"]
    lines += ["", "## 汇总", "", f"- 360 个事件判断的 micro agreement：{overall['micro_event_agreement']:.1%}，case-cluster bootstrap 95% CI [{ci['micro_event_agreement'][0]:.1%}, {ci['micro_event_agreement'][1]:.1%}]；micro κ={overall['micro_cohen_kappa']:.3f}。", f"- 同一 0.4 公式下：GPT-4o 平均奖励 {overall['gpt_process_reward_mean']:.3f}，Claude {overall['claude_process_reward_mean']:.3f}，Claude 平均低 {abs(overall['mean_signed_claude_minus_gpt']):.3f}。", f"- reward MAE={overall['reward_mae']:.3f}；完全相同 {overall['reward_exact_agreement']:.1%}；差值不超过 0.2 的病例 {overall['reward_within_0p2']:.1%}。", f"- reward Spearman ρ={overall['reward_spearman']:.3f}，bootstrap 95% CI [{ci['reward_spearman'][0]:.3f}, {ci['reward_spearman'][1]:.3f}]。", f"- representative：agreement {group_results['representative']['micro_event_agreement']:.1%}、κ={group_results['representative']['micro_cohen_kappa']:.3f}；challenge：agreement {group_results['challenge']['micro_event_agreement']:.1%}、κ={group_results['challenge']['micro_cohen_kappa']:.3f}。", "", "## 解释边界", "", "整体属于中等一致性，而不是高度可互换。图像分析事件一致性高；逻辑矛盾与错误/过时标准的阳性率差异很大，显示 rubric 边界仍依赖 Judge。真正回应审稿人仍需使用第一阶段病理专家评分。", ""]
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
