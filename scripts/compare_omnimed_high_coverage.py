#!/usr/bin/env python3
"""Paired strict-final comparison of Base and corrective Stage3 OmniMedVQA outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from external_vqa_contract import omnimed_score, sha256_file


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def scoring_record(row: dict) -> dict:
    value = dict(row)
    for letter, text in row.get("option_texts", {}).items():
        value[f"option_{letter}"] = text
    return value


def paired_bootstrap_difference(base: np.ndarray, trained: np.ndarray, seed: int = 42, draws: int = 10000) -> dict:
    rng = np.random.default_rng(seed)
    differences = []
    for size in range(0, draws, 500):
        batch = min(500, draws - size)
        indices = rng.integers(0, len(base), size=(batch, len(base)))
        differences.extend((trained[indices].mean(axis=1) - base[indices].mean(axis=1)).tolist())
    values = np.asarray(differences)
    return {
        "draws": draws,
        "seed": seed,
        "trained_minus_base": float(trained.mean() - base.mean()),
        "percentile_95_ci": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--trained", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    base_rows, trained_rows = load(args.base), load(args.trained)
    if len(base_rows) != 8518 or len(trained_rows) != 8518:
        raise ValueError("expected two complete 8518-row predictions")
    base_correct, trained_correct, base_available, trained_available = [], [], [], []
    transition = Counter()
    choice_position = {
        "base_predictions": Counter(),
        "trained_predictions": Counter(),
        "targets": Counter(),
        "base_correct_by_target": Counter(),
        "trained_correct_by_target": Counter(),
    }
    strata: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"base": [], "trained": []})
    for index, (base, trained) in enumerate(zip(base_rows, trained_rows)):
        if base.get("source_record_sha256") != trained.get("source_record_sha256"):
            raise ValueError(f"row alignment failure at {index}")
        bs = omnimed_score(str(base["completion"]), scoring_record(base))
        ts = omnimed_score(str(trained["completion"]), scoring_record(trained))
        bc, tc = bool(bs["strict_final_correct"]), bool(ts["strict_final_correct"])
        ba, ta = bool(bs["strict_final_answer_available"]), bool(ts["strict_final_answer_available"])
        target_choice = str(base["target_choice"])
        base_choice = str(bs.get("strict_final_predicted_choice") or "unresolved")
        trained_choice = str(ts.get("strict_final_predicted_choice") or "unresolved")
        base_correct.append(bc); trained_correct.append(tc)
        base_available.append(ba); trained_available.append(ta)
        transition[(bc, tc)] += 1
        choice_position["base_predictions"][base_choice] += 1
        choice_position["trained_predictions"][trained_choice] += 1
        choice_position["targets"][target_choice] += 1
        choice_position["base_correct_by_target"][target_choice] += int(bc)
        choice_position["trained_correct_by_target"][target_choice] += int(tc)
        for key in (f"dataset:{base['dataset']}", f"question_type:{base['question_type']}"):
            strata[key]["base"].append(int(bc)); strata[key]["trained"].append(int(tc))
    b = np.asarray(base_correct, dtype=float); t = np.asarray(trained_correct, dtype=float)
    base_only = transition[(True, False)]
    trained_only = transition[(False, True)]
    discordant = base_only + trained_only
    p_value = binomtest(min(base_only, trained_only), discordant, 0.5).pvalue if discordant else 1.0
    result = {
        "schema_version": 1,
        "status": "completed",
        "formal_result": False,
        "result_role": "post_hoc_strict_final_high_coverage_sensitivity",
        "base": str(args.base.resolve()),
        "base_sha256": sha256_file(args.base),
        "trained": str(args.trained.resolve()),
        "trained_sha256": sha256_file(args.trained),
        "count": len(b),
        "base_strict_coverage": sum(base_available) / len(b),
        "trained_strict_coverage": sum(trained_available) / len(b),
        "base_strict_accuracy": float(b.mean()),
        "trained_strict_accuracy": float(t.mean()),
        "paired_transitions": {
            "both_correct": transition[(True, True)],
            "both_wrong": transition[(False, False)],
            "base_correct_trained_wrong": base_only,
            "base_wrong_trained_correct": trained_only,
        },
        "mcnemar_exact_two_sided_p": float(p_value),
        "paired_bootstrap": paired_bootstrap_difference(b, t),
        "choice_position": {
            "base_prediction_counts": dict(sorted(choice_position["base_predictions"].items())),
            "trained_prediction_counts": dict(sorted(choice_position["trained_predictions"].items())),
            "target_counts": dict(sorted(choice_position["targets"].items())),
            "accuracy_by_target_choice": {
                choice: {
                    "count": count,
                    "base_accuracy": choice_position["base_correct_by_target"][choice] / count,
                    "trained_accuracy": choice_position["trained_correct_by_target"][choice] / count,
                    "trained_minus_base": (
                        choice_position["trained_correct_by_target"][choice]
                        - choice_position["base_correct_by_target"][choice]
                    ) / count,
                }
                for choice, count in sorted(choice_position["targets"].items())
            },
        },
        "strata": {
            key: {
                "count": len(value["base"]),
                "base_accuracy": float(np.mean(value["base"])),
                "trained_accuracy": float(np.mean(value["trained"])),
                "trained_minus_base": float(np.mean(value["trained"]) - np.mean(value["base"])),
            }
            for key, value in sorted(strata.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("base_strict_coverage", "trained_strict_coverage", "base_strict_accuracy", "trained_strict_accuracy", "mcnemar_exact_two_sided_p")}))


if __name__ == "__main__":
    main()
