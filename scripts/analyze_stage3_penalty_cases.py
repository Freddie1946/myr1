#!/usr/bin/env python3
"""Build a strictly offline comparison of matched Stage3 penalty pilots.

The analysis reads reward audits, cached six-event Judge responses, and matched
validation predictions.  It imports no network client and performs no inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
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
REWARD_TYPES = ("accuracy", "format", "process")
SOURCE_FIELDS = ("image", "problem", "solution")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_arm(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("--arm must be LABEL=RUN_ROOT")
    return label, Path(path)


def choice_from_completion(completion: str) -> str | None:
    matches = re.findall(r"<answer>\s*([A-D])(?:\)|\b)", completion, flags=re.IGNORECASE)
    return matches[-1].upper() if matches else None


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty collection")
    return statistics.fmean(values)


def audit_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return int(row["rank"]), int(row["call_index"]), int(row["item_index"])


def load_reward_audit(run_root: Path) -> tuple[dict[str, Any], dict[tuple[int, int, int], dict[str, Any]]]:
    files = sorted((run_root / "reward_audit").glob("*.jsonl"))
    if not files:
        raise ValueError(f"no reward audit files under {run_root}")
    by_type: dict[str, list[dict[str, Any]]] = {name: [] for name in REWARD_TYPES}
    for path in files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            row = json.loads(line)
            reward_type = row.get("reward_type")
            if reward_type not in by_type:
                raise ValueError(f"unexpected reward type at {path}:{line_number}")
            by_type[reward_type].append(row)
    keyed: dict[str, dict[tuple[int, int, int], dict[str, Any]]] = {}
    for reward_type, rows in by_type.items():
        current: dict[tuple[int, int, int], dict[str, Any]] = {}
        for row in rows:
            key = audit_key(row)
            if key in current:
                raise ValueError(f"duplicate {reward_type} audit key {key} under {run_root}")
            current[key] = row
        keyed[reward_type] = current
    reference_keys = set(keyed["process"])
    if any(set(keyed[name]) != reference_keys for name in REWARD_TYPES):
        raise ValueError(f"reward-type audit keys do not align under {run_root}")
    process_rows = keyed["process"]
    summary = {
        "files": [
            {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in files
        ],
        "unique_rollout_count": len(process_rows),
        "serialized_record_count": sum(len(rows) for rows in by_type.values()),
        "unique_record_index_count": len({int(row["record_index"]) for row in process_rows.values()}),
        "call_index_minimum": min(int(row["call_index"]) for row in process_rows.values()),
        "call_index_maximum": max(int(row["call_index"]) for row in process_rows.values()),
        "reward_means": {
            name: mean([float(row["reward"]) for row in rows]) for name, rows in by_type.items()
        },
        "reward_distributions": {
            name: {
                str(value): count
                for value, count in sorted(Counter(float(row["reward"]) for row in rows).items())
            }
            for name, rows in by_type.items()
        },
        "completion_character_mean": mean(
            [float(len(row["completion"])) for row in process_rows.values()]
        ),
        "completion_character_median": statistics.median(
            [len(row["completion"]) for row in process_rows.values()]
        ),
        "quarter_reward_means": {},
    }
    for lower in (0, 25, 50, 75):
        upper = lower + 25
        quarter: dict[str, float | None] = {}
        for name, rows in by_type.items():
            values = [
                float(row["reward"])
                for row in rows
                if lower <= int(row["call_index"]) < upper
            ]
            quarter[name] = mean(values) if values else None
        summary["quarter_reward_means"][f"calls_{lower}_{upper - 1}"] = quarter
    return summary, process_rows


def parse_judge_content(record: dict[str, Any], path: Path) -> dict[str, Any]:
    try:
        content = record["response"]["choices"][0]["message"]["content"]
        parsed = json.loads(content) if isinstance(content, str) else content
    except (KeyError, IndexError, json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"cannot parse cached Judge response: {path}") from error
    if not isinstance(parsed, dict) or any(type(parsed.get(name)) is not bool for name in EVENTS):
        raise ValueError(f"cached Judge response lacks six booleans: {path}")
    return parsed


def load_judge_cache(run_root: Path) -> dict[str, Any]:
    files = sorted((run_root / "judge" / "cache").glob("**/*.json"))
    if not files:
        raise ValueError(f"no Judge cache records under {run_root}")
    event_counts: Counter[str] = Counter()
    score_values: dict[str, list[float]] = defaultdict(list)
    served_models: Counter[str] = Counter()
    cache_keys: set[str] = set()
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        cache_key = record.get("cache_key")
        if not isinstance(cache_key, str) or cache_key in cache_keys:
            raise ValueError(f"missing or duplicate cache key: {path}")
        cache_keys.add(cache_key)
        parsed = parse_judge_content(record, path)
        for name in EVENTS:
            event_counts[name] += int(parsed[name])
        for name, value in record.get("scores", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                score_values[name].append(float(value))
        served_models[str(record.get("served_model"))] += 1
    count = len(files)
    return {
        "cache_record_count": count,
        "event_true_counts": {name: event_counts[name] for name in EVENTS},
        "event_true_rates": {name: event_counts[name] / count for name in EVENTS},
        "score_means": {name: mean(values) for name, values in sorted(score_values.items())},
        "served_models": dict(served_models),
        "cache_manifest_sha256": hashlib.sha256(
            "".join(f"{sha256(path)}  {path.name}\n" for path in files).encode("utf-8")
        ).hexdigest(),
    }


def load_validation(run_root: Path, validation_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = run_root / validation_name
    metrics_path = root / "metrics.json"
    predictions_path = root / "predictions.jsonl"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines()]
    if metrics.get("status") != "completed" or metrics.get("count") != len(rows):
        raise ValueError(f"incomplete validation result under {root}")
    if [row.get("index") for row in rows] != list(range(len(rows))):
        raise ValueError(f"validation indices are not contiguous under {root}")
    summary = {
        "root": str(root.resolve()),
        "metrics_sha256": sha256(metrics_path),
        "predictions_sha256": sha256(predictions_path),
        "count": len(rows),
        "correct": sum(int(bool(row["accuracy_reward"])) for row in rows),
        "accuracy": mean([float(row["accuracy_reward"]) for row in rows]),
        "format_accuracy": mean([float(row["format_reward"]) for row in rows]),
        "choice_extracted": sum(row.get("predicted_choice") is not None for row in rows),
        "mean_generated_tokens": mean([float(row["generated_token_count"]) for row in rows]),
    }
    return summary, rows


def compare_training(
    audits: dict[str, dict[tuple[int, int, int], dict[str, Any]]]
) -> dict[str, Any]:
    labels = list(audits)
    reference_keys = set(audits[labels[0]])
    if any(set(audits[label]) != reference_keys for label in labels[1:]):
        raise ValueError("training audit keys differ across arms")
    for key in reference_keys:
        reference = audits[labels[0]][key]
        for label in labels[1:]:
            row = audits[label][key]
            for field in ("record_index", "problem", "solution", "image_sha256"):
                if row.get(field) != reference.get(field):
                    raise ValueError(f"training source mismatch for {field} at {key}")
    pairwise: dict[str, Any] = {}
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            exact = 0
            same_choice = 0
            for key in reference_keys:
                left_completion = audits[left][key]["completion"]
                right_completion = audits[right][key]["completion"]
                exact += left_completion == right_completion
                same_choice += choice_from_completion(left_completion) == choice_from_completion(
                    right_completion
                )
            pairwise[f"{left}_vs_{right}"] = {
                "aligned_rollouts": len(reference_keys),
                "exact_completion_matches": exact,
                "same_choice_count": same_choice,
            }
    return {
        "strictly_aligned": True,
        "alignment_key": ["rank", "call_index", "item_index"],
        "aligned_rollout_count": len(reference_keys),
        "pairwise": pairwise,
    }


def compare_validation(
    rows_by_arm: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = list(rows_by_arm)
    count = len(rows_by_arm[labels[0]])
    if any(len(rows_by_arm[label]) != count for label in labels[1:]):
        raise ValueError("validation counts differ across arms")
    pattern_counts: Counter[str] = Counter()
    cases: list[dict[str, Any]] = []
    for index in range(count):
        reference = rows_by_arm[labels[0]][index]
        for label in labels[1:]:
            for field in SOURCE_FIELDS:
                if rows_by_arm[label][index].get(field) != reference.get(field):
                    raise ValueError(f"validation source mismatch for {field} at {index}")
        pattern = "".join(
            str(int(bool(rows_by_arm[label][index]["accuracy_reward"]))) for label in labels
        )
        pattern_counts[pattern] += 1
        if len(set(pattern)) == 1:
            continue
        arm_outputs = {}
        for label in labels:
            row = rows_by_arm[label][index]
            arm_outputs[label] = {
                "predicted_choice": row.get("predicted_choice"),
                "correct": bool(row["accuracy_reward"]),
                "format_correct": bool(row["format_reward"]),
                "generated_token_count": int(row["generated_token_count"]),
                "completion": row["completion"],
            }
        cases.append(
            {
                "index": index,
                "correctness_pattern": pattern,
                "image": reference["image"],
                "problem": reference["problem"],
                "solution": reference["solution"],
                "arms": arm_outputs,
            }
        )
    pairwise: dict[str, Any] = {}
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            left_rows, right_rows = rows_by_arm[left], rows_by_arm[right]
            improved = sum(
                not bool(a["accuracy_reward"]) and bool(b["accuracy_reward"])
                for a, b in zip(left_rows, right_rows)
            )
            regressed = sum(
                bool(a["accuracy_reward"]) and not bool(b["accuracy_reward"])
                for a, b in zip(left_rows, right_rows)
            )
            pairwise[f"{left}_to_{right}"] = {
                "improved": improved,
                "regressed": regressed,
                "net_correct": improved - regressed,
                "same_choice_count": sum(
                    a.get("predicted_choice") == b.get("predicted_choice")
                    for a, b in zip(left_rows, right_rows)
                ),
                "exact_completion_matches": sum(
                    a["completion"] == b["completion"] for a, b in zip(left_rows, right_rows)
                ),
            }
    return (
        {
            "strictly_aligned": True,
            "arm_order_for_patterns": labels,
            "count": count,
            "pattern_counts": dict(sorted(pattern_counts.items())),
            "unanimous_correctness_count": pattern_counts["0" * len(labels)]
            + pattern_counts["1" * len(labels)],
            "disagreement_case_count": len(cases),
            "pairwise": pairwise,
        },
        cases,
    )


def analyze(arms: list[tuple[str, Path]], validation_name: str) -> dict[str, Any]:
    if len(arms) != 3 or len({label for label, _ in arms}) != 3:
        raise ValueError("exactly three uniquely labeled arms are required")
    arm_summaries: dict[str, Any] = {}
    audits: dict[str, dict[tuple[int, int, int], dict[str, Any]]] = {}
    validation_rows: dict[str, list[dict[str, Any]]] = {}
    for label, run_root in arms:
        if not run_root.is_dir():
            raise ValueError(f"run root does not exist: {run_root}")
        reward_summary, audits[label] = load_reward_audit(run_root)
        validation_summary, validation_rows[label] = load_validation(run_root, validation_name)
        arm_summaries[label] = {
            "run_root": str(run_root.resolve()),
            "reward_audit": reward_summary,
            "judge_cache": load_judge_cache(run_root),
            "validation": validation_summary,
        }
    validation_comparison, disagreement_cases = compare_validation(validation_rows)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "analysis_type": "offline_matched_stage3_penalty_case_analysis",
        "network_calls": 0,
        "paid_calls": 0,
        "arms": arm_summaries,
        "training_alignment": compare_training(audits),
        "validation_comparison": validation_comparison,
        "validation_disagreement_cases": disagreement_cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", type=parse_arm, required=True)
    parser.add_argument("--validation-name", default="validation_0385_greedy_20260801")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite existing analysis: {args.output}")
    payload = analyze(args.arm, args.validation_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f"{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "status": "complete",
                "network_calls": 0,
                "paid_calls": 0,
                "disagreement_cases": payload["validation_comparison"]["disagreement_case_count"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
