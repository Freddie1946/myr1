#!/usr/bin/env python3
"""Paired Stage2/Stage3 PathMMU flip and bad-case analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TOKEN = re.compile(r"[a-z0-9]+")


def load(path: Path) -> dict[int, dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    mapping = {int(row["index"]): row for row in rows}
    if len(rows) != len(mapping):
        raise ValueError(f"duplicate indices: {path}")
    return mapping


def correct(row: dict[str, Any]) -> bool:
    return row.get("accuracy_reward") == 1.0


def tokens(text: str) -> set[str]:
    return set(TOKEN.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    left, right = tokens(a), tokens(b)
    return len(left & right) / len(left | right) if left or right else 1.0


def mcnemar_exact(stage2_only: int, stage3_only: int) -> float:
    n = stage2_only + stage3_only
    if n == 0:
        return 1.0
    observed = min(stage2_only, stage3_only)
    cdf = sum(math.comb(n, k) for k in range(observed + 1)) / (2 ** n)
    return min(1.0, 2 * cdf)


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--stage3", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples-per-direction", type=int, default=12)
    parser.add_argument("--expected-count", type=int, default=999)
    parser.add_argument("--split-name", default="test999")
    args = parser.parse_args()
    stage2, stage3 = load(args.stage2), load(args.stage3)
    if set(stage2) != set(stage3) or len(stage2) != args.expected_count:
        raise ValueError(
            f"paired {args.split_name} predictions of count {args.expected_count} are required"
        )
    for index in stage2:
        left_hash = stage2[index].get("source_record_sha256")
        right_hash = stage3[index].get("source_record_sha256")
        if left_hash and right_hash:
            matched = left_hash == right_hash
        else:
            matched = all(
                stage2[index].get(field) == stage3[index].get(field)
                for field in ("image", "problem", "solution", "target_choice")
            )
        if not matched:
            raise ValueError(f"source mismatch at index {index}")

    transitions: dict[str, list[int]] = defaultdict(list)
    choice_changes = 0
    for index in sorted(stage2):
        a, b = correct(stage2[index]), correct(stage3[index])
        key = (
            "both_correct" if a and b else "stage2_only" if a else
            "stage3_only" if b else "both_wrong"
        )
        transitions[key].append(index)
        choice_changes += stage2[index].get("predicted_choice") != stage3[index].get("predicted_choice")

    by_target = {}
    for target in "ABCD":
        indices = [i for i, row in stage2.items() if row.get("target_choice") == target]
        by_target[target] = {
            "count": len(indices),
            "stage2_correct": sum(correct(stage2[i]) for i in indices),
            "stage3_correct": sum(correct(stage3[i]) for i in indices),
            "stage2_accuracy": mean([float(correct(stage2[i])) for i in indices]),
            "stage3_accuracy": mean([float(correct(stage3[i])) for i in indices]),
        }

    diagnostics = {}
    for key, indices in transitions.items():
        diagnostics[key] = {
            "count": len(indices),
            "stage2_mean_generated_tokens": mean([float(stage2[i]["generated_token_count"]) for i in indices]),
            "stage3_mean_generated_tokens": mean([float(stage3[i]["generated_token_count"]) for i in indices]),
            "stage2_mean_reference_token_jaccard": mean([
                jaccard(stage2[i]["completion"], stage2[i]["solution"]) for i in indices
            ]),
            "stage3_mean_reference_token_jaccard": mean([
                jaccard(stage3[i]["completion"], stage3[i]["solution"]) for i in indices
            ]),
        }

    def select(indices: list[int]) -> list[int]:
        ranked = sorted(
            indices,
            key=lambda i: hashlib.sha256(
                f"badcase-v1:{stage2[i]['source_record_sha256']}".encode()
            ).hexdigest(),
        )
        selected, images = [], set()
        for index in ranked:
            image = str(Path(stage2[index]["image"]).resolve())
            if image in images:
                continue
            selected.append(index)
            images.add(image)
            if len(selected) == args.examples_per_direction:
                break
        return selected

    selected = {
        "regressions": select(transitions["stage2_only"]),
        "improvements": select(transitions["stage3_only"]),
    }
    examples = []
    for direction, indices in selected.items():
        for index in indices:
            examples.append({
                "direction": direction,
                "index": index,
                "source_record_sha256": stage2[index]["source_record_sha256"],
                "image": stage2[index]["image"],
                "problem": stage2[index]["problem"],
                "reference": stage2[index]["solution"],
                "target_choice": stage2[index]["target_choice"],
                "stage2_predicted_choice": stage2[index]["predicted_choice"],
                "stage2_completion": stage2[index]["completion"],
                "stage3_predicted_choice": stage3[index]["predicted_choice"],
                "stage3_completion": stage3[index]["completion"],
            })

    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage2_predictions": str(args.stage2.resolve()),
        "stage3_predictions": str(args.stage3.resolve()),
        "count": args.expected_count,
        "split_name": args.split_name,
        "stage2_correct": sum(correct(row) for row in stage2.values()),
        "stage3_correct": sum(correct(row) for row in stage3.values()),
        "stage3_minus_stage2_correct": sum(correct(row) for row in stage3.values()) - sum(correct(row) for row in stage2.values()),
        "choice_changed_count": choice_changes,
        "transition_counts": {key: len(value) for key, value in transitions.items()},
        "discordant_pair_mcnemar_exact_two_sided_p": mcnemar_exact(
            len(transitions["stage2_only"]), len(transitions["stage3_only"])
        ),
        "stage2_predicted_choice_distribution": Counter(
            row.get("predicted_choice") for row in stage2.values()
        ),
        "stage3_predicted_choice_distribution": Counter(
            row.get("predicted_choice") for row in stage3.values()
        ),
        "by_target_choice": by_target,
        "transition_diagnostics": diagnostics,
        "selected_examples": selected,
    }
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "selected_badcases.jsonl").open("w", encoding="utf-8") as handle:
        for row in examples:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        f"# Stage2 versus GPT-4o Stage3 PathMMU {args.split_name} paired analysis", "",
        f"- Stage2: {result['stage2_correct']}/{args.expected_count} ({result['stage2_correct']/args.expected_count:.2%})",
        f"- Stage3: {result['stage3_correct']}/{args.expected_count} ({result['stage3_correct']/args.expected_count:.2%})",
        f"- Net change: {result['stage3_minus_stage2_correct']:+d} questions",
        f"- Correct in Stage2 only (regressions): {len(transitions['stage2_only'])}",
        f"- Correct in Stage3 only (improvements): {len(transitions['stage3_only'])}",
        f"- Both correct: {len(transitions['both_correct'])}; both wrong: {len(transitions['both_wrong'])}",
        f"- Predicted option changed: {choice_changes}/{args.expected_count}",
        f"- Exact paired McNemar p: {result['discordant_pair_mcnemar_exact_two_sided_p']:.4f}", "",
        "## Accuracy by target option", "",
        "| Target | n | Stage2 | Stage3 | Difference |", "| --- | ---: | ---: | ---: | ---: |",
    ]
    for target, value in by_target.items():
        lines.append(
            f"| {target} | {value['count']} | {value['stage2_accuracy']:.2%} | "
            f"{value['stage3_accuracy']:.2%} | {value['stage3_accuracy']-value['stage2_accuracy']:+.2%} |"
        )
    lines.extend(["", "## Selected paired cases", ""])
    for row in examples:
        lines.extend([
            f"### {row['direction']} — index {row['index']}", "",
            f"Image: `{row['image']}`", "",
            row["problem"], "",
            f"Reference: {row['reference']}", "",
            f"Stage2 ({row['stage2_predicted_choice']}): {row['stage2_completion']}", "",
            f"Stage3 ({row['stage3_predicted_choice']}): {row['stage3_completion']}", "",
        ])
    (output / "paired_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
