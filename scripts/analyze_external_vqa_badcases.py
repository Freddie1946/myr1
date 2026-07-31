#!/usr/bin/env python3
"""Audit frozen external-VQA predictions without changing official metrics.

The parser-aware numbers emitted here are diagnostics only.  The PathVQA exact
match and OmniMedVQA whole-completion SequenceMatcher results remain the formal
scores.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


MODELS = ("sft3000", "sft4000", "stage2_rl")
EXPECTED = {"pathvqa": 6719, "omnimedvqa": 8518}
CHOICE_RE = re.compile(r"<answer\b[^>]*>\s*\(?\s*([A-D])(?=[\s).,:;<\-]|$)", re.I)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def ratio(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 4) if denominator else 0.0


def normalize(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def explicit_choice(completion: str) -> str | None:
    matches = CHOICE_RE.findall(completion)
    return matches[-1].upper() if matches else None


def compact(row: dict[str, Any], *, completion_limit: int = 1200) -> dict[str, Any]:
    keys = (
        "index", "source_record_sha256", "image", "question", "answer",
        "answer_type", "target_choice", "gt_answer", "option_texts", "dataset",
        "question_type", "official_predicted_choice", "reached_generation_cap",
        "generated_token_count",
    )
    result = {key: row.get(key) for key in keys if key in row}
    completion = row.get("completion", "")
    result["completion"] = completion[:completion_limit]
    result["completion_truncated_for_report"] = len(completion) > completion_limit
    return result


def validate_alignment(by_model: dict[str, list[dict[str, Any]]], task: str) -> None:
    expected = EXPECTED[task]
    for model, rows in by_model.items():
        if len(rows) != expected:
            raise ValueError(f"{model}/{task}: expected {expected}, got {len(rows)}")
        if [row.get("index") for row in rows] != list(range(expected)):
            raise ValueError(f"{model}/{task}: indices are not contiguous")
    baseline = [row.get("source_record_sha256") for row in by_model[MODELS[0]]]
    for model in MODELS[1:]:
        observed = [row.get("source_record_sha256") for row in by_model[model]]
        if observed != baseline:
            raise ValueError(f"{model}/{task}: source records are not aligned")


def pathvqa_report(by_model: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    total = EXPECTED["pathvqa"]
    summary: dict[str, Any] = {}
    badcases: dict[str, list[dict[str, Any]]] = {}
    for model, rows in by_model.items():
        exact = sum(bool(row["exact_match"]) for row in rows)
        capped = sum(bool(row["reached_generation_cap"]) for row in rows)
        contained = 0
        contained_wrong = 0
        leading_yes_no_correct = 0
        by_type: dict[str, dict[str, Any]] = {}
        for answer_type in ("yes_no", "free_form"):
            subset = [row for row in rows if row["answer_type"] == answer_type]
            hits = sum(bool(row["exact_match"]) for row in subset)
            by_type[answer_type] = {
                "count": len(subset), "formal_correct": hits,
                "formal_accuracy_percent": ratio(hits, len(subset)),
            }
        selected = []
        for row in rows:
            completion = normalize(row["completion"])
            target = normalize(row["answer"])
            has_target = bool(target and target in completion)
            contained += has_target
            contained_wrong += has_target and not row["exact_match"]
            if row["answer_type"] == "yes_no":
                first = completion.split(maxsplit=1)[0] if completion else ""
                leading_yes_no_correct += first == target
            if has_target and not row["exact_match"] and len(selected) < 100:
                item = compact(row)
                item["diagnostic"] = "normalized target occurs in completion but formal exact match is false"
                selected.append(item)
        cap_groups = {}
        for cap in (False, True):
            subset = [row for row in rows if bool(row["reached_generation_cap"]) is cap]
            hits = sum(bool(row["exact_match"]) for row in subset)
            cap_groups[str(cap).lower()] = {
                "count": len(subset), "formal_correct": hits,
                "formal_accuracy_percent": ratio(hits, len(subset)),
            }
        token_mean = sum(int(row["generated_token_count"]) for row in rows) / total
        summary[model] = {
            "formal_correct": exact,
            "formal_accuracy_percent": ratio(exact, total),
            "mean_generated_tokens": round(token_mean, 4),
            "generation_cap_count": capped,
            "normalized_target_contained_count_diagnostic_only": contained,
            "target_contained_but_formal_wrong_count_diagnostic_only": contained_wrong,
            "leading_yes_no_correct_count_diagnostic_only": leading_yes_no_correct,
            "by_answer_type": by_type,
            "by_generation_cap": cap_groups,
        }
        badcases[model] = selected

    patterns = Counter()
    all_wrong = []
    for index in range(total):
        pattern = "".join("T" if by_model[m][index]["exact_match"] else "F" for m in MODELS)
        patterns[pattern] += 1
        if pattern == "FFF" and len(all_wrong) < 100:
            item = compact(by_model["stage2_rl"][index])
            item["formal_pattern_sft3000_sft4000_stage2"] = pattern
            all_wrong.append(item)
    return {
        "formal_metric": "normalized exact match",
        "diagnostic_metric_policy": "diagnostic only; must not replace the frozen formal score",
        "total": total,
        "summary": summary,
        "joint_formal_correctness_patterns_sft3000_sft4000_stage2": dict(patterns.most_common()),
        "badcases_target_contained_but_formal_wrong": badcases,
        "badcases_all_three_formal_wrong": all_wrong,
    }


def omnimed_report(by_model: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    total = EXPECTED["omnimedvqa"]
    summary: dict[str, Any] = {}
    parser_conflicts: dict[str, list[dict[str, Any]]] = {}
    for model, rows in by_model.items():
        formal = sum(bool(row["official_most_similar_correct"]) for row in rows)
        capped = sum(bool(row["reached_generation_cap"]) for row in rows)
        explicit_available = explicit_correct = 0
        target_mentioned = unique_target_mentioned = 0
        mentioned_formal_wrong = 0
        matrix = Counter()
        source_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        selected = []
        for row in rows:
            target = row["target_choice"]
            parsed = explicit_choice(row["completion"])
            parsed_ok = parsed == target
            formal_ok = bool(row["official_most_similar_correct"])
            explicit_available += parsed is not None
            explicit_correct += parsed_ok
            matrix[f"formal_{formal_ok}_explicit_{parsed_ok}"] += 1
            source_counts[row["dataset"]][0] += 1
            source_counts[row["dataset"]][1] += formal_ok

            normalized_completion = normalize(row["completion"])
            mentions = [
                choice for choice, text in row["option_texts"].items()
                if normalize(text) and normalize(text) in normalized_completion
            ]
            target_is_mentioned = target in mentions
            target_mentioned += target_is_mentioned
            unique_target_mentioned += mentions == [target]
            mentioned_formal_wrong += target_is_mentioned and not formal_ok
            if parsed_ok and not formal_ok and len(selected) < 100:
                item = compact(row)
                item["diagnostic_explicit_choice"] = parsed
                item["diagnostic"] = "explicit answer choice is correct but frozen whole-completion matcher is wrong"
                selected.append(item)

        cap_groups = {}
        for cap in (False, True):
            subset = [row for row in rows if bool(row["reached_generation_cap"]) is cap]
            hits = sum(bool(row["official_most_similar_correct"]) for row in subset)
            cap_groups[str(cap).lower()] = {
                "count": len(subset), "formal_correct": hits,
                "formal_accuracy_percent": ratio(hits, len(subset)),
            }
        token_mean = sum(int(row["generated_token_count"]) for row in rows) / total
        summary[model] = {
            "formal_correct": formal,
            "formal_accuracy_percent": ratio(formal, total),
            "mean_generated_tokens": round(token_mean, 4),
            "generation_cap_count": capped,
            "explicit_choice_available_count_diagnostic_only": explicit_available,
            "explicit_choice_correct_count_diagnostic_only": explicit_correct,
            "explicit_choice_accuracy_when_available_percent_diagnostic_only": ratio(explicit_correct, explicit_available),
            "target_option_text_mentioned_count_diagnostic_only": target_mentioned,
            "unique_target_option_text_mentioned_count_diagnostic_only": unique_target_mentioned,
            "target_mentioned_but_formal_wrong_count_diagnostic_only": mentioned_formal_wrong,
            "formal_explicit_correctness_matrix_diagnostic_only": dict(matrix),
            "by_generation_cap": cap_groups,
            "by_source": {
                source: {"count": values[0], "formal_correct": values[1],
                         "formal_accuracy_percent": ratio(values[1], values[0])}
                for source, values in sorted(source_counts.items())
            },
        }
        parser_conflicts[model] = selected

    patterns = Counter()
    all_wrong = []
    for index in range(total):
        pattern = "".join(
            "T" if by_model[m][index]["official_most_similar_correct"] else "F"
            for m in MODELS
        )
        patterns[pattern] += 1
        if pattern == "FFF" and len(all_wrong) < 100:
            item = compact(by_model["stage2_rl"][index])
            item["formal_pattern_sft3000_sft4000_stage2"] = pattern
            item["diagnostic_explicit_choice"] = explicit_choice(
                by_model["stage2_rl"][index]["completion"]
            )
            all_wrong.append(item)
    return {
        "formal_metric": "official whole-completion SequenceMatcher option mapping",
        "diagnostic_metric_policy": "diagnostic only; must not replace the frozen formal score",
        "total": total,
        "summary": summary,
        "joint_formal_correctness_patterns_sft3000_sft4000_stage2": dict(patterns.most_common()),
        "badcases_explicit_choice_correct_but_formal_wrong": parser_conflicts,
        "badcases_all_three_formal_wrong": all_wrong,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    reports = {}
    for task in EXPECTED:
        by_model = {
            model: read_jsonl(args.run_root / model / task / "predictions.jsonl")
            for model in MODELS
        }
        validate_alignment(by_model, task)
        reports[task] = pathvqa_report(by_model) if task == "pathvqa" else omnimed_report(by_model)

    payload = {
        "schema_version": 1,
        "scope": list(MODELS),
        "source_run_root": str(args.run_root.resolve()),
        "metric_integrity": "Formal frozen metrics are preserved; all alternative parsers are diagnostic only.",
        **reports,
    }
    write_json(args.output_root / "badcase_analysis.json", payload)
    for task, report in reports.items():
        for category, value in report.items():
            if category.startswith("badcases_"):
                write_json(args.output_root / f"{task}_{category}.json", value)
    print(json.dumps({
        "status": "passed",
        "output": str((args.output_root / "badcase_analysis.json").resolve()),
        "pathvqa_formal": {m: reports["pathvqa"]["summary"][m]["formal_accuracy_percent"] for m in MODELS},
        "omnimedvqa_formal": {m: reports["omnimedvqa"]["summary"][m]["formal_accuracy_percent"] for m in MODELS},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
