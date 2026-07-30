#!/usr/bin/env python3
"""Build a reproducible joint PathMMU score and bad-case report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

from pathmmu_rewards import choice_letter


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="NAME=/absolute/path/to/predictions.jsonl",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def parse_models(values: list[str]) -> dict[str, Path]:
    models: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"invalid --model value: {value}")
        if name in models:
            raise ValueError(f"duplicate model name: {name}")
        models[name] = Path(raw_path)
    return models


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    args = parse_args()
    models = parse_models(args.model)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    rows_by_model = {name: read_rows(path) for name, path in models.items()}
    prediction_hashes = {name: sha256_file(path) for name, path in models.items()}
    counts = {name: len(rows) for name, rows in rows_by_model.items()}
    if set(counts.values()) != {999}:
        raise ValueError(f"every input must contain 999 rows: {counts}")
    names = list(models)

    cases: list[dict[str, Any]] = []
    for index in range(999):
        source_rows = {name: rows_by_model[name][index] for name in names}
        reference = source_rows[names[0]]
        target = choice_letter(reference["solution"])
        if target is None:
            raise ValueError(f"no target choice at row {index}")
        for name, row in source_rows.items():
            if row["index"] != index:
                raise ValueError(f"{name}: unexpected index at row {index}")
            if choice_letter(row["solution"]) != target:
                raise ValueError(f"{name}: target mismatch at row {index}")

        outputs = {}
        for name, row in source_rows.items():
            completion = row.get("completion")
            predicted = row.get("predicted_choice") or choice_letter(completion or "")
            format_correct = (
                bool(row["format_reward"]) if "format_reward" in row else None
            )
            outputs[name] = {
                "predicted_choice": predicted,
                "correct": predicted == target,
                "format_correct": format_correct,
                "generated_token_count": row.get("generated_token_count"),
                "completion": completion,
                "similarities": row.get("similarities"),
                "source_predictions_sha256": prediction_hashes[name],
            }
        correct_models = [name for name in names if outputs[name]["correct"]]
        parse_failure_models = [
            name for name in names if outputs[name]["predicted_choice"] is None
        ]
        format_failure_models = [
            name for name in names if outputs[name]["format_correct"] is False
        ]
        extracted_choices = {
            outputs[name]["predicted_choice"]
            for name in names
            if outputs[name]["predicted_choice"] is not None
        }
        cases.append(
            {
                "index": index,
                "image": reference["image"],
                "problem": reference["problem"],
                "solution": reference["solution"],
                "target_choice": target,
                "correct_count": len(correct_models),
                "correct_models": correct_models,
                "incorrect_models": [name for name in names if name not in correct_models],
                "parse_failure_models": parse_failure_models,
                "format_failure_models": format_failure_models,
                "all_models_wrong": not correct_models,
                "all_models_correct": len(correct_models) == len(names),
                "prediction_disagreement": len(extracted_choices) > 1
                or bool(parse_failure_models),
                "outputs": outputs,
            }
        )

    model_summary = {}
    for name in names:
        correct = sum(case["outputs"][name]["correct"] for case in cases)
        extracted = sum(
            case["outputs"][name]["predicted_choice"] is not None for case in cases
        )
        format_applicable = sum(
            case["outputs"][name]["format_correct"] is not None for case in cases
        )
        format_correct = sum(
            case["outputs"][name]["format_correct"] is True for case in cases
        )
        model_summary[name] = {
            "correct": correct,
            "count": 999,
            "accuracy": correct / 999,
            "choice_extracted": extracted,
            "format_applicable": format_applicable,
            "format_correct": format_correct if format_applicable else None,
            "predictions_path": str(models[name].resolve()),
            "predictions_sha256": prediction_hashes[name],
        }

    pairwise = {}
    for left, right in combinations(names, 2):
        left_only = sum(
            case["outputs"][left]["correct"]
            and not case["outputs"][right]["correct"]
            for case in cases
        )
        right_only = sum(
            case["outputs"][right]["correct"]
            and not case["outputs"][left]["correct"]
            for case in cases
        )
        pairwise[f"{left}__vs__{right}"] = {
            f"{left}_only_correct": left_only,
            f"{right}_only_correct": right_only,
            "both_correct": sum(
                case["outputs"][left]["correct"]
                and case["outputs"][right]["correct"]
                for case in cases
            ),
            "both_wrong": sum(
                not case["outputs"][left]["correct"]
                and not case["outputs"][right]["correct"]
                for case in cases
            ),
        }

    summary = {
        "schema_version": 1,
        "purpose": "pre-Stage3 PathMMU test999 development diagnostic",
        "test_accessed": True,
        "diagnostic_only": True,
        "eligible_for_untouched_final_stage3_claim": False,
        "parser": "scripts/pathmmu_rewards.py",
        "parser_sha256": sha256_file(Path(__file__).with_name("pathmmu_rewards.py")),
        "models": model_summary,
        "strata": {
            "all_models_correct": sum(case["all_models_correct"] for case in cases),
            "all_models_wrong": sum(case["all_models_wrong"] for case in cases),
            "prediction_disagreement": sum(
                case["prediction_disagreement"] for case in cases
            ),
            "any_parse_failure": sum(
                bool(case["parse_failure_models"]) for case in cases
            ),
            "any_format_failure": sum(
                bool(case["format_failure_models"]) for case in cases
            ),
            "correct_count_histogram": {
                str(count): sum(case["correct_count"] == count for case in cases)
                for count in range(len(names) + 1)
            },
        },
        "pairwise_descriptive_counts": pairwise,
    }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    badcases_path = args.output_dir / "badcases.jsonl"
    with badcases_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            if not case["all_models_correct"]:
                handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    index_path = args.output_dir / "badcase_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "target_choice",
                "correct_count",
                "correct_models",
                "parse_failure_models",
                "format_failure_models",
                "all_models_wrong",
                "prediction_disagreement",
                "image",
            ]
        )
        for case in cases:
            if not case["all_models_correct"]:
                writer.writerow(
                    [
                        case["index"],
                        case["target_choice"],
                        case["correct_count"],
                        "|".join(case["correct_models"]),
                        "|".join(case["parse_failure_models"]),
                        "|".join(case["format_failure_models"]),
                        case["all_models_wrong"],
                        case["prediction_disagreement"],
                        case["image"],
                    ]
                )

    manifest = {
        "summary_sha256": sha256_file(summary_path),
        "badcases_sha256": sha256_file(badcases_path),
        "badcase_index_sha256": sha256_file(index_path),
    }
    (args.output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**summary["strata"], "models": model_summary}, indent=2))


if __name__ == "__main__":
    main()
