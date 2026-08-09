#!/usr/bin/env python3
"""Image-cluster bootstrap confidence intervals for aligned PathMMU predictions."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> tuple[dict[str, dict[int, dict[str, Any]]], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    loaded: dict[str, dict[int, dict[str, Any]]] = {}
    for item in manifest["models"]:
        label = str(item["label"])
        rows = [json.loads(line) for line in Path(item["predictions"]).read_text().splitlines() if line.strip()]
        mapping = {int(row["index"]): row for row in rows}
        if len(mapping) != len(rows):
            raise ValueError(f"duplicate indices for {label}")
        loaded[label] = mapping
    return loaded, manifest


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-label", default="Stage3-GPT4o-selected")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    if args.replicates < 1000:
        raise ValueError("at least 1000 bootstrap replicates are required")

    loaded, manifest = load_manifest(args.manifest)
    if args.reference_label not in loaded:
        raise ValueError("reference label is absent")
    common = set.intersection(*(set(rows) for rows in loaded.values()))
    if len(common) != 999:
        raise ValueError(f"aligned full test999 required, got {len(common)} common rows")
    anchor = loaded[next(iter(loaded))]
    for index in common:
        source = anchor[index]["source_record_sha256"]
        for label, rows in loaded.items():
            if rows[index]["source_record_sha256"] != source:
                raise ValueError(f"source mismatch for {label} index {index}")

    clusters: dict[str, list[int]] = defaultdict(list)
    for index in sorted(common):
        image = str(Path(anchor[index]["image"]).resolve())
        clusters[image].append(index)
    cluster_names = sorted(clusters)
    correctness = {
        label: {i: 1.0 if rows[i].get("accuracy_reward") == 1.0 else 0.0 for i in common}
        for label, rows in loaded.items()
    }
    point = {label: sum(values.values()) / len(common) for label, values in correctness.items()}
    samples: dict[str, list[float]] = {label: [] for label in loaded}
    deltas: dict[str, list[float]] = {
        label: [] for label in loaded if label != args.reference_label
    }
    rng = random.Random(args.seed)
    for _ in range(args.replicates):
        selected = [rng.choice(cluster_names) for _ in cluster_names]
        indices = [index for cluster in selected for index in clusters[cluster]]
        boot = {
            label: sum(values[i] for i in indices) / len(indices)
            for label, values in correctness.items()
        }
        for label, value in boot.items():
            samples[label].append(value)
        for label in deltas:
            deltas[label].append(boot[args.reference_label] - boot[label])

    model_results = {}
    for label in loaded:
        values = samples[label]
        model_results[label] = {
            "correct": int(sum(correctness[label].values())),
            "questions": len(common),
            "accuracy": point[label],
            "cluster_bootstrap_95_ci": [percentile(values, 0.025), percentile(values, 0.975)],
        }
    comparisons = {}
    for label, values in deltas.items():
        observed = point[args.reference_label] - point[label]
        p_value = min(1.0, 2 * min(
            sum(value <= 0 for value in values) / len(values),
            sum(value >= 0 for value in values) / len(values),
        ))
        comparisons[label] = {
            "contrast": f"{args.reference_label} - {label}",
            "accuracy_difference": observed,
            "cluster_bootstrap_95_ci": [percentile(values, 0.025), percentile(values, 0.975)],
            "two_sided_bootstrap_p_value": p_value,
        }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "nonparametric image-cluster bootstrap with paired cluster draws",
        "seed": args.seed,
        "replicates": args.replicates,
        "question_count": len(common),
        "unique_image_cluster_count": len(cluster_names),
        "clusters_with_multiple_questions": sum(len(indices) > 1 for indices in clusters.values()),
        "maximum_questions_per_image": max(map(len, clusters.values())),
        "reference_label": args.reference_label,
        "models": model_results,
        "paired_comparisons": comparisons,
        "manifest": manifest,
    }
    atomic_json(output / "cluster_bootstrap_results.json", result)

    lines = [
        "# PathMMU image-cluster bootstrap results", "",
        f"Questions: {len(common)}; unique image clusters: {len(cluster_names)}; "
        f"bootstrap replicates: {args.replicates}.", "",
        "| Model | Accuracy | Image-cluster bootstrap 95% CI |", "| --- | ---: | ---: |",
    ]
    for label, value in model_results.items():
        low, high = value["cluster_bootstrap_95_ci"]
        lines.append(f"| {label} | {value['accuracy']:.2%} | [{low:.2%}, {high:.2%}] |")
    lines.extend(["", f"Paired differences use `{args.reference_label}` as the reference.", "",
                  "| Comparator | Difference | 95% CI | Two-sided bootstrap p |",
                  "| --- | ---: | ---: | ---: |"])
    for label, value in comparisons.items():
        low, high = value["cluster_bootstrap_95_ci"]
        lines.append(
            f"| {label} | {value['accuracy_difference']:+.2%} | [{low:+.2%}, {high:+.2%}] | "
            f"{value['two_sided_bootstrap_p_value']:.4f} |"
        )
    (output / "cluster_bootstrap_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
