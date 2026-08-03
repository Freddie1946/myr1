#!/usr/bin/env python3
"""Build the five-arm comparison package for the frozen visual-fidelity experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from verify_visual_fidelity_run import verify_run


ARM_ORDER = (
    "sft3000",
    "sft4000",
    "stage2_outcome_grpo",
    "stage3_gpt4o",
    "stage3_kimi26",
)
DISPLAY_LABELS = {
    "sft3000": "SFT3000",
    "sft4000": "SFT4000",
    "stage2_outcome_grpo": "Stage2",
    "stage3_gpt4o": "Stage3 GPT-4o",
    "stage3_kimi26": "Stage3 Kimi 2.6",
}
METRICS = (
    "baseline_accuracy",
    "mean_deletion_auc_advantage",
    "mean_insertion_auc_advantage",
    "mean_top_25pct_comprehensiveness",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_arm_labels(labels: list[str]) -> None:
    if len(labels) != len(set(labels)):
        raise ValueError("visual-fidelity arm labels are duplicated")
    if set(labels) != set(ARM_ORDER):
        raise ValueError(f"expected exactly these arms: {', '.join(ARM_ORDER)}")


def write_figure(rows: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    titles = (
        "Fixed-panel answer accuracy",
        "Deletion AUC advantage over random",
        "Insertion AUC advantage over random",
        "Top-25% comprehensiveness",
    )
    colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2")
    labels = [DISPLAY_LABELS[row["model_label"]] for row in rows]
    for axis, metric, title in zip(axes.flat, METRICS, titles):
        values = [float(row[metric]) for row in rows]
        axis.bar(range(len(rows)), values, color=colors)
        axis.axhline(0.0, color="#444444", linewidth=0.8)
        axis.set_title(title)
        axis.set_xticks(range(len(rows)), labels, rotation=24, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", nargs=3, action="append", metavar=("LABEL", "MODEL", "METRICS"), required=True)
    parser.add_argument("--expected-panel-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite visual-fidelity comparison: {args.output_dir}")
    validate_arm_labels([values[0] for values in args.arm])

    rows_by_label: dict[str, dict[str, Any]] = {}
    verification: dict[str, dict[str, Any]] = {}
    for label, model_text, metrics_text in args.arm:
        model = Path(model_text)
        metrics_path = Path(metrics_text)
        verification[label] = verify_run(
            metrics_path,
            expected_label=label,
            expected_model=model,
            expected_panel_sha256=args.expected_panel_sha256,
        )
        value = json.loads(metrics_path.read_text(encoding="utf-8"))
        missing = [name for name in METRICS if not isinstance(value.get(name), (int, float))]
        if missing:
            raise ValueError(f"{label} is missing comparison metrics: {missing}")
        rows_by_label[label] = {
            "model_label": label,
            "model_path": str(model.resolve()),
            **{name: float(value[name]) for name in METRICS},
            "baseline_correct": int(value["baseline_correct"]),
            "case_count": int(value["case_count"]),
        }
    rows = [rows_by_label[label] for label in ARM_ORDER]

    args.output_dir.mkdir(parents=True)
    figures = args.output_dir / "figures"
    figures.mkdir()
    figure_path = figures / "five_arm_fidelity_comparison.png"
    write_figure(rows, figure_path)

    csv_path = args.output_dir / "five_arm_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = args.output_dir / "five_arm_metrics.md"
    lines = [
        "# Frozen-panel perturbation-fidelity comparison",
        "",
        "This table reports post-hoc perturbation fidelity, not attention, causal localization, or lesion IoU.",
        "",
        "| Arm | Panel accuracy | Deletion AUC advantage | Insertion AUC advantage | Top-25% comprehensiveness |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {DISPLAY_LABELS[row['model_label']]} | {row['baseline_accuracy']:.4f} | "
            f"{row['mean_deletion_auc_advantage']:.6f} | "
            f"{row['mean_insertion_auc_advantage']:.6f} | "
            f"{row['mean_top_25pct_comprehensiveness']:.6f} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    comparison_path = args.output_dir / "comparison.json"
    comparison = {
        "schema_version": 1,
        "status": "completed",
        "claim": "post_hoc_patch_occlusion_fidelity_not_attention_or_causal_localization",
        "panel_sha256": args.expected_panel_sha256,
        "arms": rows,
        "verification": verification,
    }
    comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifacts = [comparison_path, csv_path, markdown_path, figure_path]
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "artifacts": [
            {
                "path": str(path.relative_to(args.output_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifacts
        ],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
