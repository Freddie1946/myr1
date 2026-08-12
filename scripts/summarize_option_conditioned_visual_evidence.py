#!/usr/bin/env python3
"""Write a compact markdown summary for option-conditioned evidence runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def number(value: object, digits: int = 4) -> str:
        return "NA" if value is None else f"{float(value):+.{digits}f}"

    lines = ["# Option-conditioned visual evidence", "", "The candidate is positive only when deletion selectively lowers the same option margin versus controls. Raw attention is reported separately as a non-option-conditioned comparator.", "", "| Model | Cases | Baseline accuracy | Layer | Option | Positive drop | Raw attention drop | Random drop | Positive selectivity | Raw selectivity | Beats random | Retention vs random |", "|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|"]
    for path in args.metrics:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        label = metrics.get("model_label", path.parent.name)
        for layer, options in metrics.get("by_layer_option", {}).items():
            for option, row in options.items():
                lines.append(
                    f"| {label} | {metrics.get('case_count', '')} | {metrics.get('baseline_accuracy', 0):.3f} | {layer} | {option} | "
                    f"{number(row.get('positive_mean_margin_drop'))} | {number(row.get('raw_attention_mean_margin_drop'))} | {number(row.get('random_mean_margin_drop'))} | "
                    f"{number(row.get('positive_selectivity'))} | {number(row.get('raw_attention_selectivity'))} | {number(row.get('positive_beats_random_rate'), 3)} | "
                    f"{number(row.get('positive_retention_margin_vs_random'))} |"
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
