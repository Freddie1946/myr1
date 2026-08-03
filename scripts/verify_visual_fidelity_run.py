#!/usr/bin/env python3
"""Verify one completed frozen-panel visual-fidelity run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_METHOD = "patch_occlusion_target_option_probability_fidelity_v1"
EXPECTED_CASE_COUNT = 24


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_run(
    metrics_path: Path,
    *,
    expected_label: str,
    expected_model: Path,
    expected_panel_sha256: str,
) -> dict[str, Any]:
    metrics_path = metrics_path.resolve()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected = {
        "status": "completed",
        "method": EXPECTED_METHOD,
        "attention_claim": False,
        "selection_uses_model_outputs": False,
        "model_label": expected_label,
        "panel_sha256": expected_panel_sha256,
        "case_count": EXPECTED_CASE_COUNT,
        "grid_rows": 6,
        "grid_columns": 6,
        "random_permutations": 5,
    }
    mismatch = {
        key: {"expected": value, "actual": metrics.get(key)}
        for key, value in expected.items()
        if metrics.get(key) != value
    }
    if mismatch:
        raise ValueError(f"visual-fidelity contract mismatch: {mismatch}")

    model = expected_model.resolve()
    if Path(metrics.get("model_path", "")).resolve() != model:
        raise ValueError("visual-fidelity model path mismatch")
    config = model / "config.json"
    if not config.is_file() or sha256_file(config) != metrics.get("model_config_sha256"):
        raise ValueError("visual-fidelity model config hash mismatch")

    results_path = Path(metrics.get("case_results", "")).resolve()
    if results_path.parent != metrics_path.parent or not results_path.is_file():
        raise ValueError("case-results path is missing or outside the run directory")
    if sha256_file(results_path) != metrics.get("case_results_sha256"):
        raise ValueError("case-results SHA-256 mismatch")
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != EXPECTED_CASE_COUNT:
        raise ValueError(f"expected {EXPECTED_CASE_COUNT} case rows, found {len(rows)}")
    if [row.get("panel_index") for row in rows] != list(range(EXPECTED_CASE_COUNT)):
        raise ValueError("visual-fidelity panel indices are incomplete or out of order")
    if len({row.get("source_record_sha256") for row in rows}) != EXPECTED_CASE_COUNT:
        raise ValueError("visual-fidelity source records are not unique")
    if any(len(row.get("patch_importance_log_target_probability_drop", [])) != 36 for row in rows):
        raise ValueError("visual-fidelity patch-importance vector length mismatch")

    figures = metrics_path.parent / "figures"
    expected_figures = {
        *(f"case_{index:02d}_heatmap.png" for index in range(EXPECTED_CASE_COUNT)),
        "aggregate_deletion_insertion_curves.png",
    }
    actual_figures = {path.name for path in figures.glob("*.png") if path.is_file()}
    if actual_figures != expected_figures:
        raise ValueError("visual-fidelity figure inventory mismatch")
    if any((figures / name).stat().st_size <= 0 for name in expected_figures):
        raise ValueError("visual-fidelity figure is empty")

    return {
        "status": "verified",
        "model_label": expected_label,
        "model_path": str(model),
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "case_results_sha256": metrics["case_results_sha256"],
        "case_count": EXPECTED_CASE_COUNT,
        "figure_count": len(expected_figures),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--expected-label", required=True)
    parser.add_argument("--expected-model", type=Path, required=True)
    parser.add_argument("--expected-panel-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = verify_run(
            args.metrics,
            expected_label=args.expected_label,
            expected_model=args.expected_model,
            expected_panel_sha256=args.expected_panel_sha256,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"visual-fidelity verification failed: {exc}") from exc
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
