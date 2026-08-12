#!/usr/bin/env python3
"""Freeze model-blind, same-domain image counterfactual pairings."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from run_option_conditioned_visual_evidence import canonical_json, sha256_file


def image_descriptor(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        pixels = np.asarray(image.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float64) / 255.0
        features: list[float] = []
        for channel in range(3):
            values = pixels[..., channel]
            features.extend([float(values.mean()), float(values.std())])
            histogram, _ = np.histogram(values, bins=8, range=(0.0, 1.0), density=False)
            features.extend((histogram / histogram.sum()).tolist())
        features.append(float(np.log(max(image.width / image.height, 1e-8))))
    return np.asarray(features, dtype=np.float64)


def nearest_pairings(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    descriptors = np.stack([image_descriptor(Path(row["image"])) for row in cases])
    scale = descriptors.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (descriptors - descriptors.mean(axis=0)) / scale
    distances = np.linalg.norm(standardized[:, None, :] - standardized[None, :, :], axis=-1)
    output = []
    for index, row in enumerate(cases):
        candidates_same = [i for i, other in enumerate(cases) if i != index and other["target_choice"] == row["target_choice"]]
        candidates_different = [i for i, other in enumerate(cases) if other["target_choice"] != row["target_choice"]]
        same = min(candidates_same, key=lambda i: (distances[index, i], cases[i]["source_record_sha256"]))
        different = min(candidates_different, key=lambda i: (distances[index, i], cases[i]["source_record_sha256"]))
        output.append({
            "panel_index": row["panel_index"],
            "source_record_sha256": row["source_record_sha256"],
            "target_choice": row["target_choice"],
            "patch_shuffle_seed": int(hashlib.sha256(f"visual-understanding-v1:{row['source_record_sha256']}".encode()).hexdigest()[:16], 16),
            "nearest_same_target": {
                "panel_index": cases[same]["panel_index"], "source_record_sha256": cases[same]["source_record_sha256"],
                "target_choice": cases[same]["target_choice"], "image": cases[same]["image"],
                "image_sha256": cases[same]["image_sha256"], "standardized_descriptor_distance": float(distances[index, same]),
            },
            "nearest_different_target": {
                "panel_index": cases[different]["panel_index"], "source_record_sha256": cases[different]["source_record_sha256"],
                "target_choice": cases[different]["target_choice"], "image": cases[different]["image"],
                "image_sha256": cases[different]["image_sha256"], "standardized_descriptor_distance": float(distances[index, different]),
            },
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--expected-panel-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen counterfactual manifest: {args.output}")
    panel_sha = sha256_file(args.panel)
    if panel_sha != args.expected_panel_sha256:
        raise ValueError(f"panel SHA-256 mismatch: {panel_sha}")
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    if panel.get("case_count") != 96 or panel.get("selection_uses_model_outputs") is not False:
        raise ValueError("counterfactual confirmation panel must be frozen, model-blind, and contain 96 cases")
    pairings = nearest_pairings(panel["cases"])
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_model_counterfactual_outputs",
        "selection_uses_model_outputs": False,
        "panel_path": str(args.panel.resolve()), "panel_sha256": panel_sha,
        "case_count": len(pairings),
        "descriptor": "64x64_RGB_mean_std_8bin_histogram_per_channel_plus_log_aspect_zscored_within_panel",
        "matching": "nearest_descriptor_distance_with_source_sha256_tie_break",
        "pairings": pairings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(canonical_json({"output": str(args.output.resolve()), "case_count": len(pairings), "sha256": sha256_file(args.output)}))


if __name__ == "__main__":
    main()
