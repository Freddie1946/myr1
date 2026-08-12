#!/usr/bin/env python3
"""Integrated-gradients visual evidence for a Qwen2.5-VL decision margin."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from run_attention_intervention_experiment import DecisionScorer
from run_rise_visual_evidence import (
    FRACTIONS,
    LETTERS,
    case_seed,
    curve_record,
    exact_area_mask,
    intervene,
    shifted_controls,
)
from run_visual_fidelity_experiment import canonical_json, model_prompt, sha256_file


def option_margin(logits: torch.Tensor, decision: int) -> torch.Tensor:
    alternatives = torch.cat((logits[:decision], logits[decision + 1 :]))
    return logits[decision] - torch.logsumexp(alternatives, dim=0)


def reconstruct_patch_attribution(
    attribution: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
    merge: int,
    patch: int,
    temporal: int,
) -> np.ndarray:
    """Invert Qwen2-VL image-processor patch flattening into a spatial map."""
    llm_h, llm_w = grid_h // merge, grid_w // merge
    channels = attribution.shape[1] // (temporal * patch * patch)
    values = attribution.reshape(
        1, llm_h, llm_w, merge, merge, channels, temporal, patch, patch
    )
    values = values.sum(dim=(0, 5, 6))
    # llm_h,llm_w,merge_h,merge_w,patch_h,patch_w -> H,W
    values = values.permute(0, 2, 4, 1, 3, 5).reshape(grid_h * patch, grid_w * patch)
    return values.detach().float().cpu().numpy()


class IGExtractor:
    def __init__(self, model_path: Path):
        self.model_path = model_path.resolve()
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True, use_fast=False)
        self.processor.image_processor.max_pixels = 65536
        self.processor.image_processor.min_pixels = 3136
        option_ids = [self.processor.tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
        if any(len(ids) != 1 for ids in option_ids):
            raise ValueError(f"A-D are not single tokens: {option_ids}")
        self.option_ids = torch.tensor([row[0] for row in option_ids], device="cuda")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to("cuda").eval()

    def extract(self, problem: str, image: Image.Image, *, steps: int) -> dict[str, Any]:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        mean_rgb = tuple(int(round(float(value))) for value in pixels.mean(axis=(0, 1)))
        baseline_image = Image.new("RGB", image.size, mean_rgb)
        prompt = model_prompt(self.processor, problem)
        source = self.processor(text=[prompt], images=[image], return_tensors="pt")
        baseline = self.processor(text=[prompt], images=[baseline_image], return_tensors="pt")
        if not torch.equal(source["image_grid_thw"], baseline["image_grid_thw"]):
            raise ValueError("source and IG baseline visual grids differ")
        fixed = {
            key: value.to("cuda")
            for key, value in source.items()
            if key != "pixel_values"
        }
        source_pixels = source["pixel_values"].to("cuda", dtype=self.model.visual.dtype)
        baseline_pixels = baseline["pixel_values"].to("cuda", dtype=self.model.visual.dtype)
        delta = source_pixels - baseline_pixels
        with torch.no_grad():
            raw = self.model(pixel_values=source_pixels, **fixed, use_cache=False).logits[0, -1, :]
            options = raw.index_select(0, self.option_ids).float()
            decision = int(torch.argmax(options))
            source_margin = float(option_margin(options, decision).cpu())
            base_raw = self.model(pixel_values=baseline_pixels, **fixed, use_cache=False).logits[0, -1, :]
            base_margin = float(option_margin(base_raw.index_select(0, self.option_ids).float(), decision).cpu())
        gradient_sum = torch.zeros_like(source_pixels, dtype=torch.float32)
        # Midpoint Riemann rule avoids endpoint-specific gradients.
        for step in range(steps):
            alpha = (step + 0.5) / steps
            current = (baseline_pixels + alpha * delta).detach().requires_grad_(True)
            logits = self.model(pixel_values=current, **fixed, use_cache=False).logits[0, -1, :]
            margin = option_margin(logits.index_select(0, self.option_ids).float(), decision)
            gradient = torch.autograd.grad(margin, current, retain_graph=False, create_graph=False)[0]
            gradient_sum += gradient.float()
        attribution = delta.float() * gradient_sum / steps
        grid_t, grid_h, grid_w = map(int, source["image_grid_thw"][0])
        processor = self.processor.image_processor
        spatial = reconstruct_patch_attribution(
            attribution,
            grid_h=grid_h,
            grid_w=grid_w,
            merge=int(processor.merge_size),
            patch=int(processor.patch_size),
            temporal=int(processor.temporal_patch_size),
        )
        positive = np.maximum(spatial, 0)
        negative = np.maximum(-spatial, 0)
        signed_sum = float(attribution.sum().detach().cpu())
        completeness_delta = (source_margin - base_margin) - signed_sum
        return {
            "decision_index": decision,
            "source_option_probabilities": torch.softmax(options, dim=0).cpu().tolist(),
            "source_margin": source_margin,
            "baseline_margin": base_margin,
            "signed_attribution_sum": signed_sum,
            "completeness_residual": completeness_delta,
            "processed_height": spatial.shape[0],
            "processed_width": spatial.shape[1],
            "positive_evidence": positive.astype(np.float32),
            "negative_evidence": negative.astype(np.float32),
            "signed_evidence": spatial.astype(np.float32),
            "mean_baseline_rgb": list(mean_rgb),
        }

    def close(self) -> None:
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.min()) / max(float(values.max() - values.min()), 1e-12)


def evaluate_interventions(
    scorer: DecisionScorer,
    case: dict[str, Any],
    extraction: dict[str, Any],
    *,
    random_controls: int,
) -> dict[str, Any]:
    image = Image.open(case["image"]).convert("RGB")
    source = np.asarray(image, dtype=np.float32)
    mean_rgb = np.mean(source, axis=(0, 1))
    mean = np.broadcast_to(mean_rgb, source.shape).astype(np.float32)
    prompt = model_prompt(scorer.processor, case["problem"])
    baseline = scorer.score(prompt, [image])[0]
    decision = int(np.argmax(baseline))
    if decision != extraction["decision_index"]:
        raise ValueError("IG and intervention scorer baseline decisions differ")
    target = LETTERS.index(case["target_choice"])
    evidence = np.asarray(extraction["positive_evidence"], dtype=np.float32)
    # Resize attribution without smoothing beyond the input-patch reconstruction.
    evidence = np.asarray(
        Image.fromarray(evidence, mode="F").resize(image.size, Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    curves: dict[str, Any] = {}
    for mode in ("deletion", "retention"):
        curves[mode] = {}
        for name, high in (("ig_positive_high", True), ("ig_positive_low", False)):
            masks = [exact_area_mask(evidence, fraction, high=high) for fraction in FRACTIONS]
            probabilities = scorer.score(prompt, [intervene(source, mean, mask, mode) for mask in masks])
            curves[mode][name] = curve_record(probabilities, decision, target)
        random_rows = []
        for control in range(random_controls):
            images = []
            for fraction in FRACTIONS:
                mask = exact_area_mask(evidence, fraction, high=True)
                shifted = shifted_controls(
                    mask,
                    random_controls,
                    case_seed(case["source_record_sha256"], f"ig:{fraction}"),
                )[control]
                images.append(intervene(source, mean, shifted, mode))
            random_rows.append(curve_record(scorer.score(prompt, images), decision, target))
        curves[mode]["random_shifted_shape_matched"] = random_rows
    return {
        "panel_index": case["panel_index"],
        "source_index": case["index"],
        "image": case["image"],
        "image_sha256": case["image_sha256"],
        "target_choice": case["target_choice"],
        "baseline_option_probabilities": dict(zip(LETTERS, baseline)),
        "baseline_predicted_choice": LETTERS[decision],
        "baseline_correct": decision == target,
        "source_margin": extraction["source_margin"],
        "ig_baseline_margin": extraction["baseline_margin"],
        "ig_signed_attribution_sum": extraction["signed_attribution_sum"],
        "ig_completeness_residual": extraction["completeness_residual"],
        "processed_height": extraction["processed_height"],
        "processed_width": extraction["processed_width"],
        "positive_evidence": extraction["positive_evidence"].tolist(),
        "negative_evidence": extraction["negative_evidence"].tolist(),
        "fractions": list(FRACTIONS),
        "curves": curves,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    output = {
        "case_count": len(records),
        "baseline_accuracy": float(np.mean([row["baseline_correct"] for row in records])),
        "mean_absolute_completeness_residual": float(np.mean([abs(row["ig_completeness_residual"]) for row in records])),
        "strategy_fidelity": {},
    }
    for mode in ("deletion", "retention"):
        random_auc = [
            float(np.mean([value["auc"] for value in row["curves"][mode]["random_shifted_shape_matched"]]))
            for row in records
        ]
        output["strategy_fidelity"][mode] = {}
        for strategy in ("ig_positive_high", "ig_positive_low"):
            observed = [row["curves"][mode][strategy]["auc"] for row in records]
            advantage = [random - value if mode == "deletion" else value - random for value, random in zip(observed, random_auc)]
            output["strategy_fidelity"][mode][strategy] = {
                "mean_auc": float(np.mean(observed)),
                "mean_random_auc": float(np.mean(random_auc)),
                "mean_fidelity_advantage": float(np.mean(advantage)),
                "sample_std_fidelity_advantage": float(np.std(advantage, ddof=1)) if len(advantage) > 1 else 0.0,
                "positive_cases": int(sum(value > 0 for value in advantage)),
            }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--expected-panel-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-index", type=int, action="append")
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--random-controls", type=int, default=5)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run: {args.output_dir}")
    if args.ig_steps <= 0:
        raise ValueError("IG steps must be positive")
    panel_sha = sha256_file(args.panel)
    if panel_sha != args.expected_panel_sha256:
        raise ValueError("panel hash mismatch")
    panel = json.loads(args.panel.read_text())
    if panel.get("case_count") != 24 or panel.get("selection_uses_model_outputs") is not False:
        raise ValueError("panel contract mismatch")
    selected = set(args.panel_index or range(24))
    cases = [row for row in panel["cases"] if row["panel_index"] in selected]
    args.output_dir.mkdir(parents=True)
    extraction_path = args.output_dir / "ig_maps.jsonl"
    extraction_rows = []
    extractor = IGExtractor(args.model)
    with extraction_path.open("w") as handle:
        for case in cases:
            image = Image.open(case["image"]).convert("RGB")
            values = extractor.extract(case["problem"], image, steps=args.ig_steps)
            serializable = {
                **{key: value for key, value in values.items() if not isinstance(value, np.ndarray)},
                "panel_index": case["panel_index"],
                "positive_evidence": values["positive_evidence"].tolist(),
                "negative_evidence": values["negative_evidence"].tolist(),
            }
            extraction_rows.append(values)
            handle.write(canonical_json(serializable) + "\n")
            handle.flush()
            print(canonical_json({"phase": "ig", "completed": len(extraction_rows), "total": len(cases)}), flush=True)
    extractor.close()
    scorer = DecisionScorer(args.model, args.batch_size)
    result_path = args.output_dir / "case_results.jsonl"
    records = []
    with result_path.open("w") as handle:
        for case, extraction in zip(cases, extraction_rows):
            row = evaluate_interventions(scorer, case, extraction, random_controls=args.random_controls)
            records.append(row)
            handle.write(canonical_json(row) + "\n")
            handle.flush()
            print(canonical_json({"phase": "intervention", "completed": len(records), "total": len(cases)}), flush=True)
    metrics = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_exploratory" if len(cases) == 24 else "completed_smoke",
        "formal_result": False,
        "method": "integrated_gradients_decision_margin_mean_baseline_v1",
        "ig_steps": args.ig_steps,
        "attribution_target": "original_prediction_logit_minus_logsumexp_other_option_logits",
        "ig_baseline": "per_image_global_mean_rgb",
        "validation_perturbation": "hard_mean_fill",
        "model_label": args.model_label,
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "panel_path": str(args.panel.resolve()),
        "panel_sha256": panel_sha,
        "panel_indices": [row["panel_index"] for row in cases],
        "random_controls": args.random_controls,
        "ig_maps": str(extraction_path.resolve()),
        "ig_maps_sha256": sha256_file(extraction_path),
        "case_results": str(result_path.resolve()),
        "case_results_sha256": sha256_file(result_path),
        **aggregate(records),
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(canonical_json(metrics))


if __name__ == "__main__":
    main()
