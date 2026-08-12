#!/usr/bin/env python3
"""Causally trace image-specific decision information through decoder layers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from PIL import Image, ImageOps

from run_option_conditioned_visual_evidence import (
    OPTION_LETTERS,
    OptionEvidenceModel,
    canonical_json,
    margins,
    model_prompt,
    parse_layers,
    sha256_file,
)
from run_visual_understanding_counterfactual import bootstrap_ci


def geometry_match(image: Image.Image, reference: Image.Image) -> Image.Image:
    """Center-crop and resize a counterfactual to the exact reference geometry."""
    return ImageOps.fit(
        image.convert("RGB"), reference.size,
        method=Image.Resampling.BICUBIC,
        centering=(0.5, 0.5),
    )


def decoder_hidden(output: Any, layer: int) -> torch.Tensor:
    """Return hidden states from old tensor or current tuple layer outputs."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"decoder layer {layer} returned an unsupported output")


def replace_decoder_hidden(output: Any, replacement: torch.Tensor, layer: int) -> Any:
    """Preserve the decoder layer output container while replacing hidden state."""
    hidden = decoder_hidden(output, layer)
    patched = hidden.clone()
    patched[:, -1, :] = replacement.to(device=hidden.device, dtype=hidden.dtype)
    if isinstance(output, torch.Tensor):
        return patched
    return (patched, *output[1:])


class ActivationPatchingModel(OptionEvidenceModel):
    @property
    def decoder_layers(self):
        backbone = self.model.model
        language_model = getattr(backbone, "language_model", None)
        if language_model is not None and hasattr(language_model, "layers"):
            return language_model.layers
        if hasattr(backbone, "layers"):
            return backbone.layers
        raise AttributeError("cannot locate decoder layers in the loaded Qwen2.5-VL model")

    def prepare(self, prompt: str, image: Image.Image) -> dict[str, torch.Tensor]:
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        return {name: value.to("cuda") for name, value in inputs.items()}

    def probabilities_from_inputs(self, inputs: dict[str, torch.Tensor]) -> list[float]:
        with torch.inference_mode():
            # Transformers 4.49 does not expose the later ``logits_to_keep``
            # optimization. Selecting the final position after the forward is
            # numerically equivalent and keeps the runner version-compatible.
            logits = self.model(**inputs, use_cache=False).logits[0, -1]
            probabilities = torch.softmax(logits.index_select(0, self.option_token_ids).float(), dim=0)
        return probabilities.cpu().tolist()

    def capture(
        self, inputs: dict[str, torch.Tensor], layers: list[int],
    ) -> tuple[list[float], dict[int, torch.Tensor]]:
        captured: dict[int, torch.Tensor] = {}
        handles = []
        for layer in layers:
            def hook(_module, _arguments, output, layer_index=layer):
                hidden = decoder_hidden(output, layer_index)
                captured[layer_index] = hidden[:, -1, :].detach().clone()
            handles.append(self.decoder_layers[layer].register_forward_hook(hook))
        try:
            probabilities = self.probabilities_from_inputs(inputs)
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != set(layers):
            raise RuntimeError("failed to capture every requested decoder layer")
        return probabilities, captured

    def patch(
        self, inputs: dict[str, torch.Tensor], layer: int, replacement: torch.Tensor,
    ) -> list[float]:
        def hook(_module, _arguments, output):
            return replace_decoder_hidden(output, replacement, layer)
        handle = self.decoder_layers[layer].register_forward_hook(hook)
        try:
            return self.probabilities_from_inputs(inputs)
        finally:
            handle.remove()


def condition(probabilities: list[float], target: int) -> dict[str, Any]:
    option_margins = margins(probabilities)
    prediction = int(np.argmax(probabilities))
    return {
        "option_probabilities": dict(zip(OPTION_LETTERS, probabilities)),
        "option_margins": dict(zip(OPTION_LETTERS, option_margins)),
        "predicted_choice": OPTION_LETTERS[prediction],
        "correct": prediction == target,
        "target_probability": float(probabilities[target]),
        "target_margin": float(option_margins[target]),
    }


def evaluate_case(
    model: ActivationPatchingModel,
    case: dict[str, Any],
    pairing: dict[str, Any],
    layers: list[int],
    permutation_count: int,
) -> dict[str, Any]:
    original_path = Path(case["image"])
    mismatch_path = Path(pairing["nearest_same_target"]["image"])
    if sha256_file(original_path) != case["image_sha256"]:
        raise ValueError(f"original image hash mismatch: {case['panel_index']}")
    if sha256_file(mismatch_path) != pairing["nearest_same_target"]["image_sha256"]:
        raise ValueError(f"mismatch image hash mismatch: {case['panel_index']}")
    original_image = Image.open(original_path).convert("RGB")
    mismatch_native = Image.open(mismatch_path).convert("RGB")
    mismatch_image = geometry_match(mismatch_native, original_image)
    prompt = model_prompt(model.processor, case["problem"])
    original_inputs = model.prepare(prompt, original_image)
    mismatch_inputs = model.prepare(prompt, mismatch_image)
    if original_inputs["input_ids"].shape != mismatch_inputs["input_ids"].shape:
        raise ValueError(f"geometry matching did not align sequence lengths: {case['panel_index']}")
    if not torch.equal(original_inputs["image_grid_thw"], mismatch_inputs["image_grid_thw"]):
        raise ValueError(f"geometry matching did not align visual grids: {case['panel_index']}")
    original, original_states = model.capture(original_inputs, layers)
    mismatch, mismatch_states = model.capture(mismatch_inputs, layers)
    original_repeat = model.probabilities_from_inputs(original_inputs)
    mismatch_repeat = model.probabilities_from_inputs(mismatch_inputs)
    target = OPTION_LETTERS.index(case["target_choice"])
    patched: dict[str, Any] = {}
    for layer in layers:
        causal = model.patch(mismatch_inputs, layer, original_states[layer])
        delta = original_states[layer] - mismatch_states[layer]
        anti_state = mismatch_states[layer] - delta
        anti = model.patch(mismatch_inputs, layer, anti_state)
        permuted_controls = []
        for control_index in range(permutation_count):
            seed = int.from_bytes(
                hashlib.sha256(
                    f"activation-permutation-v1:{case['source_record_sha256']}:{layer}:{control_index}".encode()
                ).digest()[:8], "big"
            )
            generator = torch.Generator(device="cpu").manual_seed(seed)
            order = torch.randperm(delta.numel(), generator=generator).to(delta.device)
            permuted_delta = delta.reshape(-1).index_select(0, order).reshape_as(delta)
            permuted = model.patch(mismatch_inputs, layer, mismatch_states[layer] + permuted_delta)
            permuted_controls.append({
                "control_index": control_index,
                "seed": seed,
                "delta_l2": float(torch.linalg.vector_norm(permuted_delta.float()).cpu()),
                "condition": condition(permuted, target),
            })
        patched[str(layer)] = {
            "correct_direction": condition(causal, target),
            "opposite_direction": condition(anti, target),
            "permuted_direction_controls": permuted_controls,
            "activation_delta_l2": float(torch.linalg.vector_norm(
                delta.float()
            ).cpu()),
        }
    result = {
        "panel_index": case["panel_index"],
        "source_index": case["index"],
        "source_record_sha256": case["source_record_sha256"],
        "image": str(original_path),
        "image_sha256": case["image_sha256"],
        "target_choice": case["target_choice"],
        "mismatch": pairing["nearest_same_target"],
        "mismatch_geometry": {
            "native_size": list(mismatch_native.size),
            "matched_size": list(mismatch_image.size),
            "reference_size": list(original_image.size),
            "method": "center_crop_resize_bicubic",
            "sequence_length": int(original_inputs["input_ids"].shape[1]),
            "image_grid_thw": original_inputs["image_grid_thw"][0].cpu().tolist(),
        },
        "original": condition(original, target),
        "mismatch_baseline": condition(mismatch, target),
        "patched_by_layer": patched,
        "original_repeatability_max_abs_probability_diff": float(
            np.max(np.abs(np.asarray(original) - np.asarray(original_repeat)))
        ),
        "mismatch_repeatability_max_abs_probability_diff": float(
            np.max(np.abs(np.asarray(mismatch) - np.asarray(mismatch_repeat)))
        ),
    }
    del original_inputs, mismatch_inputs, original_states, mismatch_states
    gc.collect()
    torch.cuda.empty_cache()
    return result


def aggregate(records: list[dict[str, Any]], layers: list[int]) -> dict[str, Any]:
    by_layer: dict[str, Any] = {}
    for layer in layers:
        key = str(layer)
        baseline_gap: list[float] = []
        recovery: list[float] = []
        opposite_effect: list[float] = []
        directional_contrast: list[float] = []
        selective_recovery: list[float] = []
        permuted_mean_effect: list[float] = []
        correct_minus_permuted: list[float] = []
        final_probability_error: list[float] = []
        for row in records:
            target = OPTION_LETTERS.index(row["target_choice"])
            original = np.asarray([row["original"]["option_margins"][x] for x in OPTION_LETTERS])
            mismatch = np.asarray([row["mismatch_baseline"]["option_margins"][x] for x in OPTION_LETTERS])
            causal = np.asarray([row["patched_by_layer"][key]["correct_direction"]["option_margins"][x] for x in OPTION_LETTERS])
            anti = np.asarray([row["patched_by_layer"][key]["opposite_direction"]["option_margins"][x] for x in OPTION_LETTERS])
            baseline_gap.append(float(original[target] - mismatch[target]))
            recovery.append(float(causal[target] - mismatch[target]))
            opposite_effect.append(float(anti[target] - mismatch[target]))
            directional_contrast.append(float(causal[target] - anti[target]))
            all_recovery = causal - mismatch
            selective_recovery.append(float(all_recovery[target] - np.delete(all_recovery, target).mean()))
            controls = row["patched_by_layer"][key].get("permuted_direction_controls", [])
            if controls:
                effects = [
                    control["condition"]["option_margins"][row["target_choice"]]
                    - mismatch[target]
                    for control in controls
                ]
                permuted_mean_effect.append(float(np.mean(effects)))
                correct_minus_permuted.append(float(causal[target] - mismatch[target] - np.mean(effects)))
            final_probability_error.append(max(abs(
                row["patched_by_layer"][key]["correct_direction"]["option_probabilities"][letter]
                - row["original"]["option_probabilities"][letter]
            ) for letter in OPTION_LETTERS))
        gap_array = np.asarray(baseline_gap)
        recovery_array = np.asarray(recovery)
        by_layer[key] = {
            "mean_original_minus_mismatch_target_margin": float(np.mean(baseline_gap)),
            "mean_correct_patch_target_margin_recovery": float(np.mean(recovery)),
            "correct_patch_recovery_bootstrap_95ci": bootstrap_ci(recovery, f"activation-recovery:{layer}"),
            "mean_opposite_patch_target_margin_effect": float(np.mean(opposite_effect)),
            "mean_correct_minus_opposite_directional_contrast": float(np.mean(directional_contrast)),
            "directional_contrast_bootstrap_95ci": bootstrap_ci(directional_contrast, f"activation-direction:{layer}"),
            "mean_correct_patch_selective_recovery": float(np.mean(selective_recovery)),
            "mean_permuted_direction_target_margin_effect": (
                float(np.mean(permuted_mean_effect)) if permuted_mean_effect else None
            ),
            "mean_correct_minus_permuted_direction": (
                float(np.mean(correct_minus_permuted)) if correct_minus_permuted else None
            ),
            "correct_minus_permuted_bootstrap_95ci": (
                bootstrap_ci(correct_minus_permuted, f"activation-permuted:{layer}")
                if correct_minus_permuted else None
            ),
            "correct_patch_beats_permuted_rate": (
                float(np.mean(np.asarray(correct_minus_permuted) > 0))
                if correct_minus_permuted else None
            ),
            "positive_recovery_rate": float(np.mean(recovery_array > 0)),
            "correct_patch_beats_opposite_rate": float(np.mean(np.asarray(directional_contrast) > 0)),
            "aggregate_recovery_fraction": (
                float(np.sum(recovery_array) / np.sum(gap_array))
                if abs(float(np.sum(gap_array))) > 1e-12 else None
            ),
            "max_correct_patch_vs_original_probability_error": float(np.max(final_probability_error)),
        }
    return {
        "case_count": len(records),
        "original_accuracy": float(np.mean([row["original"]["correct"] for row in records])),
        "geometry_matched_mismatch_accuracy": float(np.mean([
            row["mismatch_baseline"]["correct"] for row in records
        ])),
        "by_layer": by_layer,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--expected-panel-sha256", required=True)
    parser.add_argument("--counterfactual-manifest", type=Path, required=True)
    parser.add_argument("--expected-counterfactual-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", default="0,4,8,12,16,20,24,27")
    parser.add_argument("--panel-index", type=int, action="append")
    parser.add_argument("--permutation-count", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run: {args.output_dir}")
    panel_sha = sha256_file(args.panel)
    counterfactual_sha = sha256_file(args.counterfactual_manifest)
    if panel_sha != args.expected_panel_sha256 or counterfactual_sha != args.expected_counterfactual_sha256:
        raise ValueError("frozen input SHA-256 mismatch")
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    manifest = json.loads(args.counterfactual_manifest.read_text(encoding="utf-8"))
    if panel.get("case_count") != 96 or manifest.get("panel_sha256") != panel_sha:
        raise ValueError("requires the frozen 96-case panel and its paired manifest")
    selected = set(args.panel_index or range(96))
    cases = [row for row in panel["cases"] if row["panel_index"] in selected]
    pairings = {row["panel_index"]: row for row in manifest["pairings"]}
    if len(cases) != len(selected):
        raise ValueError("requested panel index missing")
    if args.permutation_count < 0 or args.permutation_count > 16:
        raise ValueError("permutation-count must be in [0, 16]")
    args.output_dir.mkdir(parents=True)
    model = ActivationPatchingModel(args.model, batch_size=1)
    layers = parse_layers(args.layers, len(model.decoder_layers))
    records: list[dict[str, Any]] = []
    path = args.output_dir / "case_results.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            row = evaluate_case(
                model, case, pairings[case["panel_index"]], layers,
                args.permutation_count,
            )
            records.append(row)
            handle.write(canonical_json(row) + "\n")
            handle.flush()
            print(canonical_json({
                "completed": len(records), "total": len(cases),
                "panel_index": case["panel_index"],
            }), flush=True)
    metrics = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_confirmation" if len(cases) == 96 else "completed_smoke",
        "formal_result": False,
        "method": (
            "geometry_matched_layerwise_residual_activation_patching_with_permuted_direction_controls_v2"
            if args.permutation_count else
            "geometry_matched_same_answer_position_layerwise_residual_activation_patching_v1"
        ),
        "claim_boundary": "causally traces image-specific decision information, not spatial localization",
        "model_label": args.model_label,
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "panel_path": str(args.panel.resolve()),
        "panel_sha256": panel_sha,
        "counterfactual_manifest_path": str(args.counterfactual_manifest.resolve()),
        "counterfactual_manifest_sha256": counterfactual_sha,
        "layers_evaluated": layers,
        "permutation_control_count": args.permutation_count,
        "score_batch_size": 1,
        "software_environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "attention_implementation": "eager",
        },
        "max_original_repeatability_abs_probability_diff": max(
            row["original_repeatability_max_abs_probability_diff"] for row in records
        ),
        "max_mismatch_repeatability_abs_probability_diff": max(
            row["mismatch_repeatability_max_abs_probability_diff"] for row in records
        ),
        "case_results": str(path.resolve()),
        "case_results_sha256": sha256_file(path),
        **aggregate(records, layers),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(canonical_json(metrics), flush=True)


if __name__ == "__main__":
    main()
