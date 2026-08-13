#!/usr/bin/env python3
"""Dual-stream causal tracing after predefined pathology-evidence deletion.

For each decoder layer this runner separately restores (1) all visual-token
states and (2) the final query/decision-position state from the clean run into
the evidence-deleted run.  The former measures remaining opportunity for
visual information to propagate through later attention layers; the latter
measures whether the decision residual stream already carries sufficient clean
information.  Their layerwise curves must be interpreted jointly.
"""
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

from run_layerwise_visual_activation_patching import (
    ActivationPatchingModel,
    OPTION_LETTERS,
    condition,
    decoder_hidden,
    sha256_file,
)
from run_option_conditioned_visual_evidence import margins, model_prompt
from run_reference_evidence_behavior import generate, parsed, reasoning_prompt
from run_reference_evidence_layerwise_patching import (
    delete_region,
    mask_from_boxes,
    visual_positions,
)


def capture_states(
    model: ActivationPatchingModel,
    inputs: dict[str, torch.Tensor],
    layers: list[int],
    positions: torch.Tensor,
) -> tuple[list[float], dict[int, dict[str, torch.Tensor]]]:
    captured: dict[int, dict[str, torch.Tensor]] = {}
    handles = []
    for layer in layers:
        def hook(_module, _args, output, layer_index=layer):
            hidden = decoder_hidden(output, layer_index)
            captured[layer_index] = {
                "visual": hidden[:, positions, :].detach().clone(),
                "query": hidden[:, -1:, :].detach().clone(),
            }
        handles.append(model.decoder_layers[layer].register_forward_hook(hook))
    try:
        probabilities = model.probabilities_from_inputs(inputs)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(layers):
        raise RuntimeError("failed to capture requested visual/query states")
    return probabilities, captured


def patch_positions(
    model: ActivationPatchingModel,
    inputs: dict[str, torch.Tensor],
    layer: int,
    positions: torch.Tensor,
    replacement: torch.Tensor,
) -> list[float]:
    def hook(_module, _args, output):
        hidden = decoder_hidden(output, layer)
        patched = hidden.clone()
        patched[:, positions, :] = replacement.to(hidden.device, hidden.dtype)
        if isinstance(output, torch.Tensor):
            return patched
        return (patched, *output[1:])
    handle = model.decoder_layers[layer].register_forward_hook(hook)
    try:
        return model.probabilities_from_inputs(inputs)
    finally:
        handle.remove()


def evaluate_case(model, case: dict[str, Any], layers: list[int]) -> dict[str, Any]:
    path = Path(case["image"])
    if sha256_file(path) != case["image_sha256"]:
        raise ValueError("image hash mismatch")
    image = Image.open(path).convert("RGB")
    deleted = delete_region(image, mask_from_boxes(case, image.size))
    prompt = model_prompt(model.processor, case["problem"])
    clean_inputs = model.prepare(prompt, image)
    deleted_inputs = model.prepare(prompt, deleted)
    if clean_inputs["input_ids"].shape != deleted_inputs["input_ids"].shape:
        raise ValueError("deletion changed token geometry")
    if not torch.equal(clean_inputs["image_grid_thw"], deleted_inputs["image_grid_thw"]):
        raise ValueError("deletion changed image grid")
    visual = visual_positions(clean_inputs, model.image_token_id)
    if not torch.equal(visual, visual_positions(deleted_inputs, model.image_token_id)):
        raise ValueError("visual positions differ")
    query = torch.tensor([clean_inputs["input_ids"].shape[1] - 1], device="cuda")
    clean, clean_states = capture_states(model, clean_inputs, layers, visual)
    corrupted, corrupted_states = capture_states(model, deleted_inputs, layers, visual)
    target = OPTION_LETTERS.index(case["target_choice"])
    clean_margin = float(margins(clean)[target])
    corrupted_margin = float(margins(corrupted)[target])
    gap = clean_margin - corrupted_margin
    patched: dict[str, Any] = {}
    for layer in layers:
        visual_score = patch_positions(
            model, deleted_inputs, layer, visual, clean_states[layer]["visual"]
        )
        query_score = patch_positions(
            model, deleted_inputs, layer, query, clean_states[layer]["query"]
        )
        visual_recovery = float(margins(visual_score)[target] - corrupted_margin)
        query_recovery = float(margins(query_score)[target] - corrupted_margin)
        patched[str(layer)] = {
            "visual_token_patch": {
                "condition": condition(visual_score, target),
                "raw_margin_recovery": visual_recovery,
                "recovery_fraction": visual_recovery / gap if gap > 1e-4 else None,
                "delta_l2": float(torch.linalg.vector_norm(
                    (clean_states[layer]["visual"] - corrupted_states[layer]["visual"]).float()
                ).cpu()),
            },
            "query_position_patch": {
                "condition": condition(query_score, target),
                "raw_margin_recovery": query_recovery,
                "recovery_fraction": query_recovery / gap if gap > 1e-4 else None,
                "delta_l2": float(torch.linalg.vector_norm(
                    (clean_states[layer]["query"] - corrupted_states[layer]["query"]).float()
                ).cpu()),
            },
        }
    result = {
        "panel_index": case["panel_index"],
        "source_index": case["index"],
        "source_record_sha256": case["source_record_sha256"],
        "image_sha256": case["image_sha256"],
        "target_choice": case["target_choice"],
        "clean": condition(clean, target),
        "evidence_deleted": condition(corrupted, target),
        "clean_deleted_target_margin_gap": gap,
        "patched_by_layer": patched,
    }
    del clean_inputs, deleted_inputs, clean_states, corrupted_states
    gc.collect()
    torch.cuda.empty_cache()
    return result


def summarize(records: list[dict[str, Any]], layers: list[int], stream: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for layer in layers:
        values = [r["patched_by_layer"][str(layer)][stream] for r in records]
        raw = [float(x["raw_margin_recovery"]) for x in values]
        fractions = [float(x["recovery_fraction"]) for x in values if x["recovery_fraction"] is not None]
        result[str(layer)] = {
            "case_count": len(values),
            "positive_deletion_gap_case_count": len(fractions),
            "mean_raw_margin_recovery": float(np.mean(raw)) if raw else None,
            "median_raw_margin_recovery": float(np.median(raw)) if raw else None,
            "mean_recovery_fraction_positive_gap_only": float(np.mean(fractions)) if fractions else None,
            "median_recovery_fraction_positive_gap_only": float(np.median(fractions)) if fractions else None,
            "mean_state_delta_l2": float(np.mean([x["delta_l2"] for x in values])) if values else None,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", default="0,4,8,12,16,20,24,27")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    args.output_dir.mkdir(parents=True)
    model = ActivationPatchingModel(args.model, batch_size=1)
    selected = []
    with (args.output_dir / "candidate_filter.jsonl").open("w") as handle:
        for case in panel["cases"]:
            image = Image.open(case["image"]).convert("RGB")
            scores = model.probabilities_from_inputs(
                model.prepare(model_prompt(model.processor, case["problem"]), image)
            )
            forced = OPTION_LETTERS[int(np.argmax(scores))]
            completion = generate(
                model, reasoning_prompt(model.processor, case["problem"]), image
            )
            reasoning = parsed(completion)
            row = {
                "panel_index": case["panel_index"], "target_choice": case["target_choice"],
                "forced_choice": forced, "reasoning_choice": reasoning,
                "reasoning_completion": completion,
                "reasoning_clean_correct": reasoning == case["target_choice"],
                "interface_agreement": reasoning == forced,
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            if row["reasoning_clean_correct"] and row["interface_agreement"]:
                selected.append(case)
    records = []
    with (args.output_dir / "case_results.jsonl").open("w") as handle:
        for case in selected:
            row = evaluate_case(model, case, layers)
            records.append(row)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(json.dumps({"done": len(records), "total": len(selected)}), flush=True)
    metrics = {
        "schema_version": 1,
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model.resolve()),
        "panel": str(args.panel.resolve()),
        "panel_sha256": sha256_file(args.panel),
        "candidate_count": len(panel["cases"]),
        "primary_reasoning_clean_correct_and_interface_agreement_count": len(selected),
        "layers": layers,
        "visual_token_patch_by_layer": summarize(records, layers, "visual_token_patch"),
        "query_position_patch_by_layer": summarize(records, layers, "query_position_patch"),
        "interpretation_boundary": (
            "visual-token late-layer recovery is structurally opportunity-limited; query-position "
            "recovery and visual-token recovery must be interpreted jointly; external boxes are "
            "reference regions pending blinded pathology-expert verification"
        ),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
