#!/usr/bin/env python3
"""Layerwise visual-token activation patching for predefined evidence deletion.

The panel is frozen before target outputs.  We first score all candidates with
the target checkpoint, then run the mechanistic intervention only on clean
correct cases.  A clean visual-token state is patched into an otherwise
identical evidence-deleted image at selected decoder layers.  This is a
causal-contribution probe, not a claim that the external boxes are ground
truth causal regions.
"""
from __future__ import annotations

import argparse, gc, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter

from run_layerwise_visual_activation_patching import (
    ActivationPatchingModel, OPTION_LETTERS, condition, decoder_hidden,
    sha256_file,
)
from run_option_conditioned_visual_evidence import margins, model_prompt
from run_reference_evidence_behavior import generate, parsed, reasoning_prompt


def mask_from_boxes(case: dict[str, Any], size: tuple[int, int]) -> np.ndarray:
    w, h = size
    mask = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in case["external_annotation"]["boxes"]:
        left = max(0, min(w - 1, round(float(x0) * w / 1000)))
        top = max(0, min(h - 1, round(float(y0) * h / 1000)))
        right = max(left + 1, min(w, round(float(x1) * w / 1000)))
        bottom = max(top + 1, min(h, round(float(y1) * h / 1000)))
        mask[top:bottom, left:right] = True
    return mask


def delete_region(image: Image.Image, mask: np.ndarray) -> Image.Image:
    base = image.filter(ImageFilter.GaussianBlur(radius=max(2.0, min(image.size) * 0.035)))
    value = np.asarray(image.convert("RGB")).copy()
    value[mask] = np.asarray(base)[mask]
    return Image.fromarray(value, mode="RGB")


def visual_positions(inputs: dict[str, torch.Tensor], image_token_id: int) -> torch.Tensor:
    positions = (inputs["input_ids"][0] == image_token_id).nonzero().flatten()
    if not len(positions) or not torch.all(positions[1:] - positions[:-1] == 1):
        raise ValueError("visual tokens are absent or non-contiguous")
    return positions.to("cuda")


def capture_visual(model: ActivationPatchingModel, inputs: dict[str, torch.Tensor],
                   layers: list[int], positions: torch.Tensor):
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for layer in layers:
        def hook(_module, _args, output, layer_index=layer):
            hidden = decoder_hidden(output, layer_index)
            captured[layer_index] = hidden[:, positions, :].detach().clone()
        handles.append(model.decoder_layers[layer].register_forward_hook(hook))
    try:
        probabilities = model.probabilities_from_inputs(inputs)
    finally:
        for h in handles: h.remove()
    if set(captured) != set(layers):
        raise RuntimeError("failed to capture requested visual-token states")
    return probabilities, captured


def patch_visual(model: ActivationPatchingModel, inputs: dict[str, torch.Tensor],
                 layer: int, positions: torch.Tensor, replacement: torch.Tensor):
    def hook(_module, _args, output):
        hidden = decoder_hidden(output, layer)
        patched = hidden.clone()
        patched[:, positions, :] = replacement.to(hidden.device, hidden.dtype)
        if isinstance(output, torch.Tensor): return patched
        return (patched, *output[1:])
    handle = model.decoder_layers[layer].register_forward_hook(hook)
    try:
        return model.probabilities_from_inputs(inputs)
    finally:
        handle.remove()


def evaluate_case(model, case, layers):
    image_path = Path(case["image"])
    if sha256_file(image_path) != case["image_sha256"]: raise ValueError("image hash mismatch")
    image = Image.open(image_path).convert("RGB")
    deleted = delete_region(image, mask_from_boxes(case, image.size))
    prompt = model_prompt(model.processor, case["problem"])
    clean_inputs = model.prepare(prompt, image)
    deleted_inputs = model.prepare(prompt, deleted)
    if clean_inputs["input_ids"].shape != deleted_inputs["input_ids"].shape:
        raise ValueError("deletion changed token sequence geometry")
    if not torch.equal(clean_inputs["image_grid_thw"], deleted_inputs["image_grid_thw"]):
        raise ValueError("deletion changed visual grid")
    pos_clean = visual_positions(clean_inputs, model.image_token_id)
    pos_deleted = visual_positions(deleted_inputs, model.image_token_id)
    if not torch.equal(pos_clean, pos_deleted): raise ValueError("visual positions differ")
    clean, clean_states = capture_visual(model, clean_inputs, layers, pos_clean)
    deleted_scores, deleted_states = capture_visual(model, deleted_inputs, layers, pos_deleted)
    target = OPTION_LETTERS.index(case["target_choice"])
    patched = {}
    for layer in layers:
        scores = patch_visual(model, deleted_inputs, layer, pos_deleted, clean_states[layer])
        delta = clean_states[layer] - deleted_states[layer]
        gap = float(margins(clean)[target] - margins(deleted_scores)[target])
        raw_recovery = float(margins(scores)[target] - margins(deleted_scores)[target])
        patched[str(layer)] = {
            "visual_token_delta_l2": float(torch.linalg.vector_norm(delta.float()).cpu()),
            "clean_patch": condition(scores, target),
            "margin_recovery": raw_recovery,
            "clean_deleted_margin_gap": gap,
            # A ratio is scientifically meaningful only when deletion caused a
            # nontrivial positive loss.  Keep raw recovery for every case.
            "recovery_fraction": (raw_recovery / gap if gap > 1e-4 else None),
        }
    result = {
        "panel_index": case["panel_index"], "source_index": case["index"],
        "source_record_sha256": case["source_record_sha256"], "image_sha256": case["image_sha256"],
        "target_choice": case["target_choice"], "clean": condition(clean, target),
        "evidence_deleted": condition(deleted_scores, target), "patched_by_layer": patched,
        "patch_target": "all_visual_tokens_at_decoder_layer_output",
    }
    del clean_inputs, deleted_inputs, clean_states, deleted_states
    gc.collect(); torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--layers", default="0,4,8,12,16,20,24,27")
    args = ap.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    panel = json.loads(args.panel.read_text())
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    args.output_dir.mkdir(parents=True)
    model = ActivationPatchingModel(args.model, batch_size=1)
    selected = []
    for case in panel["cases"]:
        image = Image.open(case["image"]).convert("RGB")
        prompt = model_prompt(model.processor, case["problem"])
        scores = model.probabilities_from_inputs(model.prepare(prompt, image))
        pred = OPTION_LETTERS[int(np.argmax(scores))]
        completion = generate(model, reasoning_prompt(model.processor, case["problem"]), image)
        reasoning_choice = parsed(completion)
        row = {"panel_index": case["panel_index"], "predicted_choice": pred,
               "reasoning_choice": reasoning_choice, "reasoning_completion": completion,
               "target_choice": case["target_choice"],
               "forced_clean_correct": pred == case["target_choice"],
               "reasoning_clean_correct": reasoning_choice == case["target_choice"],
               "interface_agreement": reasoning_choice == pred}
        (args.output_dir / "candidate_filter.jsonl").open("a").write(json.dumps(row)+"\n")
        if row["reasoning_clean_correct"] and row["interface_agreement"]: selected.append(case)
    records = []
    with (args.output_dir / "case_results.jsonl").open("w") as out:
        for case in selected:
            row = evaluate_case(model, case, layers); records.append(row)
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+"\n"); out.flush()
            print(json.dumps({"done":len(records),"total_clean_correct":len(selected),"panel_index":case["panel_index"]}), flush=True)
    by_layer = {}
    for layer in layers:
        raw=[r["patched_by_layer"][str(layer)]["margin_recovery"] for r in records]
        vals=[r["patched_by_layer"][str(layer)]["recovery_fraction"] for r in records
              if r["patched_by_layer"][str(layer)]["recovery_fraction"] is not None]
        by_layer[str(layer)]={"case_count":len(raw),
                              "positive_deletion_gap_case_count":len(vals),
                              "mean_raw_margin_recovery":float(np.mean(raw)) if raw else None,
                              "median_raw_margin_recovery":float(np.median(raw)) if raw else None,
                              "mean_recovery_fraction_positive_gap_only":float(np.mean(vals)) if vals else None,
                              "median_recovery_fraction_positive_gap_only":float(np.median(vals)) if vals else None}
    metrics={"schema_version":1,"status":"completed","created_at":datetime.now(timezone.utc).isoformat(),
             "model":str(args.model.resolve()),"panel":str(args.panel.resolve()),"panel_sha256":sha256_file(args.panel),
             "candidate_count":len(panel["cases"]),
             "primary_reasoning_clean_correct_and_interface_agreement_count":len(selected),
             "layers":layers,"patch_target":"all_visual_tokens_at_decoder_layer_output",
             "by_layer":by_layer,"claim_boundary":"visual-token activation recovery supports mediation, but is not sufficient to establish universal causal necessity"}
    (args.output_dir/"metrics.json").write_text(json.dumps(metrics,indent=2,sort_keys=True)+"\n")
    print(json.dumps(metrics, sort_keys=True))

if __name__ == "__main__": main()
