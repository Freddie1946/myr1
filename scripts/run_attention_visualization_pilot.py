#!/usr/bin/env python3
"""Generate Qwen2.5-VL text-to-vision attention maps for a frozen panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


OPTION_LETTERS = "ABCD"
PROMPT_SUFFIX = "\nReturn only one option letter: A, B, C, or D. Do not provide reasoning or other text."


def normalize_attention(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("attention values must be a finite nonnegative vector")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("attention vector has zero mass")
    return values / total


def option_margin(option_logits: torch.Tensor, option_index: int) -> torch.Tensor:
    if option_logits.ndim != 1 or option_logits.numel() < 2:
        raise ValueError("option_logits must contain at least two scores")
    alternatives = torch.cat((option_logits[:option_index], option_logits[option_index + 1 :]))
    return option_logits[option_index] - torch.logsumexp(alternatives, dim=0)


def gradient_attention_relevance(
    score: torch.Tensor,
    attentions: tuple[torch.Tensor, ...],
    visual_positions: torch.Tensor,
    *,
    grid_rows: int,
    grid_columns: int,
    retain_graph: bool,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    gradients = torch.autograd.grad(
        score, attentions, retain_graph=retain_graph, allow_unused=False
    )
    direct_maps = []
    rollout_matrices = []
    sequence_length = attentions[0].shape[-1]
    identity = torch.eye(sequence_length, device=attentions[0].device, dtype=torch.float32)
    for attention, gradient in zip(attentions, gradients):
        contribution = (attention.float() * gradient.float()).clamp(min=0).mean(dim=1)[0]
        direct = contribution[-1].index_select(0, visual_positions)
        direct_maps.append(
            normalize_attention(direct.detach().cpu().numpy()).reshape(grid_rows, grid_columns)
        )
        augmented = contribution + identity
        augmented = augmented / augmented.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        rollout_matrices.append(augmented)
    direct_last4 = np.mean(direct_maps[-4:], axis=0)
    direct_last4 /= direct_last4.sum()
    joint = rollout_matrices[0]
    for matrix in rollout_matrices[1:]:
        joint = matrix @ joint
    rollout = joint[-1].index_select(0, visual_positions).detach().cpu().numpy()
    rollout = normalize_attention(rollout).reshape(grid_rows, grid_columns)
    return direct_last4, rollout, direct_maps


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_prompt(processor: Any, problem: str) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": problem + PROMPT_SUFFIX},
    ]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def resized_map(values: np.ndarray, width: int, height: int) -> np.ndarray:
    grid = Image.fromarray(values.astype(np.float32), mode="F")
    return np.asarray(grid.resize((width, height), resample=Image.Resampling.BILINEAR))


def save_triptych(path: Path, image: Image.Image, values: np.ndarray, title: str) -> None:
    heat = resized_map(values, image.width, image.height)
    vmax = float(np.max(heat)) or 1.0
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[1].imshow(heat, cmap="inferno", vmin=0, vmax=vmax)
    axes[1].set_title(title)
    axes[2].imshow(image)
    overlay = axes[2].imshow(heat, cmap="inferno", vmin=0, vmax=vmax, alpha=0.5)
    axes[2].set_title("Image + attention")
    for axis in axes:
        axis.axis("off")
    figure.colorbar(overlay, ax=axes, shrink=0.75, label="normalized visual attention mass")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_layer_grid(
    path: Path, image: Image.Image, maps: list[np.ndarray], *, title: str
) -> None:
    columns = 4
    rows = math.ceil(len(maps) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(16, 4 * rows), constrained_layout=True)
    flat_axes = np.asarray(axes).reshape(-1)
    for layer, (axis, values) in enumerate(zip(flat_axes, maps)):
        heat = resized_map(values, image.width, image.height)
        axis.imshow(image)
        axis.imshow(heat, cmap="inferno", alpha=0.5, vmin=0, vmax=float(np.max(heat)) or 1.0)
        axis.set_title(f"Layer {layer}")
        axis.axis("off")
    for axis in flat_axes[len(maps):]:
        axis.axis("off")
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-index", action="append", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_raw = args.panel.read_bytes()
    panel = json.loads(panel_raw)
    cases = {case["panel_index"]: case for case in panel["cases"]}

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, use_fast=False)
    processor.image_processor.max_pixels = 65536
    processor.image_processor.min_pixels = 3136
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    option_ids = [processor.tokenizer.encode(letter, add_special_tokens=False) for letter in OPTION_LETTERS]
    if any(len(ids) != 1 for ids in option_ids):
        raise ValueError(f"A-D are not single tokens: {option_ids}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to("cuda").eval()

    records = []
    for panel_index in args.panel_index:
        case = cases[panel_index]
        image_path = Path(case["image"])
        if sha256_file(image_path) != case["image_sha256"]:
            raise ValueError(f"image hash mismatch for case {panel_index}")
        image = Image.open(image_path).convert("RGB")
        prompt = model_prompt(processor, case["problem"])
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        image_positions = (inputs["input_ids"][0] == image_token_id).nonzero().flatten()
        if not len(image_positions) or not torch.all(image_positions[1:] - image_positions[:-1] == 1):
            raise ValueError("image tokens are absent or non-contiguous")
        grid_t, grid_h_raw, grid_w_raw = map(int, inputs["image_grid_thw"][0])
        merge_size = int(processor.image_processor.merge_size)
        grid_rows = grid_h_raw // merge_size
        grid_columns = grid_w_raw // merge_size
        expected_tokens = grid_t * grid_rows * grid_columns
        if grid_t != 1 or expected_tokens != len(image_positions):
            raise ValueError(
                f"unexpected visual grid: thw={(grid_t, grid_h_raw, grid_w_raw)}, "
                f"merge={merge_size}, tokens={len(image_positions)}"
            )
        device_inputs = {name: value.to("cuda") for name, value in inputs.items()}
        output = model(
            **device_inputs,
            use_cache=False,
            output_attentions=True,
            return_dict=True,
        )
        visual_positions = image_positions.to("cuda")
        text_query_start = int(image_positions[-1]) + 1
        text_query_positions = torch.arange(text_query_start, inputs["input_ids"].shape[1], device="cuda")
        last_query_maps = []
        blog_text_query_maps = []
        visual_mass_by_layer = []
        for attention in output.attentions:
            last_raw = (
                attention[0, :, -1, :]
                .mean(dim=0)
                .index_select(0, visual_positions)
                .float()
                .detach()
                .cpu()
                .numpy()
            )
            text_raw = (
                attention[0, :, text_query_positions, :]
                .mean(dim=(0, 1))
                .index_select(0, visual_positions)
                .float()
                .detach()
                .cpu()
                .numpy()
            )
            last_query_maps.append(normalize_attention(last_raw).reshape(grid_rows, grid_columns))
            blog_text_query_maps.append(normalize_attention(text_raw).reshape(grid_rows, grid_columns))
            visual_mass_by_layer.append(float(last_raw.sum()))

        last4 = np.mean(last_query_maps[-4:], axis=0)
        blog_all = np.mean(blog_text_query_maps, axis=0)
        last4 /= last4.sum()
        blog_all /= blog_all.sum()
        option_logits = output.logits[0, -1, [ids[0] for ids in option_ids]].float()
        option_probabilities = torch.softmax(option_logits, dim=0).detach().cpu().tolist()
        predicted_choice = OPTION_LETTERS[int(torch.argmax(option_logits))]
        predicted_index = OPTION_LETTERS.index(predicted_choice)
        target_index = OPTION_LETTERS.index(case["target_choice"])
        if predicted_index == target_index:
            target_direct, target_rollout, target_layers = gradient_attention_relevance(
                option_margin(option_logits, target_index),
                output.attentions,
                visual_positions,
                grid_rows=grid_rows,
                grid_columns=grid_columns,
                retain_graph=False,
            )
            predicted_direct, predicted_rollout, predicted_layers = (
                target_direct,
                target_rollout,
                target_layers,
            )
        else:
            target_direct, target_rollout, target_layers = gradient_attention_relevance(
                option_margin(option_logits, target_index),
                output.attentions,
                visual_positions,
                grid_rows=grid_rows,
                grid_columns=grid_columns,
                retain_graph=True,
            )
            predicted_direct, predicted_rollout, predicted_layers = gradient_attention_relevance(
                option_margin(option_logits, predicted_index),
                output.attentions,
                visual_positions,
                grid_rows=grid_rows,
                grid_columns=grid_columns,
                retain_graph=False,
            )
        record = {
            "panel_index": panel_index,
            "source_index": case["index"],
            "image": str(image_path.resolve()),
            "image_sha256": case["image_sha256"],
            "target_choice": case["target_choice"],
            "predicted_choice": predicted_choice,
            "correct": predicted_choice == case["target_choice"],
            "option_probabilities_diagnostic_only_not_used_to_rank_regions": dict(zip(OPTION_LETTERS, option_probabilities)),
            "grid_rows": grid_rows,
            "grid_columns": grid_columns,
            "visual_token_count": len(image_positions),
            "layer_count": len(output.attentions),
            "visual_mass_by_layer_last_query": visual_mass_by_layer,
            "last_query_last4_attention": last4.tolist(),
            "last_query_last4_order": np.argsort(-last4.reshape(-1)).tolist(),
            "blog_text_query_all_layers_attention": blog_all.tolist(),
            "blog_text_query_all_layers_order": np.argsort(-blog_all.reshape(-1)).tolist(),
            "target_option_margin": float(option_margin(option_logits, target_index).detach()),
            "target_grad_attention_last4": target_direct.tolist(),
            "target_grad_attention_last4_order": np.argsort(-target_direct.reshape(-1)).tolist(),
            "target_chefer_style_rollout": target_rollout.tolist(),
            "target_chefer_style_rollout_order": np.argsort(-target_rollout.reshape(-1)).tolist(),
            "predicted_option_margin": float(option_margin(option_logits, predicted_index).detach()),
            "predicted_grad_attention_last4": predicted_direct.tolist(),
            "predicted_grad_attention_last4_order": np.argsort(-predicted_direct.reshape(-1)).tolist(),
            "predicted_chefer_style_rollout": predicted_rollout.tolist(),
            "predicted_chefer_style_rollout_order": np.argsort(-predicted_rollout.reshape(-1)).tolist(),
            "last_query_layer_maps": [values.tolist() for values in last_query_maps],
            "blog_text_query_layer_maps": [values.tolist() for values in blog_text_query_maps],
            "target_grad_attention_layer_maps": [values.tolist() for values in target_layers],
            "predicted_grad_attention_layer_maps": [values.tolist() for values in predicted_layers],
        }
        records.append(record)
        save_triptych(
            args.output_dir / f"case_{panel_index:02d}_last_query_last4.png",
            image,
            last4,
            "Last query, last 4 layers",
        )
        save_triptych(
            args.output_dir / f"case_{panel_index:02d}_blog_text_all_layers.png",
            image,
            blog_all,
            "Post-image text queries, all layers",
        )
        save_layer_grid(
            args.output_dir / f"case_{panel_index:02d}_last_query_layers.png",
            image,
            last_query_maps,
            title="Last-query visual attention by language-model layer",
        )
        save_triptych(
            args.output_dir / f"case_{panel_index:02d}_target_grad_attention_last4.png",
            image,
            target_direct,
            "Target answer: gradient x attention",
        )
        save_triptych(
            args.output_dir / f"case_{panel_index:02d}_target_relevance_rollout.png",
            image,
            target_rollout,
            "Target answer: relevance rollout",
        )
        if predicted_index != target_index:
            save_triptych(
                args.output_dir / f"case_{panel_index:02d}_predicted_grad_attention_last4.png",
                image,
                predicted_direct,
                "Predicted answer: gradient x attention",
            )
            save_triptych(
                args.output_dir / f"case_{panel_index:02d}_predicted_relevance_rollout.png",
                image,
                predicted_rollout,
                "Predicted answer: relevance rollout",
            )
        del output, device_inputs
        torch.cuda.empty_cache()

    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_pilot",
        "formal_result": False,
        "claim": "attention_visualization_not_causal_explanation",
        "model_path": str(args.model.resolve()),
        "panel_path": str(args.panel.resolve()),
        "panel_sha256": hashlib.sha256(panel_raw).hexdigest(),
        "records": records,
    }
    (args.output_dir / "attention_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "cases": len(records), "output": str(args.output_dir.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
