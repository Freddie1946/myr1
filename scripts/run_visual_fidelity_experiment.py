#!/usr/bin/env python3
"""Run outcome-blind occlusion fidelity analysis for a frozen PathMMU panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


OPTION_LETTERS = "ABCD"
PROMPT_SUFFIX = (
    "\nReturn only one option letter: A, B, C, or D. Do not provide reasoning or other text."
)
DEFAULT_FRACTIONS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def grid_boxes(width: int, height: int, rows: int, columns: int) -> list[tuple[int, int, int, int]]:
    if min(width, height, rows, columns) <= 0:
        raise ValueError("image and grid dimensions must be positive")
    return [
        (
            width * column // columns,
            height * row // rows,
            width * (column + 1) // columns,
            height * (row + 1) // rows,
        )
        for row in range(rows)
        for column in range(columns)
    ]


def mean_fill(image: Image.Image) -> tuple[int, int, int]:
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    return tuple(int(round(float(value))) for value in pixels.mean(axis=(0, 1)))


def perturb_image(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    selected: Iterable[int],
    *,
    mode: str,
    fill: tuple[int, int, int],
) -> Image.Image:
    source = image.convert("RGB")
    selected_set = set(selected)
    if mode == "deletion":
        result = source.copy()
        for index in selected_set:
            result.paste(fill, boxes[index])
        return result
    if mode == "insertion":
        result = Image.new("RGB", source.size, fill)
        for index in selected_set:
            result.paste(source.crop(boxes[index]), boxes[index])
        return result
    raise ValueError(f"unknown perturbation mode: {mode}")


def trapezoid_auc(fractions: list[float], values: list[float]) -> float:
    if len(fractions) != len(values) or len(values) < 2:
        raise ValueError("AUC inputs must have equal length of at least two")
    if fractions[0] != 0.0 or fractions[-1] != 1.0:
        raise ValueError("AUC fractions must span zero through one")
    if any(right <= left for left, right in zip(fractions, fractions[1:])):
        raise ValueError("AUC fractions must be strictly increasing")
    return float(np.trapz(values, fractions))


def selected_prefix(order: list[int], fraction: float) -> list[int]:
    count = min(len(order), max(0, int(round(fraction * len(order)))))
    return order[:count]


def model_prompt(processor: Any, problem: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": problem + PROMPT_SUFFIX},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


class OptionScorer:
    def __init__(self, model_path: Path, *, batch_size: int):
        self.model_path = model_path.resolve()
        self.batch_size = batch_size
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, local_files_only=True, use_fast=False
        )
        self.processor.tokenizer.padding_side = "left"
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.max_pixels = 65536
            self.processor.image_processor.min_pixels = 3136
        option_ids = [
            self.processor.tokenizer.encode(letter, add_special_tokens=False)
            for letter in OPTION_LETTERS
        ]
        if any(len(ids) != 1 for ids in option_ids):
            raise ValueError(f"A-D are not single tokens for this tokenizer: {option_ids}")
        self.option_ids = torch.tensor([ids[0] for ids in option_ids], device="cuda")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to("cuda")
        self.model.eval()

    def score(self, prompt: str, images: list[Image.Image]) -> list[list[float]]:
        rows: list[list[float]] = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            inputs = self.processor(
                text=[prompt] * len(batch), images=batch, return_tensors="pt", padding=True
            )
            inputs = {name: value.to("cuda") for name, value in inputs.items()}
            with torch.inference_mode():
                logits = self.model(**inputs, use_cache=False).logits[:, -1, :]
                option_logits = logits.index_select(1, self.option_ids)
                probabilities = torch.softmax(option_logits.float(), dim=-1)
            rows.extend(probabilities.cpu().tolist())
        return rows


def random_orders(case_sha: str, patch_count: int, count: int) -> list[list[int]]:
    seed = int(hashlib.sha256(f"visual-random-v1:{case_sha}".encode()).hexdigest()[:16], 16)
    generator = random.Random(seed)
    orders = []
    for _ in range(count):
        order = list(range(patch_count))
        generator.shuffle(order)
        orders.append(order)
    return orders


def curve_images(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    order: list[int],
    fractions: list[float],
    fill: tuple[int, int, int],
) -> tuple[list[Image.Image], list[Image.Image]]:
    deletion = [
        perturb_image(image, boxes, selected_prefix(order, fraction), mode="deletion", fill=fill)
        for fraction in fractions
    ]
    insertion = [
        perturb_image(image, boxes, selected_prefix(order, fraction), mode="insertion", fill=fill)
        for fraction in fractions
    ]
    return deletion, insertion


def target_values(probabilities: list[list[float]], target_index: int) -> list[float]:
    return [float(row[target_index]) for row in probabilities]


def evaluate_case(
    scorer: OptionScorer,
    case: dict[str, Any],
    *,
    rows: int,
    columns: int,
    fractions: list[float],
    random_permutations: int,
) -> dict[str, Any]:
    image_path = Path(case["image"])
    if sha256_file(image_path) != case["image_sha256"]:
        raise ValueError(f"image hash mismatch for panel case {case['panel_index']}")
    with Image.open(image_path) as handle:
        image = handle.convert("RGB")
    boxes = grid_boxes(image.width, image.height, rows, columns)
    fill = mean_fill(image)
    prompt = model_prompt(scorer.processor, case["problem"])
    target_index = OPTION_LETTERS.index(case["target_choice"])

    single_patch_images = [image] + [
        perturb_image(image, boxes, [index], mode="deletion", fill=fill)
        for index in range(len(boxes))
    ]
    single_scores = scorer.score(prompt, single_patch_images)
    baseline = single_scores[0]
    baseline_target = float(baseline[target_index])
    importance = [
        math.log(max(baseline_target, 1e-12))
        - math.log(max(float(row[target_index]), 1e-12))
        for row in single_scores[1:]
    ]
    top_order = sorted(range(len(boxes)), key=lambda index: (-importance[index], index))

    top_deletion_images, top_insertion_images = curve_images(
        image, boxes, top_order, fractions, fill
    )
    top_scores = scorer.score(prompt, top_deletion_images + top_insertion_images)
    split = len(fractions)
    top_deletion = target_values(top_scores[:split], target_index)
    top_insertion = target_values(top_scores[split:], target_index)

    random_deletion_curves = []
    random_insertion_curves = []
    for order in random_orders(case["source_record_sha256"], len(boxes), random_permutations):
        deletion_images, insertion_images = curve_images(image, boxes, order, fractions, fill)
        scores = scorer.score(prompt, deletion_images + insertion_images)
        random_deletion_curves.append(target_values(scores[:split], target_index))
        random_insertion_curves.append(target_values(scores[split:], target_index))
    random_deletion = np.asarray(random_deletion_curves).mean(axis=0).tolist()
    random_insertion = np.asarray(random_insertion_curves).mean(axis=0).tolist()
    top_fraction_index = min(
        range(len(fractions)), key=lambda index: abs(fractions[index] - 0.25)
    )
    return {
        "panel_index": case["panel_index"],
        "source_index": case["index"],
        "source_record_sha256": case["source_record_sha256"],
        "image": case["image"],
        "image_sha256": case["image_sha256"],
        "image_width": image.width,
        "image_height": image.height,
        "target_choice": case["target_choice"],
        "baseline_option_probabilities": dict(zip(OPTION_LETTERS, baseline)),
        "baseline_predicted_choice": OPTION_LETTERS[int(np.argmax(baseline))],
        "baseline_correct": int(np.argmax(baseline)) == target_index,
        "baseline_target_probability": baseline_target,
        "mean_fill_rgb": list(fill),
        "patch_importance_log_target_probability_drop": importance,
        "top_patch_order": top_order,
        "fractions": fractions,
        "top_deletion_target_probability": top_deletion,
        "top_insertion_target_probability": top_insertion,
        "random_deletion_target_probability_mean": random_deletion,
        "random_insertion_target_probability_mean": random_insertion,
        "random_deletion_target_probability_all": random_deletion_curves,
        "random_insertion_target_probability_all": random_insertion_curves,
        "top_deletion_auc": trapezoid_auc(fractions, top_deletion),
        "random_deletion_auc": trapezoid_auc(fractions, random_deletion),
        "top_insertion_auc": trapezoid_auc(fractions, top_insertion),
        "random_insertion_auc": trapezoid_auc(fractions, random_insertion),
        "deletion_auc_advantage": trapezoid_auc(fractions, random_deletion)
        - trapezoid_auc(fractions, top_deletion),
        "insertion_auc_advantage": trapezoid_auc(fractions, top_insertion)
        - trapezoid_auc(fractions, random_insertion),
        "top_25pct_target_probability": top_deletion[top_fraction_index],
        "random_25pct_target_probability": random_deletion[top_fraction_index],
        "top_25pct_comprehensiveness": baseline_target - top_deletion[top_fraction_index],
        "random_25pct_comprehensiveness": baseline_target
        - random_deletion[top_fraction_index],
    }


def heatmap_array(values: list[float], rows: int, columns: int, size: tuple[int, int]) -> np.ndarray:
    grid = np.asarray(values, dtype=np.float32).reshape(rows, columns)
    scaled = Image.fromarray(grid, mode="F").resize(size, resample=Image.Resampling.BILINEAR)
    return np.asarray(scaled, dtype=np.float32)


def save_case_figure(
    case: dict[str, Any], result: dict[str, Any], output: Path, *, rows: int, columns: int,
    color_limit: float,
) -> None:
    with Image.open(case["image"]) as handle:
        image = np.asarray(handle.convert("RGB"))
    heatmap = heatmap_array(
        result["patch_importance_log_target_probability_drop"],
        rows,
        columns,
        (image.shape[1], image.shape[0]),
    )
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title(f"Case {case['panel_index']:02d}: target {case['target_choice']}")
    mapped = axes[1].imshow(heatmap, cmap="coolwarm", vmin=-color_limit, vmax=color_limit)
    axes[1].set_title("Occlusion importance")
    axes[2].imshow(image)
    axes[2].imshow(heatmap, cmap="coolwarm", vmin=-color_limit, vmax=color_limit, alpha=0.52)
    axes[2].set_title("Image + importance")
    for axis in axes:
        axis.axis("off")
    colorbar = figure.colorbar(mapped, ax=axes[1:], shrink=0.82)
    colorbar.set_label("log p(target) drop after patch occlusion")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def save_curve_figure(results: list[dict[str, Any]], output: Path) -> None:
    fractions = np.asarray(results[0]["fractions"])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, prefix, title in (
        (axes[0], "deletion", "Deletion fidelity"),
        (axes[1], "insertion", "Insertion fidelity"),
    ):
        top = np.asarray([row[f"top_{prefix}_target_probability"] for row in results])
        random_values = np.asarray(
            [row[f"random_{prefix}_target_probability_mean"] for row in results]
        )
        for values, label, color in ((top, "Importance order", "#b2182b"), (random_values, "Random order", "#2166ac")):
            mean = values.mean(axis=0)
            sem = values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
            axis.plot(fractions, mean, marker="o", label=label, color=color)
            axis.fill_between(fractions, mean - sem, mean + sem, color=color, alpha=0.18)
        axis.set_title(title)
        axis.set_xlabel("Fraction of image patches")
        axis.set_ylabel("Normalized target probability")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = (
        "baseline_target_probability",
        "top_deletion_auc",
        "random_deletion_auc",
        "top_insertion_auc",
        "random_insertion_auc",
        "deletion_auc_advantage",
        "insertion_auc_advantage",
        "top_25pct_comprehensiveness",
        "random_25pct_comprehensiveness",
    )
    return {
        "case_count": len(results),
        "baseline_correct": sum(row["baseline_correct"] for row in results),
        "baseline_accuracy": sum(row["baseline_correct"] for row in results) / len(results),
        **{
            f"mean_{name}": float(np.mean([row[name] for row in results]))
            for name in numeric
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--expected-panel-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-rows", type=int, default=6)
    parser.add_argument("--grid-columns", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--random-permutations", type=int, default=5)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite visualization run: {args.output_dir}")
    panel_sha = sha256_file(args.panel)
    if panel_sha != args.expected_panel_sha256:
        raise ValueError(f"panel SHA-256 mismatch: {panel_sha}")
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    if (
        panel.get("status") != "frozen_before_model_visualization_outputs"
        or panel.get("selection_uses_model_outputs") is not False
        or panel.get("case_count") != 24
        or len({row["image_sha256"] for row in panel.get("cases", [])}) != 24
    ):
        raise ValueError("visualization panel contract mismatch")
    if min(args.grid_rows, args.grid_columns, args.batch_size, args.random_permutations) <= 0:
        raise ValueError("grid, batch, and permutation values must be positive")
    args.output_dir.mkdir(parents=True)
    scorer = OptionScorer(args.model, batch_size=args.batch_size)
    results = []
    for case in panel["cases"]:
        result = evaluate_case(
            scorer,
            case,
            rows=args.grid_rows,
            columns=args.grid_columns,
            fractions=list(DEFAULT_FRACTIONS),
            random_permutations=args.random_permutations,
        )
        results.append(result)
        print(canonical_json({
            "model": args.model_label,
            "completed_cases": len(results),
            "case_count": len(panel["cases"]),
            "panel_index": case["panel_index"],
        }), flush=True)
    predictions = args.output_dir / "case_results.jsonl"
    predictions.write_text(
        "".join(canonical_json(row) + "\n" for row in results), encoding="utf-8"
    )
    color_limit = max(
        1e-6,
        max(abs(value) for row in results for value in row["patch_importance_log_target_probability_drop"]),
    )
    figures = args.output_dir / "figures"
    figures.mkdir()
    for case, result in zip(panel["cases"], results):
        save_case_figure(
            case,
            result,
            figures / f"case_{case['panel_index']:02d}_heatmap.png",
            rows=args.grid_rows,
            columns=args.grid_columns,
            color_limit=color_limit,
        )
    save_curve_figure(results, figures / "aggregate_deletion_insertion_curves.png")
    metrics = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "method": "patch_occlusion_target_option_probability_fidelity_v1",
        "attention_claim": False,
        "target_probability_definition": "softmax_over_next_token_logits_for_A_B_C_D",
        "model_label": args.model_label,
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "panel_path": str(args.panel.resolve()),
        "panel_sha256": panel_sha,
        "selection_uses_model_outputs": False,
        "grid_rows": args.grid_rows,
        "grid_columns": args.grid_columns,
        "fractions": list(DEFAULT_FRACTIONS),
        "random_permutations": args.random_permutations,
        "shared_color_scale_min": -color_limit,
        "shared_color_scale_max": color_limit,
        "case_results": str(predictions.resolve()),
        "case_results_sha256": sha256_file(predictions),
        **aggregate(results),
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(canonical_json(metrics))


if __name__ == "__main__":
    main()
