#!/usr/bin/env python3
"""Test whether raw attention-ranked image regions causally affect model decisions.

The attention ranking is computed once from the untouched image.  It is then
frozen before any intervention outcome is observed.  Deletion and retention
(insertion) curves are evaluated against low-attention and area-matched random
orders with both mean-fill and Gaussian-blur perturbations.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageFilter
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from run_attention_visualization_pilot import (
    OPTION_LETTERS,
    PROMPT_SUFFIX,
    normalize_attention,
    question_option_query_positions,
)
from run_visual_fidelity_experiment import (
    canonical_json,
    grid_boxes,
    model_prompt,
    selected_prefix,
    sha256_file,
    trapezoid_auc,
)


DEFAULT_FRACTIONS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
DEFAULT_LAYER_SPECS = ("layer14", "layer16", "layers14_17", "middle8_17", "late18_27")


def parse_layer_spec(spec: str, layer_count: int) -> list[int]:
    aliases = {
        "layer14": [14],
        "layer16": [16],
        "layers14_17": list(range(14, 18)),
        "middle8_17": list(range(8, 18)),
        "late18_27": list(range(18, 28)),
    }
    if spec not in aliases:
        raise ValueError(f"unknown layer strategy: {spec}")
    layers = aliases[spec]
    if max(layers) >= layer_count:
        raise ValueError(f"{spec} requires layer {max(layers)}, model has {layer_count} layers")
    return layers


def attention_orders(
    layer_maps: list[list[list[float]]], specs: Iterable[str]
) -> tuple[dict[str, list[int]], dict[str, list[float]]]:
    arrays = [np.asarray(values, dtype=np.float64) for values in layer_maps]
    if not arrays or any(values.shape != arrays[0].shape for values in arrays):
        raise ValueError("attention layers must be nonempty and share one grid shape")
    orders: dict[str, list[int]] = {}
    maps: dict[str, list[float]] = {}
    for spec in specs:
        layers = parse_layer_spec(spec, len(arrays))
        values = np.mean([arrays[layer] for layer in layers], axis=0)
        values = normalize_attention(values.reshape(-1))
        maps[spec] = values.tolist()
        orders[spec] = np.argsort(-values, kind="stable").tolist()
    primary = "layers14_17"
    if primary not in orders:
        raise ValueError(f"required primary exploratory strategy missing: {primary}")
    orders["low_attention_layers14_17"] = list(reversed(orders[primary]))
    maps["low_attention_layers14_17"] = maps[primary]
    return orders, maps


def deterministic_random_orders(case_sha: str, patch_count: int, count: int) -> list[list[int]]:
    seed_text = f"attention-intervention-v1:{case_sha}"
    generator = random.Random(int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16))
    output = []
    for _ in range(count):
        order = list(range(patch_count))
        generator.shuffle(order)
        output.append(order)
    return output


def perturbation_base(image: Image.Image, kind: str) -> Image.Image:
    image = image.convert("RGB")
    if kind == "mean":
        pixels = np.asarray(image, dtype=np.float32)
        fill = tuple(int(round(float(value))) for value in pixels.mean(axis=(0, 1)))
        return Image.new("RGB", image.size, fill)
    if kind == "blur":
        radius = max(2.0, min(image.size) * 0.035)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    raise ValueError(f"unknown perturbation kind: {kind}")


def intervene(
    image: Image.Image,
    base: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    selected: Iterable[int],
    mode: str,
) -> Image.Image:
    source = image.convert("RGB")
    selected = set(selected)
    if mode == "deletion":
        result = source.copy()
        for index in selected:
            result.paste(base.crop(boxes[index]), boxes[index])
        return result
    if mode == "retention":
        result = base.copy()
        for index in selected:
            result.paste(source.crop(boxes[index]), boxes[index])
        return result
    raise ValueError(f"unknown intervention mode: {mode}")


class AttentionExtractor:
    def __init__(self, model_path: Path):
        self.model_path = model_path.resolve()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, local_files_only=True, use_fast=False
        )
        self.processor.image_processor.max_pixels = 65536
        self.processor.image_processor.min_pixels = 3136
        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to("cuda").eval()

    def extract(self, problem: str, image: Image.Image) -> dict[str, Any]:
        prompt = model_prompt(self.processor, problem)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        image_positions = (inputs["input_ids"][0] == self.image_token_id).nonzero().flatten()
        if not len(image_positions) or not torch.all(image_positions[1:] - image_positions[:-1] == 1):
            raise ValueError("image tokens are absent or non-contiguous")
        grid_t, grid_h_raw, grid_w_raw = map(int, inputs["image_grid_thw"][0])
        merge = int(self.processor.image_processor.merge_size)
        rows, columns = grid_h_raw // merge, grid_w_raw // merge
        if grid_t != 1 or rows * columns != len(image_positions):
            raise ValueError("visual-token count does not match merged image grid")
        query_cpu, query_tokens = question_option_query_positions(
            self.processor.tokenizer,
            inputs["input_ids"][0],
            last_image_position=int(image_positions[-1]),
        )
        device_inputs = {name: value.to("cuda") for name, value in inputs.items()}
        query = query_cpu.to("cuda")
        visual = image_positions.to("cuda")
        with torch.inference_mode():
            output = self.model(
                **device_inputs,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
            maps = []
            for attention in output.attentions:
                raw = (
                    attention[0, :, query, :]
                    .mean(dim=(0, 1))
                    .index_select(0, visual)
                    .float()
                    .cpu()
                    .numpy()
                )
                maps.append(normalize_attention(raw).reshape(rows, columns).tolist())
        del output, device_inputs
        torch.cuda.empty_cache()
        return {
            "grid_rows": rows,
            "grid_columns": columns,
            "question_option_query_tokens": query_tokens,
            "question_options_query_layer_maps": maps,
        }

    def close(self) -> None:
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


class DecisionScorer:
    def __init__(self, model_path: Path, batch_size: int):
        self.model_path = model_path.resolve()
        self.batch_size = batch_size
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, local_files_only=True, use_fast=False
        )
        self.processor.tokenizer.padding_side = "left"
        self.processor.image_processor.max_pixels = 65536
        self.processor.image_processor.min_pixels = 3136
        option_ids = [self.processor.tokenizer.encode(letter, add_special_tokens=False) for letter in OPTION_LETTERS]
        if any(len(ids) != 1 for ids in option_ids):
            raise ValueError(f"A-D are not single tokens: {option_ids}")
        self.option_ids = torch.tensor([ids[0] for ids in option_ids], device="cuda")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to("cuda").eval()

    def score(self, prompt: str, images: list[Image.Image]) -> list[list[float]]:
        output: list[list[float]] = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start : start + self.batch_size]
            inputs = self.processor(text=[prompt] * len(batch), images=batch, return_tensors="pt", padding=True)
            inputs = {name: value.to("cuda") for name, value in inputs.items()}
            with torch.inference_mode():
                logits = self.model(**inputs, use_cache=False).logits[:, -1, :]
                probabilities = torch.softmax(logits.index_select(1, self.option_ids).float(), dim=-1)
            output.extend(probabilities.cpu().tolist())
        return output


def score_curve(
    scorer: DecisionScorer,
    prompt: str,
    image: Image.Image,
    base: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    order: list[int],
    fractions: list[float],
    mode: str,
) -> list[list[float]]:
    images = [
        intervene(image, base, boxes, selected_prefix(order, fraction), mode)
        for fraction in fractions
    ]
    return scorer.score(prompt, images)


def curve_summary(
    probabilities: list[list[float]],
    fractions: list[float],
    decision_index: int,
    target_index: int,
) -> dict[str, Any]:
    decisions = [int(np.argmax(row)) for row in probabilities]
    decision_probabilities = [float(row[decision_index]) for row in probabilities]
    return {
        "option_probabilities": probabilities,
        "predicted_choices": [OPTION_LETTERS[index] for index in decisions],
        "original_decision_probabilities": decision_probabilities,
        "answer_changed": [index != decision_index for index in decisions],
        "correct": [index == target_index for index in decisions],
        "original_decision_probability_auc": trapezoid_auc(fractions, decision_probabilities),
    }


def evaluate_case(
    scorer: DecisionScorer,
    case: dict[str, Any],
    attention: dict[str, Any],
    layer_specs: list[str],
    fractions: list[float],
    random_permutations: int,
) -> dict[str, Any]:
    with Image.open(case["image"]) as handle:
        image = handle.convert("RGB")
    if sha256_file(Path(case["image"])) != case["image_sha256"]:
        raise ValueError(f"image hash mismatch for case {case['panel_index']}")
    rows, columns = attention["grid_rows"], attention["grid_columns"]
    boxes = grid_boxes(image.width, image.height, rows, columns)
    orders, aggregated_maps = attention_orders(
        attention["question_options_query_layer_maps"], layer_specs
    )
    random_orders = deterministic_random_orders(
        case["source_record_sha256"], len(boxes), random_permutations
    )
    prompt = model_prompt(scorer.processor, case["problem"])
    baseline = scorer.score(prompt, [image])[0]
    decision_index = int(np.argmax(baseline))
    target_index = OPTION_LETTERS.index(case["target_choice"])
    curves: dict[str, Any] = {}
    for perturbation in ("mean", "blur"):
        base = perturbation_base(image, perturbation)
        curves[perturbation] = {}
        for mode in ("deletion", "retention"):
            curves[perturbation][mode] = {}
            for name, order in orders.items():
                probabilities = score_curve(
                    scorer, prompt, image, base, boxes, order, fractions, mode
                )
                curves[perturbation][mode][name] = curve_summary(
                    probabilities, fractions, decision_index, target_index
                )
            random_rows = []
            for order in random_orders:
                probabilities = score_curve(
                    scorer, prompt, image, base, boxes, order, fractions, mode
                )
                random_rows.append(
                    curve_summary(probabilities, fractions, decision_index, target_index)
                )
            curves[perturbation][mode]["random_area_matched"] = random_rows
    return {
        "panel_index": case["panel_index"],
        "source_index": case["index"],
        "image": case["image"],
        "image_sha256": case["image_sha256"],
        "target_choice": case["target_choice"],
        "grid_rows": rows,
        "grid_columns": columns,
        "baseline_option_probabilities": dict(zip(OPTION_LETTERS, baseline)),
        "baseline_predicted_choice": OPTION_LETTERS[decision_index],
        "baseline_correct": decision_index == target_index,
        "attention_maps": aggregated_maps,
        "attention_orders": orders,
        "fractions": fractions,
        "curves": curves,
    }


def mean_random_auc(random_rows: list[dict[str, Any]]) -> float:
    return float(np.mean([row["original_decision_probability_auc"] for row in random_rows]))


def aggregate(records: list[dict[str, Any]], layer_specs: list[str]) -> dict[str, Any]:
    strategies = layer_specs + ["low_attention_layers14_17"]
    output: dict[str, Any] = {
        "case_count": len(records),
        "baseline_accuracy": float(np.mean([row["baseline_correct"] for row in records])),
        "by_perturbation_mode_strategy": {},
    }
    for perturbation in ("mean", "blur"):
        for mode in ("deletion", "retention"):
            key = f"{perturbation}:{mode}"
            output["by_perturbation_mode_strategy"][key] = {}
            random_auc = [
                mean_random_auc(row["curves"][perturbation][mode]["random_area_matched"])
                for row in records
            ]
            for strategy in strategies:
                auc = [
                    row["curves"][perturbation][mode][strategy]["original_decision_probability_auc"]
                    for row in records
                ]
                if mode == "deletion":
                    advantage = [random - observed for random, observed in zip(random_auc, auc)]
                else:
                    advantage = [observed - random for random, observed in zip(random_auc, auc)]
                output["by_perturbation_mode_strategy"][key][strategy] = {
                    "mean_original_decision_probability_auc": float(np.mean(auc)),
                    "mean_random_auc": float(np.mean(random_auc)),
                    "mean_fidelity_advantage_over_random": float(np.mean(advantage)),
                    "sample_std_fidelity_advantage": float(np.std(advantage, ddof=1)) if len(advantage) > 1 else 0.0,
                }
    return output


def validate_panel(panel: dict[str, Any]) -> None:
    if (
        panel.get("status") != "frozen_before_model_visualization_outputs"
        or panel.get("selection_uses_model_outputs") is not False
        or panel.get("case_count") != 24
        or len(panel.get("cases", [])) != 24
        or len({row["image_sha256"] for row in panel["cases"]}) != 24
    ):
        raise ValueError("visual-fidelity panel contract mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--expected-panel-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--random-permutations", type=int, default=5)
    parser.add_argument("--panel-index", type=int, action="append")
    parser.add_argument("--layer-strategy", action="append", choices=DEFAULT_LAYER_SPECS)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite run: {args.output_dir}")
    if min(args.batch_size, args.random_permutations) <= 0:
        raise ValueError("batch size and random permutations must be positive")
    panel_sha = sha256_file(args.panel)
    if panel_sha != args.expected_panel_sha256:
        raise ValueError(f"panel SHA-256 mismatch: {panel_sha}")
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    validate_panel(panel)
    selected = set(args.panel_index or range(24))
    cases = [row for row in panel["cases"] if row["panel_index"] in selected]
    if len(cases) != len(selected):
        raise ValueError("requested panel index is missing")
    layer_specs = list(args.layer_strategy or DEFAULT_LAYER_SPECS)
    fractions = list(DEFAULT_FRACTIONS)
    args.output_dir.mkdir(parents=True)
    attention_path = args.output_dir / "attention_maps.jsonl"
    extractor = AttentionExtractor(args.model)
    attention_records = []
    with attention_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            with Image.open(case["image"]) as source:
                image = source.convert("RGB")
            row = {
                "panel_index": case["panel_index"],
                "image_sha256": case["image_sha256"],
                **extractor.extract(case["problem"], image),
            }
            attention_records.append(row)
            handle.write(canonical_json(row) + "\n")
            handle.flush()
            print(canonical_json({"phase": "attention", "completed": len(attention_records), "total": len(cases)}), flush=True)
    extractor.close()
    attention_by_index = {row["panel_index"]: row for row in attention_records}
    scorer = DecisionScorer(args.model, args.batch_size)
    case_path = args.output_dir / "case_results.jsonl"
    records = []
    with case_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            row = evaluate_case(
                scorer,
                case,
                attention_by_index[case["panel_index"]],
                layer_specs,
                fractions,
                args.random_permutations,
            )
            records.append(row)
            handle.write(canonical_json(row) + "\n")
            handle.flush()
            print(canonical_json({"phase": "intervention", "completed": len(records), "total": len(cases)}), flush=True)
    metrics = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_exploratory" if len(cases) == 24 else "completed_smoke",
        "formal_result": False,
        "method": "raw_question_option_attention_guided_deletion_retention_v1",
        "attention_region_selection_uses_answer_logits": False,
        "primary_endpoint": "original_model_decision_probability_and_answer_change",
        "secondary_endpoint": "accuracy_under_intervention",
        "model_label": args.model_label,
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "panel_path": str(args.panel.resolve()),
        "panel_sha256": panel_sha,
        "panel_indices": [row["panel_index"] for row in cases],
        "layer_strategies": layer_specs,
        "perturbations": ["mean", "blur"],
        "modes": ["deletion", "retention"],
        "fractions": fractions,
        "random_permutations": args.random_permutations,
        "attention_maps": str(attention_path.resolve()),
        "attention_maps_sha256": sha256_file(attention_path),
        "case_results": str(case_path.resolve()),
        "case_results_sha256": sha256_file(case_path),
        **aggregate(records, layer_specs),
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(canonical_json(metrics), flush=True)


if __name__ == "__main__":
    main()
