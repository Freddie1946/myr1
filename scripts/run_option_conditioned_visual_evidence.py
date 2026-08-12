#!/usr/bin/env python3
"""Measure whether visual regions support individual answer options.

For every frozen case and every A/B/C/D option, the option-margin gradient
with respect to each transformer layer's visual attention is used only to
select a positive and negative region.  The selected regions are then tested
with independent deletion/retention interventions, while recording all four
option probabilities and margins.  Random area-matched shifts and the
opposite (low-relevance) region are controls.

This is a causal diagnostic, not a claim that attention alone explains a
model.  A region is counted as option support only when its deletion lowers
that option's margin more than it lowers competing options and more than the
matched controls.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

OPTION_LETTERS = "ABCD"
PROMPT_SUFFIX = "\nReturn only one option letter: A, B, C, or D. Do not provide reasoning or other text."


def model_prompt(processor: Any, problem: str) -> str:
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": problem + PROMPT_SUFFIX},
    ]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def option_margin(option_logits: torch.Tensor, option_index: int) -> torch.Tensor:
    alternatives = torch.cat((option_logits[:option_index], option_logits[option_index + 1:]))
    return option_logits[option_index] - torch.logsumexp(alternatives, dim=0)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_partial_records(path: Path, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and load flushed per-case records from an interrupted run."""
    expected = {row["panel_index"]: row for row in cases}
    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid partial JSONL at line {line_number}") from error
            panel_index = row.get("panel_index")
            if panel_index not in expected or panel_index in seen:
                raise ValueError(f"unexpected or duplicate partial panel index: {panel_index}")
            case = expected[panel_index]
            if row.get("image_sha256") != case["image_sha256"] or row.get("target_choice") != case["target_choice"]:
                raise ValueError(f"partial record does not match frozen case: {panel_index}")
            seen.add(panel_index)
            records.append(row)
    return records


def grid_boxes(width: int, height: int, rows: int, columns: int) -> list[tuple[int, int, int, int]]:
    return [(width * c // columns, height * r // rows,
             width * (c + 1) // columns, height * (r + 1) // rows)
            for r in range(rows) for c in range(columns)]


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if total <= 0 or not np.isfinite(total):
        return np.zeros_like(values, dtype=np.float64)
    return values / total


def parse_layers(value: str, layer_count: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(layer_count))
    layers = sorted({int(x) for x in value.split(",") if x.strip()})
    if not layers or min(layers) < 0 or max(layers) >= layer_count:
        raise ValueError(f"invalid layer list {value!r} for {layer_count} layers")
    return layers


def margins(probabilities: list[float]) -> list[float]:
    logits = np.log(np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0))
    return [float(logits[i] - np.logaddexp.reduce(np.delete(logits, i))) for i in range(4)]


def option_ids(processor: Any) -> torch.Tensor:
    encoded = [processor.tokenizer.encode(letter, add_special_tokens=False) for letter in OPTION_LETTERS]
    if any(len(row) != 1 for row in encoded):
        raise ValueError(f"A-D are not single tokens: {encoded}")
    return torch.tensor([row[0] for row in encoded], device="cuda")


def expand_cell_mask(cell_mask: np.ndarray, boxes: list[tuple[int, int, int, int]], width: int, height: int) -> np.ndarray:
    output = np.zeros((height, width), dtype=bool)
    for index, selected in enumerate(cell_mask.reshape(-1)):
        if selected:
            left, top, right, bottom = boxes[index]
            output[top:bottom, left:right] = True
    return output


def top_cell_mask(values: np.ndarray, fraction: float, high: bool) -> np.ndarray:
    flat = values.reshape(-1)
    count = max(1, min(len(flat), int(round(len(flat) * fraction))))
    order = np.argsort(-flat if high else flat, kind="stable")
    selected = np.zeros(len(flat), dtype=bool)
    selected[order[:count]] = True
    return selected.reshape(values.shape)


def shifted_cell_masks(mask: np.ndarray, count: int, seed: int) -> list[np.ndarray]:
    rng = random.Random(seed)
    rows, columns = mask.shape
    output: list[np.ndarray] = []
    seen: set[tuple[int, int]] = set()
    while len(output) < count:
        dy, dx = rng.randrange(rows), rng.randrange(columns)
        if (dy, dx) in seen or (dy, dx) == (0, 0):
            continue
        seen.add((dy, dx))
        output.append(np.roll(mask, (dy, dx), axis=(0, 1)))
    return output


def perturb(image: Image.Image, base: Image.Image, mask: np.ndarray, mode: str) -> Image.Image:
    source = np.asarray(image.convert("RGB"), dtype=np.uint8)
    fill = np.asarray(base.convert("RGB"), dtype=np.uint8)
    values = source.copy() if mode == "deletion" else fill.copy()
    if mode == "deletion":
        values[mask] = fill[mask]
    elif mode == "retention":
        values[mask] = source[mask]
    else:
        raise ValueError(mode)
    return Image.fromarray(values, mode="RGB")


class OptionEvidenceModel:
    def __init__(self, model_path: Path, batch_size: int):
        self.model_path = model_path.resolve()
        self.batch_size = batch_size
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True, use_fast=False)
        self.processor.tokenizer.padding_side = "left"
        self.processor.image_processor.max_pixels = 65536
        self.processor.image_processor.min_pixels = 3136
        self.option_token_ids = option_ids(self.processor)
        self.image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager", low_cpu_mem_usage=True,
        ).to("cuda").eval()

    def score(self, prompt: str, images: list[Image.Image]) -> list[list[float]]:
        result: list[list[float]] = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start:start + self.batch_size]
            inputs = self.processor(text=[prompt] * len(batch), images=batch, return_tensors="pt", padding=True)
            inputs = {name: value.to("cuda") for name, value in inputs.items()}
            with torch.inference_mode():
                logits = self.model(**inputs, use_cache=False).logits[:, -1, :]
                values = torch.softmax(logits.index_select(1, self.option_token_ids).float(), dim=-1)
            result.extend(values.cpu().tolist())
        return result

    def attribute(self, prompt: str, image: Image.Image) -> dict[str, Any]:
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt")
        image_positions = (inputs["input_ids"][0] == self.image_token_id).nonzero().flatten()
        if not len(image_positions) or not torch.all(image_positions[1:] - image_positions[:-1] == 1):
            raise ValueError("image tokens are absent or non-contiguous")
        grid_t, grid_h_raw, grid_w_raw = map(int, inputs["image_grid_thw"][0])
        merge = int(self.processor.image_processor.merge_size)
        rows, columns = grid_h_raw // merge, grid_w_raw // merge
        if grid_t != 1 or rows * columns != len(image_positions):
            raise ValueError("visual-token count does not match image grid")
        device_inputs = {name: value.to("cuda") for name, value in inputs.items()}
        visual_positions = image_positions.to("cuda")
        output = self.model(**device_inputs, use_cache=False, output_attentions=True, return_dict=True)
        option_logits = output.logits[0, -1, self.option_token_ids].float()
        probabilities = torch.softmax(option_logits, dim=0).detach().cpu().tolist()
        layer_count = len(output.attentions)
        raw_layers: list[list[float]] = []
        option_maps: dict[str, list[list[list[float]]]] = {letter: [] for letter in OPTION_LETTERS}
        option_negative_maps: dict[str, list[list[list[float]]]] = {letter: [] for letter in OPTION_LETTERS}
        for attention in output.attentions:
            raw = attention[0, :, -1, :].mean(dim=0).index_select(0, visual_positions).float().detach().cpu().numpy()
            raw_layers.append(normalize(raw).reshape(rows, columns).tolist())
        for option_index, letter in enumerate(OPTION_LETTERS):
            score = option_margin(option_logits, option_index)
            gradients = torch.autograd.grad(score, output.attentions, retain_graph=option_index < 3, allow_unused=False)
            for attention, gradient in zip(output.attentions, gradients):
                contribution = (attention.float() * gradient.float()).mean(dim=1)[0, -1].index_select(0, visual_positions)
                positive = contribution.clamp(min=0).detach().cpu().numpy()
                negative = (-contribution).clamp(min=0).detach().cpu().numpy()
                option_maps[letter].append(normalize(positive).reshape(rows, columns).tolist())
                option_negative_maps[letter].append(normalize(negative).reshape(rows, columns).tolist())
        del output, device_inputs
        gc.collect()
        torch.cuda.empty_cache()
        return {"grid_rows": rows, "grid_columns": columns, "layer_count": layer_count,
                "baseline_option_probabilities": dict(zip(OPTION_LETTERS, probabilities)),
                "baseline_option_margins": dict(zip(OPTION_LETTERS, margins(probabilities))),
                "baseline_predicted_choice": OPTION_LETTERS[int(np.argmax(probabilities))],
                "raw_attention_layers": raw_layers, "positive_option_layers": option_maps,
                "negative_option_layers": option_negative_maps}


def evaluate_case(model: OptionEvidenceModel, case: dict[str, Any], layers: list[int], fraction: float, random_count: int) -> dict[str, Any]:
    image_path = Path(case["image"])
    if sha256_file(image_path) != case["image_sha256"]:
        raise ValueError(f"image hash mismatch for case {case['panel_index']}")
    image = Image.open(image_path).convert("RGB")
    prompt = model_prompt(model.processor, case["problem"])
    attribution = model.attribute(prompt, image)
    # The causal baseline must use the exact same no-attention inference path
    # as every intervention.  Scoring twice is a per-case determinism gate.
    baseline, baseline_repeat = model.score(prompt, [image, image])
    baseline_repeatability_max_abs_diff = float(
        np.max(np.abs(np.asarray(baseline) - np.asarray(baseline_repeat)))
    )
    base = Image.new("RGB", image.size, tuple(int(round(x)) for x in np.asarray(image).mean(axis=(0, 1))))
    boxes = grid_boxes(image.width, image.height, attribution["grid_rows"], attribution["grid_columns"])
    baseline_margins = margins(baseline)
    records: dict[str, Any] = {}
    sentinel_intervention_repeatability_max_abs_diff: float | None = None
    for layer in layers:
        layer_key = str(layer)
        records[layer_key] = {}
        for letter in OPTION_LETTERS:
            values = np.asarray(attribution["positive_option_layers"][letter][layer], dtype=np.float64)
            negative_values = np.asarray(attribution["negative_option_layers"][letter][layer], dtype=np.float64)
            positive_available = bool(float(values.sum()) > 0.0)
            negative_available = bool(float(negative_values.sum()) > 0.0)
            raw_cells = top_cell_mask(np.asarray(attribution["raw_attention_layers"][layer], dtype=np.float64), fraction, True)
            methods = {"raw_attention": raw_cells}
            positive_area = None
            if positive_available:
                high_cells = top_cell_mask(values, fraction, True)
                low_cells = top_cell_mask(values, fraction, False)
                controls = shifted_cell_masks(high_cells, random_count, int(hashlib.sha256(f"{case['source_record_sha256']}:{layer}:{letter}".encode()).hexdigest()[:16], 16))
                methods.update({"positive": high_cells, "low_positive": low_cells})
                methods.update({f"random_{i}": mask for i, mask in enumerate(controls)})
                positive_area = float(high_cells.mean())
            if negative_available:
                methods["negative"] = top_cell_mask(negative_values, fraction, True)
            records[layer_key][letter] = {
                "positive_available": positive_available,
                "negative_available": negative_available,
                "positive_area": positive_area,
                "methods": {},
            }
            for mode in ("deletion", "retention"):
                images = [perturb(image, base, expand_cell_mask(mask, boxes, image.width, image.height), mode) for mask in methods.values()]
                scored = model.score(prompt, images)
                if (
                    sentinel_intervention_repeatability_max_abs_diff is None
                    and mode == "deletion"
                    and "positive" in methods
                ):
                    sentinel_index = list(methods).index("positive")
                    repeated = model.score(prompt, [images[sentinel_index]])[0]
                    sentinel_intervention_repeatability_max_abs_diff = float(
                        np.max(
                            np.abs(
                                np.asarray(scored[sentinel_index])
                                - np.asarray(repeated)
                            )
                        )
                    )
                for method, probs in zip(methods, scored):
                    records[layer_key][letter]["methods"].setdefault(method, {})[mode] = {
                        "option_probabilities": dict(zip(OPTION_LETTERS, probs)),
                        "option_margins": dict(zip(OPTION_LETTERS, margins(probs))),
                    }
    return {"panel_index": case["panel_index"], "source_index": case["index"], "image": str(image_path),
            "image_sha256": case["image_sha256"], "target_choice": case["target_choice"],
            "attribution": attribution, "layers_evaluated": layers, "fraction": fraction,
            "baseline_option_probabilities": dict(zip(OPTION_LETTERS, baseline)),
            "baseline_option_margins": dict(zip(OPTION_LETTERS, baseline_margins)),
            "baseline_repeatability_max_abs_diff": baseline_repeatability_max_abs_diff,
            "sentinel_intervention_repeatability_max_abs_diff": sentinel_intervention_repeatability_max_abs_diff,
            "attribution_vs_causal_baseline_max_abs_probability_diff": float(
                np.max(np.abs(np.asarray(baseline) - np.asarray([
                    attribution["baseline_option_probabilities"][x] for x in OPTION_LETTERS
                ])))
            ),
            "baseline_predicted_choice": OPTION_LETTERS[int(np.argmax(baseline))],
            "baseline_correct": OPTION_LETTERS[int(np.argmax(baseline))] == case["target_choice"],
            "interventions": records}


def aggregate(records: list[dict[str, Any]], layers: list[int]) -> dict[str, Any]:
    def safe_mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    result: dict[str, Any] = {"case_count": len(records), "baseline_accuracy": float(np.mean([r["baseline_correct"] for r in records])), "by_layer_option": {}, "by_layer_role": {}}
    for layer in layers:
        result["by_layer_option"][str(layer)] = {}
        for letter in OPTION_LETTERS:
            drops = {"positive": [], "low_positive": [], "negative": [], "random": []}
            selectivity = {key: [] for key in drops}
            retention_advantage = []
            retention_selectivity = []
            for row in records:
                intervention = row["interventions"][str(layer)][letter]
                entry = intervention["methods"]
                original = np.asarray([row["baseline_option_margins"][x] for x in OPTION_LETTERS])
                row_random_drop = []
                row_random_selectivity = []
                row_random_retention = []
                row_random_retention_selectivity = []
                for method in entry:
                    deletion = np.asarray([entry[method]["deletion"]["option_margins"][x] for x in OPTION_LETTERS])
                    effect = original - deletion
                    key = "random" if method.startswith("random_") else method
                    drops.setdefault(key, []).append(float(effect[OPTION_LETTERS.index(letter)]))
                    value = float(effect[OPTION_LETTERS.index(letter)] - np.delete(effect, OPTION_LETTERS.index(letter)).mean())
                    if key == "random":
                        row_random_drop.append(float(effect[OPTION_LETTERS.index(letter)]))
                        row_random_selectivity.append(value)
                        retained = np.asarray([entry[method]["retention"]["option_margins"][x] for x in OPTION_LETTERS])
                        row_random_retention.append(float(retained[OPTION_LETTERS.index(letter)]))
                        row_random_retention_selectivity.append(float(retained[OPTION_LETTERS.index(letter)] - np.delete(retained, OPTION_LETTERS.index(letter)).mean()))
                    else:
                        selectivity.setdefault(key, []).append(value)
                if row_random_drop:
                    selectivity["random"].append(float(np.mean(row_random_selectivity)))
                    positive_retained = np.asarray([entry["positive"]["retention"]["option_margins"][x] for x in OPTION_LETTERS])
                    retention_advantage.append(float(positive_retained[OPTION_LETTERS.index(letter)] - np.mean(row_random_retention)))
                    retention_selectivity.append(float((positive_retained[OPTION_LETTERS.index(letter)] - np.delete(positive_retained, OPTION_LETTERS.index(letter)).mean()) - np.mean(row_random_retention_selectivity)))
            result["by_layer_option"][str(layer)][letter] = {
                "positive_available_case_count": len(drops["positive"]),
                "negative_available_case_count": len(drops["negative"]),
                "positive_mean_margin_drop": safe_mean(drops["positive"]),
                "low_positive_mean_margin_drop": safe_mean(drops["low_positive"]),
                "negative_mean_margin_drop": safe_mean(drops["negative"]),
                "raw_attention_mean_margin_drop": safe_mean(drops.get("raw_attention", [])),
                "random_mean_margin_drop": safe_mean(drops["random"]),
                "positive_selectivity": safe_mean(selectivity["positive"]),
                "raw_attention_selectivity": safe_mean(selectivity.get("raw_attention", [])),
                "random_selectivity": safe_mean(selectivity["random"]),
                "positive_beats_random_rate": float(np.mean(np.asarray(selectivity["positive"]) > np.asarray(selectivity["random"]))) if selectivity["positive"] else None,
                "positive_retention_margin_vs_random": safe_mean(retention_advantage),
                "positive_retention_selectivity_vs_random": safe_mean(retention_selectivity),
            }
        role_values = {"ground_truth": [], "model_prediction": [], "other_options": []}
        for row in records:
            target = row["target_choice"]
            predicted = row["baseline_predicted_choice"]
            for role, choices in (("ground_truth", [target]), ("model_prediction", [predicted]),
                                  ("other_options", [x for x in OPTION_LETTERS if x != target])):
                per_choice = []
                for letter in choices:
                    entry = row["interventions"][str(layer)][letter]["methods"]
                    if "positive" not in entry or not any(name.startswith("random_") for name in entry):
                        continue
                    index = OPTION_LETTERS.index(letter)
                    original = np.asarray([row["baseline_option_margins"][x] for x in OPTION_LETTERS])
                    positive = np.asarray([entry["positive"]["deletion"]["option_margins"][x] for x in OPTION_LETTERS])
                    random = np.mean([np.asarray([entry[name]["deletion"]["option_margins"][x] for x in OPTION_LETTERS]) for name in entry if name.startswith("random_")], axis=0)
                    positive_effect, random_effect = original - positive, original - random
                    per_choice.append({"drop": float(positive_effect[index]), "selectivity": float(positive_effect[index] - np.delete(positive_effect, index).mean()),
                                       "random_drop": float(random_effect[index]), "random_selectivity": float(random_effect[index] - np.delete(random_effect, index).mean())})
                if per_choice:
                    role_values[role].append({key: float(np.mean([v[key] for v in per_choice])) for key in per_choice[0]})
        result["by_layer_role"][str(layer)] = {
            role: ({"available_case_count": len(values)}
                  | {f"positive_{key}": safe_mean([v[key] for v in values]) for key in ("drop", "selectivity")}
                  | {f"random_{key}": safe_mean([v[f"random_{key}"] for v in values]) for key in ("drop", "selectivity")})
            for role, values in role_values.items()
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--expected-panel-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-index", type=int, action="append")
    parser.add_argument("--layers", default="0,4,8,12,16,20,24,27")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--random-count", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite run: {args.output_dir}")
    if args.resume and not args.output_dir.exists():
        raise FileNotFoundError(f"resume output directory does not exist: {args.output_dir}")
    panel_sha = sha256_file(args.panel)
    if panel_sha != args.expected_panel_sha256:
        raise ValueError(f"panel SHA-256 mismatch: {panel_sha}")
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    if panel.get("status") != "frozen_before_model_visualization_outputs" or panel.get("selection_uses_model_outputs") is not False:
        raise ValueError("panel is not outcome-blind and frozen")
    if panel.get("case_count", 0) < 48 or len(panel.get("cases", [])) != panel.get("case_count"):
        raise ValueError("confirmation panel must contain at least 48 cases")
    if not 0 < args.fraction < 1 or args.random_count < 1:
        raise ValueError("fraction must be in (0,1), random-count positive")
    selected = set(args.panel_index or range(panel["case_count"]))
    cases = [row for row in panel["cases"] if row["panel_index"] in selected]
    if len(cases) != len(selected):
        raise ValueError("requested panel index is missing")
    if args.batch_size != 1:
        raise ValueError(
            "option-conditioned causal scoring requires --batch-size 1; "
            "BF16 multimodal scores changed with batch position in the smoke gate"
        )
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    case_path = args.output_dir / "case_results.jsonl"
    metrics_path = args.output_dir / "metrics.json"
    if args.resume and metrics_path.exists():
        raise FileExistsError(f"run already has final metrics: {metrics_path}")
    if args.resume and not case_path.exists():
        raise FileNotFoundError(f"resume case file does not exist: {case_path}")
    records = load_partial_records(case_path, cases) if args.resume else []
    completed = {row["panel_index"] for row in records}
    remaining_cases = [row for row in cases if row["panel_index"] not in completed]
    model = OptionEvidenceModel(args.model, args.batch_size)
    model_layer_count = int(getattr(model.model.config, "num_hidden_layers", 28))
    layers = parse_layers(args.layers, model_layer_count)
    with case_path.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for case in remaining_cases:
            row = evaluate_case(model, case, layers, args.fraction, args.random_count)
            records.append(row)
            handle.write(canonical_json(row) + "\n"); handle.flush()
            print(canonical_json({"completed": len(records), "total": len(cases), "panel_index": case["panel_index"]}), flush=True)
    records.sort(key=lambda row: row["panel_index"])
    layers = parse_layers(args.layers, records[0]["attribution"]["layer_count"])
    metrics = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
               "status": "completed_confirmation" if len(cases) >= 48 else "completed_smoke", "formal_result": False,
               "method": "option_conditioned_gradient_attention_counterfactual_v1",
               "claim_boundary": "option_support_requires_selective_margin_drop_over_controls",
               "model_label": args.model_label, "model_path": str(args.model.resolve()),
               "model_config_sha256": sha256_file(args.model / "config.json"),
               "panel_path": str(args.panel.resolve()), "panel_sha256": panel_sha,
               "panel_indices": [row["panel_index"] for row in cases], "layers_evaluated": layers,
               "fraction": args.fraction, "random_count": args.random_count,
               "score_batch_size": args.batch_size,
               "max_baseline_repeatability_abs_probability_diff": max(
                   row["baseline_repeatability_max_abs_diff"] for row in records
               ),
               "max_sentinel_intervention_repeatability_abs_probability_diff": max(
                   row["sentinel_intervention_repeatability_max_abs_diff"]
                   for row in records
                   if row["sentinel_intervention_repeatability_max_abs_diff"] is not None
               ),
               "max_attribution_vs_causal_baseline_abs_probability_diff": max(
                   row["attribution_vs_causal_baseline_max_abs_probability_diff"] for row in records
               ),
               "case_results": str(case_path.resolve()), "case_results_sha256": sha256_file(case_path),
               **aggregate(records, layers)}
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(canonical_json(metrics), flush=True)


if __name__ == "__main__":
    main()
