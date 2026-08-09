#!/usr/bin/env python3
"""Probe whether a PathVLM checkpoint can directly self-report evidence boxes.

This is a self-report localization diagnostic, not ground truth or a causal
explanation. External-model boxes are used only after generation for comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


EVIDENCE_TYPES = {"focal", "multifocal", "diffuse", "not_localizable"}
DEFAULT_CASES = (0, 1, 2, 4, 9, 10, 20, 22)
PROMPT = """{problem}

Answer the multiple-choice question from the image, then identify only the image region(s) that
directly support that answer. Coordinates must be normalized to the full original image on a
0-1000 scale: x increases left-to-right and y increases top-to-bottom.

If the evidence is a focal region, use evidence_type \"focal\". If several separate regions are
needed, use \"multifocal\" and at most four boxes. If evidence is distributed across most of the
image, use \"diffuse\" with no boxes. If the answer cannot be spatially localized, use
\"not_localizable\" with no boxes. Do not invent a box for diffuse or absent evidence.

Return exactly one JSON object and no Markdown or other text. The object must have exactly these
fields: `answer`, `evidence_type`, `boxes`, and `reason`. Each item in `boxes` must have exactly
`x_min`, `y_min`, `x_max`, `y_max`, and `label`. Use actual image-derived coordinates, never
placeholder coordinates. Do not use a whole-image box unless a smaller region is genuinely
impossible; in that situation prefer `diffuse` with an empty box list.
"""
SYSTEM_PROMPT = (
    "You are a deterministic visual grounding engine. Follow the requested JSON schema exactly. "
    "Never emit think/answer XML tags, Markdown, or prose outside the JSON object."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def answer_from_completion(text: str) -> str | None:
    match = re.search(r"<answer>\s*([ABCD])\b", text, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def frozen_answers(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    output: dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            answer = row.get("predicted_choice") or answer_from_completion(row.get("completion", ""))
            if answer is None or answer not in "ABCD":
                raise ValueError(f"no fixed answer for panel index {row.get('panel_index')}")
            output[int(row["panel_index"])] = answer
    return output


def extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip() if len(lines) >= 3 else stripped
    decoder = json.JSONDecoder()
    starts = [index for index, char in enumerate(stripped) if char == "{"]
    errors = []
    for start in starts:
        try:
            value, _ = decoder.raw_decode(stripped[start:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError as exc:
            errors.append(str(exc))
    raise ValueError(f"no JSON object found: {errors[:2]}")


def validate_annotation(value: dict[str, Any]) -> dict[str, Any]:
    answer = str(value.get("answer", "")).strip().upper()[:1]
    evidence_type = str(value.get("evidence_type", "")).strip().lower()
    boxes = value.get("boxes")
    reason = value.get("reason")
    if answer not in "ABCD":
        raise ValueError("answer must be A/B/C/D")
    if evidence_type not in EVIDENCE_TYPES:
        raise ValueError("invalid evidence_type")
    if not isinstance(boxes, list) or len(boxes) > 4:
        raise ValueError("boxes must be a list of at most four")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason is absent")
    spatial = evidence_type in {"focal", "multifocal"}
    if spatial != bool(boxes):
        raise ValueError("spatial evidence requires boxes and non-spatial evidence forbids boxes")
    normalized = []
    for box in boxes:
        if not isinstance(box, dict):
            raise ValueError("box must be an object")
        coordinates = []
        for key in ("x_min", "y_min", "x_max", "y_max"):
            coordinate = box.get(key)
            if not isinstance(coordinate, (int, float)) or isinstance(coordinate, bool):
                raise ValueError(f"invalid coordinate {key}")
            coordinates.append(float(coordinate))
        x_min, y_min, x_max, y_max = coordinates
        if not (0 <= x_min < x_max <= 1000 and 0 <= y_min < y_max <= 1000):
            raise ValueError("coordinates must be ordered within 0..1000")
        normalized.append(
            {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "label": str(box.get("label", "")).strip(),
            }
        )
    return {
        "answer": answer,
        "evidence_type": evidence_type,
        "boxes": normalized,
        "reason": reason.strip(),
    }


def union_mask(boxes: list[dict[str, Any]], size: int = 1000) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for box in boxes:
        x0 = max(0, min(size, int(np.floor(float(box["x_min"]) * size / 1000))))
        y0 = max(0, min(size, int(np.floor(float(box["y_min"]) * size / 1000))))
        x1 = max(0, min(size, int(np.ceil(float(box["x_max"]) * size / 1000))))
        y1 = max(0, min(size, int(np.ceil(float(box["y_max"]) * size / 1000))))
        mask[y0:y1, x0:x1] = True
    return mask


def overlap_metrics(predicted: list[dict[str, Any]], reference: list[dict[str, Any]]) -> dict[str, float]:
    pred = union_mask(predicted)
    ref = union_mask(reference)
    intersection = int(np.logical_and(pred, ref).sum())
    union = int(np.logical_or(pred, ref).sum())
    precision = intersection / max(1, int(pred.sum()))
    recall = intersection / max(1, int(ref.sum()))
    center_hits = []
    for box in predicted:
        x = min(999, max(0, int(round((box["x_min"] + box["x_max"]) / 2))))
        y = min(999, max(0, int(round((box["y_min"] + box["y_max"]) / 2))))
        center_hits.append(bool(ref[y, x]))
    return {
        "union_iou": intersection / max(1, union),
        "predicted_area_precision": precision,
        "reference_area_recall": recall,
        "any_predicted_center_hit": float(any(center_hits)),
        "predicted_area_fraction": float(pred.mean()),
        "reference_area_fraction": float(ref.mean()),
    }


def render_overlay(
    image_path: Path,
    predicted: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    output_path: Path,
) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for boxes, color, prefix in ((reference, "red", "REF"), (predicted, "cyan", "MODEL")):
        for index, box in enumerate(boxes):
            xy = (
                box["x_min"] * width / 1000,
                box["y_min"] * height / 1000,
                box["x_max"] * width / 1000,
                box["y_max"] * height / 1000,
            )
            draw.rectangle(xy, outline=color, width=max(2, min(width, height) // 250))
            draw.text((xy[0] + 2, xy[1] + 2), f"{prefix}{index + 1}", fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["format_valid"]]
    payload_valid = [row for row in rows if row.get("bbox_payload_valid", False)]
    spatial_reference = [row for row in rows if row["reference_spatial"]]
    nonspatial_reference = [row for row in rows if not row["reference_spatial"]]
    comparable = [row for row in spatial_reference if row.get("overlap")]
    return {
        "case_count": len(rows),
        "format_valid_count": len(valid),
        "format_valid_rate": len(valid) / len(rows),
        "bbox_payload_valid_count": len(payload_valid),
        "bbox_payload_valid_rate": len(payload_valid) / len(rows),
        "answer_accuracy": sum(row.get("answer_correct", False) for row in rows) / len(rows),
        "answer_accuracy_among_valid_payloads": sum(
            row.get("answer_correct", False) for row in payload_valid
        ) / max(1, len(payload_valid)),
        "spatial_applicability_accuracy": sum(
            row.get("predicted_spatial") == row["reference_spatial"] for row in rows
        ) / len(rows),
        "nonspatial_abstention_rate": sum(
            row.get("bbox_payload_valid", False)
            and not row.get("predicted_spatial", False)
            for row in nonspatial_reference
        ) / max(1, len(nonspatial_reference)),
        "spatial_box_output_rate": sum(
            row.get("predicted_spatial", False) for row in spatial_reference
        ) / max(1, len(spatial_reference)),
        "spatial_comparable_count": len(comparable),
        "mean_union_iou": float(np.mean([row["overlap"]["union_iou"] for row in comparable]))
        if comparable
        else None,
        "mean_predicted_area_precision": float(
            np.mean([row["overlap"]["predicted_area_precision"] for row in comparable])
        )
        if comparable
        else None,
        "mean_reference_area_recall": float(
            np.mean([row["overlap"]["reference_area_recall"] for row in comparable])
        )
        if comparable
        else None,
        "predicted_center_hit_rate": float(
            np.mean([row["overlap"]["any_predicted_center_hit"] for row in comparable])
        )
        if comparable
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-index", type=int, action="append")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--force-json-prefix", action="store_true")
    parser.add_argument("--fixed-answer-predictions", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    references = json.loads(args.references.read_text(encoding="utf-8"))
    panel_by_index = {row["panel_index"]: row for row in panel["cases"]}
    reference_by_index = {row["panel_index"]: row for row in references["results"]}
    selected = tuple(args.panel_index or DEFAULT_CASES)
    fixed_answer_by_index = frozen_answers(args.fixed_answer_predictions)
    if any(index not in panel_by_index or index not in reference_by_index for index in selected):
        raise ValueError("selected case is absent from panel or references")
    args.output_dir.mkdir(parents=True)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, use_fast=False)
    processor.image_processor.max_pixels = 65536
    processor.image_processor.min_pixels = 3136
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    rows = []
    prediction_path = args.output_dir / "predictions.jsonl"
    for count, index in enumerate(selected, 1):
        case = panel_by_index[index]
        reference_record = reference_by_index[index]
        reference = reference_record["annotation"]
        with Image.open(case["image"]) as source:
            image = source.convert("RGB")
        fixed_answer = fixed_answer_by_index.get(index)
        prompt_text = PROMPT.format(problem=case["problem"])
        if fixed_answer:
            prompt_text += (
                f"\nThe model's decision was already frozen as option {fixed_answer}. Do not "
                "reconsider or change that answer. Only localize the visual evidence the model "
                "would use to support this fixed decision."
            )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ]},
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if args.force_json_prefix and fixed_answer:
            assistant_prefix = f'{{"answer":"{fixed_answer}","evidence_type":"'
        else:
            assistant_prefix = '{"answer":"' if args.force_json_prefix else ""
        prompt += assistant_prefix
        inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
        inputs = {name: value.to("cuda") for name, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True
            )
        completion_ids = generated[:, inputs["input_ids"].shape[1]:]
        completion = assistant_prefix + processor.batch_decode(
            completion_ids, skip_special_tokens=True
        )[0].strip()
        row: dict[str, Any] = {
            "panel_index": index,
            "image": case["image"],
            "image_sha256": case["image_sha256"],
            "target_choice": case["target_choice"],
            "fixed_answer": fixed_answer,
            "reference_evidence_type": reference["evidence_type"],
            "reference_spatial": reference["evidence_type"] in {"focal", "multifocal"},
            "reference_boxes": reference["boxes"],
            "completion": completion,
            "generated_token_count": int(completion_ids.shape[1]),
            "format_valid": False,
            "bbox_payload_valid": False,
        }
        try:
            parsed = extract_json(completion)
            annotation = validate_annotation(parsed)
            row.update(
                {
                    "format_valid": True,
                    "bbox_payload_valid": True,
                    "annotation": annotation,
                    "answer_correct": annotation["answer"] == case["target_choice"],
                    "predicted_spatial": annotation["evidence_type"] in {"focal", "multifocal"},
                    "evidence_type_exact": annotation["evidence_type"] == reference["evidence_type"],
                }
            )
            if row["reference_spatial"] and row["predicted_spatial"]:
                row["overlap"] = overlap_metrics(annotation["boxes"], reference["boxes"])
            render_overlay(
                Path(case["image"]), annotation["boxes"], reference["boxes"],
                args.output_dir / "overlays" / f"case_{index:02d}.png",
            )
        except Exception as exc:
            row["validation_error"] = f"{type(exc).__name__}: {exc}"
            try:
                parsed = extract_json(completion)
                repaired = dict(parsed)
                repair = []
                if not isinstance(repaired.get("reason"), str) or not repaired["reason"].strip():
                    labels = [
                        str(box.get("label", "")).strip()
                        for box in repaired.get("boxes", [])
                        if isinstance(box, dict) and str(box.get("label", "")).strip()
                    ]
                    if labels:
                        repaired["reason"] = "; ".join(labels)
                        repair.append("reason_from_box_labels")
                annotation = validate_annotation(repaired)
                row.update(
                    {
                        "bbox_payload_valid": True,
                        "payload_repair": repair,
                        "annotation": annotation,
                        "answer_correct": annotation["answer"] == case["target_choice"],
                        "predicted_spatial": annotation["evidence_type"]
                        in {"focal", "multifocal"},
                        "evidence_type_exact": annotation["evidence_type"]
                        == reference["evidence_type"],
                    }
                )
                if row["reference_spatial"] and row["predicted_spatial"]:
                    row["overlap"] = overlap_metrics(
                        annotation["boxes"], reference["boxes"]
                    )
                render_overlay(
                    Path(case["image"]),
                    annotation["boxes"],
                    reference["boxes"],
                    args.output_dir / "overlays" / f"case_{index:02d}.png",
                )
            except Exception as salvage_exc:
                row["payload_validation_error"] = (
                    f"{type(salvage_exc).__name__}: {salvage_exc}"
                )
        rows.append(row)
        with prediction_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(json.dumps({"completed": count, "total": len(selected), "panel_index": index, "valid": row["format_valid"]}), flush=True)
    metrics = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_smoke",
        "formal_result": False,
        "method": "direct_bbox_self_report_v1",
        "claim_boundary": "model_self_report_not_ground_truth_or_causal_explanation",
        "model_label": args.model_label,
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "panel_path": str(args.panel.resolve()),
        "panel_sha256": sha256_file(args.panel),
        "reference_path": str(args.references.resolve()),
        "reference_sha256": sha256_file(args.references),
        "panel_indices": list(selected),
        "max_new_tokens": args.max_new_tokens,
        "force_json_prefix": args.force_json_prefix,
        "fixed_answer_predictions": str(args.fixed_answer_predictions.resolve())
        if args.fixed_answer_predictions
        else None,
        "do_sample": False,
        "test_accessed": False,
        **aggregate(rows),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
