#!/usr/bin/env python3
"""Compare bbox encodings that preserve the trained think/answer envelope."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from run_direct_bbox_self_report import (
    answer_from_completion,
    overlap_metrics,
    render_overlay,
    sha256_file,
)


FORMATS = {
    "xml_bbox_in_think": """
Keep the familiar output envelope `<think>...</think><answer>...</answer>`. Inside `<think>`, give
concise image-grounded reasoning followed by exactly one `<evidence type="TYPE">...</evidence>`
element. TYPE must be focal, multifocal, diffuse, or not_localizable. For focal or multifocal
evidence, put at most four self-closing bbox elements inside it. Every bbox must have x_min, y_min,
x_max, y_max, and label attributes. Coordinates are actual integer locations on the full image in
the 0-1000 coordinate system. For diffuse or not_localizable evidence, leave the evidence element
empty. Do not use placeholder words or a whole-image box. The final `<answer>` contains exactly one
letter A, B, C, or D.
""",
    "qwen_ref_box_in_think": """
Keep the familiar output envelope `<think>...</think><answer>...</answer>`. Inside `<think>`, first
give concise image-grounded reasoning and state `evidence_type=focal`, `multifocal`, `diffuse`, or
`not_localizable`. For focal or multifocal evidence, append at most four Qwen grounding pairs in
this syntax: `<ref>short finding</ref><box>(x_min,y_min),(x_max,y_max)</box>`. Replace every coordinate
name with an actual integer from 0 to 1000 relative to the full image. For diffuse or
not_localizable evidence, output no `<box>`. The final `<answer>` contains exactly one letter.
""",
    "plain_bbox_in_think": """
Keep the familiar output envelope `<think>...</think><answer>...</answer>`. At the end of `<think>`,
write one line beginning `VISUAL_EVIDENCE:` followed by the evidence type and zero to four actual
normalized boxes in `[x_min,y_min,x_max,y_max]` form. Coordinates are integers from 0 to 1000. Use
`VISUAL_EVIDENCE: diffuse; boxes=[]` or `VISUAL_EVIDENCE: not_localizable; boxes=[]` when a local
box is inappropriate. Do not use placeholders or a whole-image box. The final `<answer>` contains
exactly one option letter.
""",
    "evidence_json_in_think": """
Keep the familiar output envelope `<think>...</think><answer>...</answer>`. Inside `<think>`, give
concise reasoning and then exactly one `<evidence_json>` element containing strict JSON with keys
`evidence_type` and `boxes`. Each box has integer x_min, y_min, x_max, y_max and a label, using
actual 0-1000 full-image coordinates. Focal/multifocal evidence needs one to four boxes;
diffuse/not_localizable evidence needs an empty list. Do not use placeholders or a whole-image
box. The final `<answer>` contains exactly one option letter.
""",
}


def baseline_answers(path: Path) -> dict[int, str]:
    output = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            answer = row.get("predicted_choice") or answer_from_completion(row.get("completion", ""))
            if answer:
                output[int(row["panel_index"])] = answer
    return output


def numbers(box: tuple[str, str, str, str]) -> dict[str, float] | None:
    values = [float(value) for value in box]
    x_min, y_min, x_max, y_max = values
    if not (0 <= x_min < x_max <= 1000 and 0 <= y_min < y_max <= 1000):
        return None
    return dict(zip(("x_min", "y_min", "x_max", "y_max"), values))


def extract_boxes(text: str, format_name: str) -> tuple[str | None, list[dict[str, Any]]]:
    lower = text.lower()
    evidence_type = None
    patterns = (
        r'<evidence\s+type=["\'](focal|multifocal|diffuse|not_localizable)["\']',
        r'evidence_type\s*=\s*(focal|multifocal|diffuse|not_localizable)',
        r'visual_evidence\s*:\s*(focal|multifocal|diffuse|not_localizable)',
        r'"evidence_type"\s*:\s*"(focal|multifocal|diffuse|not_localizable)"',
    )
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            evidence_type = match.group(1)
            break
    boxes: list[dict[str, Any]] = []
    if format_name == "xml_bbox_in_think":
        pattern = (
            r'<bbox\b[^>]*x_min=["\'](\d+(?:\.\d+)?)["\'][^>]*'
            r'y_min=["\'](\d+(?:\.\d+)?)["\'][^>]*'
            r'x_max=["\'](\d+(?:\.\d+)?)["\'][^>]*'
            r'y_max=["\'](\d+(?:\.\d+)?)["\'][^>]*'
            r'(?:label=["\']([^"\']*)["\'])?[^>]*/?>'
        )
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            box = numbers(match.groups()[:4])
            if box:
                box["label"] = match.group(5) or ""
                boxes.append(box)
    elif format_name == "qwen_ref_box_in_think":
        pattern = r'<box>\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)\s*,\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)\s*</box>'
        refs = re.findall(r'<ref>(.*?)</ref>', text, flags=re.IGNORECASE | re.DOTALL)
        for index, match in enumerate(re.finditer(pattern, text, flags=re.IGNORECASE)):
            box = numbers(match.groups())
            if box:
                box["label"] = refs[index].strip() if index < len(refs) else ""
                boxes.append(box)
    elif format_name == "plain_bbox_in_think":
        for match in re.finditer(
            r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]',
            text,
        ):
            box = numbers(match.groups())
            if box:
                box["label"] = ""
                boxes.append(box)
    else:
        match = re.search(r'<evidence_json>\s*(\{.*?\})\s*</evidence_json>', text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            try:
                payload = json.loads(match.group(1))
                evidence_type = str(payload.get("evidence_type", evidence_type)).lower()
                for item in payload.get("boxes", []):
                    box = numbers(tuple(str(item[key]) for key in ("x_min", "y_min", "x_max", "y_max")))
                    if box:
                        box["label"] = str(item.get("label", ""))
                        boxes.append(box)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
    return evidence_type, boxes[:4]


def valid_payload(evidence_type: str | None, boxes: list[dict[str, Any]]) -> bool:
    if evidence_type in {"focal", "multifocal"}:
        return 1 <= len(boxes) <= 4
    if evidence_type in {"diffuse", "not_localizable"}:
        return len(boxes) == 0
    return False


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {"by_format": {}}
    for format_name in FORMATS:
        selected = [row for row in rows if row["format"] == format_name]
        valid = [row for row in selected if row["bbox_payload_valid"]]
        comparable = [row for row in valid if row.get("overlap")]
        output["by_format"][format_name] = {
            "case_count": len(selected),
            "answer_extracted_rate": float(np.mean([row["answer"] is not None for row in selected])),
            "answer_preservation_rate": float(np.mean([row["answer"] == row["baseline_answer"] for row in selected])),
            "answer_accuracy": float(np.mean([row["answer"] == row["target_choice"] for row in selected])),
            "bbox_payload_valid_rate": len(valid) / len(selected),
            "spatial_applicability_accuracy": float(np.mean([
                row["bbox_payload_valid"] and row["predicted_spatial"] == row["reference_spatial"]
                for row in selected
            ])),
            "comparable_count": len(comparable),
            "mean_union_iou": float(np.mean([row["overlap"]["union_iou"] for row in comparable])) if comparable else None,
            "center_hit_rate": float(np.mean([row["overlap"]["any_predicted_center_hit"] for row in comparable])) if comparable else None,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-index", type=int, action="append")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    panel = json.loads(args.panel.read_text())
    refs = json.loads(args.references.read_text())
    cases = {row["panel_index"]: row for row in panel["cases"]}
    ref_by_index = {row["panel_index"]: row for row in refs["results"]}
    baselines = baseline_answers(args.baseline_predictions)
    selected = tuple(args.panel_index or (0, 1, 2))
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, use_fast=False)
    processor.image_processor.max_pixels = 65536
    processor.image_processor.min_pixels = 3136
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to("cuda").eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    rows = []
    path = args.output_dir / "predictions.jsonl"
    for format_name, instruction in FORMATS.items():
        for index in selected:
            case = cases[index]
            reference = ref_by_index[index]["annotation"]
            with Image.open(case["image"]) as source:
                image = source.convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": case["problem"] + "\n\n" + instruction},
            ]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
            inputs = {name: value.to("cuda") for name, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True
                )
            completion_ids = generated[:, inputs["input_ids"].shape[1]:]
            completion = processor.batch_decode(completion_ids, skip_special_tokens=True)[0].strip()
            answer = answer_from_completion(completion)
            evidence_type, boxes = extract_boxes(completion, format_name)
            payload_valid = valid_payload(evidence_type, boxes)
            reference_spatial = reference["evidence_type"] in {"focal", "multifocal"}
            predicted_spatial = evidence_type in {"focal", "multifocal"}
            row: dict[str, Any] = {
                "panel_index": index, "format": format_name, "image": case["image"],
                "target_choice": case["target_choice"], "baseline_answer": baselines.get(index),
                "answer": answer, "answer_preserved": answer == baselines.get(index),
                "evidence_type": evidence_type, "boxes": boxes,
                "bbox_payload_valid": payload_valid, "reference_spatial": reference_spatial,
                "predicted_spatial": predicted_spatial, "completion": completion,
            }
            if payload_valid and reference_spatial and predicted_spatial:
                row["overlap"] = overlap_metrics(boxes, reference["boxes"])
            if payload_valid:
                render_overlay(
                    Path(case["image"]), boxes, reference["boxes"],
                    args.output_dir / "overlays" / format_name / f"case_{index:02d}.png",
                )
            rows.append(row)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            print(json.dumps({"format": format_name, "panel_index": index, "answer": answer, "bbox_valid": payload_valid}), flush=True)
    metrics = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_smoke", "formal_result": False,
        "method": "think_answer_bbox_output_format_probe_v1",
        "model_label": args.model_label, "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "panel_indices": list(selected), "formats": list(FORMATS),
        "test_accessed": False, **summarize(rows),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
