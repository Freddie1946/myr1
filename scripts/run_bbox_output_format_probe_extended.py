#!/usr/bin/env python3
"""Extended bbox-format search with tag nesting, attributes, prefixes, and follow-ups."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from run_bbox_output_format_probe import numbers
from run_direct_bbox_self_report import (
    answer_from_completion,
    overlap_metrics,
    render_overlay,
    sha256_file,
)


@dataclass(frozen=True)
class FormatSpec:
    name: str
    instruction: str
    assistant_prefix: str = ""
    follow_up: bool = False


SPECS = (
    FormatSpec(
        "prefill_plain_inside_think",
        "Use `<think>VISUAL_EVIDENCE: TYPE; boxes=[[x1,y1,x2,y2]]; reasoning...</think><answer>A</answer>`. "
        "TYPE is focal/multifocal/diffuse/not_localizable; coordinates are actual 0-1000 integers. "
        "Diffuse or non-localizable evidence uses boxes=[]. The answer is exactly one letter.",
        "<think>VISUAL_EVIDENCE: ",
    ),
    FormatSpec(
        "prefill_xml_inside_think",
        "Begin `<think>` with `<evidence type=\"TYPE\">`. For focal/multifocal evidence put one to "
        "four `<bbox x_min=\"N\" y_min=\"N\" x_max=\"N\" y_max=\"N\"/>` elements inside, then "
        "continue reasoning and close think. Diffuse/not_localizable uses no bbox. End with a one-letter answer.",
        "<think><evidence type=\"",
    ),
    FormatSpec(
        "prefill_qwen_box_inside_think",
        "Begin `<think>` with a Qwen grounding pair `<ref>finding</ref><box>(x1,y1),(x2,y2)</box>` "
        "using actual 0-1000 coordinates, then reason and end with `<answer>A</answer>`. If no local "
        "box is appropriate, begin with `<ref>not_localizable</ref><box>none</box>`.",
        "<think><ref>",
    ),
    FormatSpec(
        "bbox_nested_in_answer",
        "Return `<think>reasoning</think><answer><choice>A</choice><evidence_type>TYPE</evidence_type>" 
        "<bbox>[x1,y1,x2,y2]</bbox></answer>`. Up to four bbox tags are allowed. Use no bbox for "
        "diffuse/not_localizable evidence. Coordinates are actual 0-1000 integers.",
    ),
    FormatSpec(
        "answer_tag_attributes",
        "Return `<think>reasoning</think><answer choice=\"A\" evidence_type=\"TYPE\" "
        "bbox=\"[x1,y1,x2,y2]\"/>`. Use `bbox=\"[]\"` for diffuse/not_localizable evidence. "
        "Replace placeholders with the actual option, type and 0-1000 image coordinates.",
    ),
    FormatSpec(
        "think_tag_attributes",
        "Return `<think evidence_type=\"TYPE\" bbox=\"[x1,y1,x2,y2]\">reasoning</think>" 
        "<answer>A</answer>`. Use bbox=\"[]\" for diffuse/not_localizable evidence. Replace every "
        "placeholder with actual values and 0-1000 image coordinates.",
    ),
    FormatSpec(
        "second_turn_bbox_tag_prefill",
        "The preceding answer is frozen. Do not answer again. Return only localization as "
        "`<bbox evidence_type=\"TYPE\">[x1,y1,x2,y2]</bbox>`. Use [] for diffuse or "
        "not_localizable. Coordinates are actual 0-1000 integers.",
        "<bbox evidence_type=\"",
        True,
    ),
    FormatSpec(
        "second_turn_plain_prefill",
        "The preceding answer is frozen. Do not answer again. Return only `VISUAL_EVIDENCE: "
        "TYPE; boxes=[[x1,y1,x2,y2]]` with actual 0-1000 coordinates. Use boxes=[] for diffuse "
        "or not_localizable evidence.",
        "VISUAL_EVIDENCE: ",
        True,
    ),
)


def load_baselines(path: Path) -> dict[int, dict[str, str]]:
    output = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            completion = row.get("completion", "")
            answer = row.get("predicted_choice") or answer_from_completion(completion)
            if answer:
                output[int(row["panel_index"])] = {"answer": answer, "completion": completion}
    return output


def extract_answer(text: str, fallback: str | None) -> str | None:
    answer = answer_from_completion(text)
    if answer:
        return answer
    for pattern in (
        r'<choice>\s*([ABCD])\b',
        r'<answer\b[^>]*\bchoice=["\']?([ABCD])\b',
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return fallback


def extract_payload(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    lower = text.lower()
    evidence_type = None
    for pattern in (
        r'evidence_type\s*=\s*["\']?(focal|multifocal|diffuse|not_localizable)',
        r'<evidence_type>\s*(focal|multifocal|diffuse|not_localizable)',
        r'<evidence\s+type\s*=\s*["\']?(focal|multifocal|diffuse|not_localizable)',
        r'visual_evidence\s*:\s*(focal|multifocal|diffuse|not_localizable)',
        r'<ref>\s*(not_localizable|diffuse)\s*</ref>',
    ):
        match = re.search(pattern, lower)
        if match:
            evidence_type = match.group(1)
            break

    raw_boxes: list[tuple[str, str, str, str]] = []
    patterns = (
        r'<bbox\b[^>]*x_min=["\']?(\d+(?:\.\d+)?)["\']?[^>]*y_min=["\']?(\d+(?:\.\d+)?)["\']?[^>]*x_max=["\']?(\d+(?:\.\d+)?)["\']?[^>]*y_max=["\']?(\d+(?:\.\d+)?)["\']?[^>]*>',
        r'<bbox\b[^>]*>\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]\s*</bbox>',
        r'<box>\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)\s*,\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)\s*</box>',
        r'(?:bbox|boxes)\s*=\s*["\']?\[\s*\[?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]?',
        r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]',
    )
    for pattern in patterns:
        raw_boxes.extend(match.groups() for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    boxes = []
    seen = set()
    for raw in raw_boxes:
        box = numbers(raw)
        if box:
            key = tuple(box.values())
            if key not in seen:
                seen.add(key)
                box["label"] = ""
                boxes.append(box)
    if evidence_type is None and boxes:
        evidence_type = "focal" if len(boxes) == 1 else "multifocal"
    return evidence_type, boxes[:4]


def explicit_evidence_type(text: str) -> str | None:
    lower = text.lower()
    for pattern in (
        r'evidence_type\s*=\s*["\']?(focal|multifocal|diffuse|not_localizable)',
        r'<evidence_type>\s*(focal|multifocal|diffuse|not_localizable)',
        r'<evidence\s+type\s*=\s*["\']?(focal|multifocal|diffuse|not_localizable)',
        r'visual_evidence\s*:\s*(focal|multifocal|diffuse|not_localizable)',
        r'<ref>\s*(not_localizable|diffuse)\s*</ref>',
    ):
        match = re.search(pattern, lower)
        if match:
            return match.group(1)
    return None


def valid_payload(evidence_type: str | None, boxes: list[dict[str, Any]]) -> bool:
    return (
        evidence_type in {"focal", "multifocal"} and 1 <= len(boxes) <= 4
    ) or (
        evidence_type in {"diffuse", "not_localizable"} and len(boxes) == 0
    )


def summary(rows: list[dict[str, Any]], specs: tuple[FormatSpec, ...]) -> dict[str, Any]:
    output = {}
    for spec in specs:
        subset = [row for row in rows if row["format"] == spec.name]
        valid = [row for row in subset if row["bbox_payload_valid"]]
        comparable = [row for row in valid if row.get("overlap")]
        output[spec.name] = {
            "case_count": len(subset),
            "answer_preservation_rate": float(np.mean([row["answer_preserved"] for row in subset])),
            "answer_accuracy": float(np.mean([row["answer"] == row["target_choice"] for row in subset])),
            "bbox_payload_valid_rate": len(valid) / len(subset),
            "protocol_compliance_rate": float(np.mean([
                row["protocol_compliant"] for row in subset
            ])),
            "spatial_applicability_accuracy": float(np.mean([
                row["bbox_payload_valid"] and row["predicted_spatial"] == row["reference_spatial"]
                for row in subset
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
    parser.add_argument(
        "--format", dest="formats", action="append",
        choices=[spec.name for spec in SPECS],
        help="Run only the selected format; repeat to select multiple formats.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    panel = json.loads(args.panel.read_text())
    references = json.loads(args.references.read_text())
    cases = {row["panel_index"]: row for row in panel["cases"]}
    refs = {row["panel_index"]: row for row in references["results"]}
    baselines = load_baselines(args.baseline_predictions)
    selected = tuple(args.panel_index or (0, 1, 2))
    active_specs = tuple(
        spec for spec in SPECS if not args.formats or spec.name in args.formats
    )
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
    output_path = args.output_dir / "predictions.jsonl"
    for spec in active_specs:
        for index in selected:
            case = cases[index]
            baseline = baselines[index]
            reference = refs[index]["annotation"]
            original_user = {"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": case["problem"]},
            ]}
            if spec.follow_up:
                messages = [
                    original_user,
                    {"role": "assistant", "content": baseline["completion"]},
                    {"role": "user", "content": spec.instruction},
                ]
            else:
                messages = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": case["problem"] + "\n\n" + spec.instruction},
                ]}]
            with Image.open(case["image"]) as source:
                image = source.convert("RGB")
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt += spec.assistant_prefix
            inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
            inputs = {name: value.to("cuda") for name, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True
                )
            completion_ids = generated[:, inputs["input_ids"].shape[1]:]
            completion = spec.assistant_prefix + processor.batch_decode(
                completion_ids, skip_special_tokens=True
            )[0].strip()
            answer = extract_answer(completion, baseline["answer"] if spec.follow_up else None)
            evidence_type, boxes = extract_payload(completion)
            payload_valid = valid_payload(evidence_type, boxes)
            reported_evidence_type = explicit_evidence_type(completion)
            protocol_compliant = payload_valid and reported_evidence_type is not None
            reference_spatial = reference["evidence_type"] in {"focal", "multifocal"}
            predicted_spatial = evidence_type in {"focal", "multifocal"}
            row: dict[str, Any] = {
                "format": spec.name, "follow_up": spec.follow_up, "panel_index": index,
                "image": case["image"], "target_choice": case["target_choice"],
                "baseline_answer": baseline["answer"], "answer": answer,
                "answer_preserved": answer == baseline["answer"],
                "evidence_type": evidence_type, "boxes": boxes,
                "bbox_payload_valid": payload_valid, "reference_spatial": reference_spatial,
                "reported_evidence_type": reported_evidence_type,
                "protocol_compliant": protocol_compliant,
                "predicted_spatial": predicted_spatial, "completion": completion,
            }
            if payload_valid and reference_spatial and predicted_spatial:
                row["overlap"] = overlap_metrics(boxes, reference["boxes"])
            if payload_valid:
                render_overlay(
                    Path(case["image"]), boxes, reference["boxes"],
                    args.output_dir / "overlays" / spec.name / f"case_{index:02d}.png",
                )
            rows.append(row)
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            print(json.dumps({"format": spec.name, "case": index, "answer": answer, "valid": payload_valid}), flush=True)
    metrics = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed_smoke", "formal_result": False,
        "method": "extended_bbox_output_format_probe_v1", "model_label": args.model_label,
        "model_path": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "panel_indices": list(selected), "formats": [spec.name for spec in active_specs],
        "test_accessed": False, "by_format": summary(rows, active_specs),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
