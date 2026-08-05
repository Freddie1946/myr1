#!/usr/bin/env python3
"""Create blinded reference ROIs with an external vision model."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_URL = "https://api2.aigcbest.top"
ALLOWED_EVIDENCE_TYPES = {"focal", "multifocal", "diffuse", "not_localizable"}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def image_data_url(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    if not mime.startswith("image/"):
        raise ValueError(f"unsupported image MIME type: {mime}")
    return (
        f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}",
        hashlib.sha256(raw).hexdigest(),
    )


def annotation_prompt(case: dict[str, Any]) -> str:
    return f"""You are producing an independent reference-region annotation for a pathology image.
You are blinded to every evaluated model, heatmap, and perturbation result.

Question and choices:
{case['problem']}

The dataset's reference answer is {case['target_choice']}.

Identify only image regions that provide direct visual evidence for that reference answer. Coordinates
must be normalized integers from 0 to 1000, with origin at the top-left, x increasing rightward and y
increasing downward. Each box must use [x_min, y_min, x_max, y_max]. Use tight boxes, not the entire
image. For multifocal evidence, return at most six boxes. If the evidence is genuinely diffuse across
the field, use evidence_type "diffuse" and no boxes. If the question concerns absence or cannot be
localized from the image, use "not_localizable" and no boxes. Do not force a box.

Return JSON only with this exact structure:
{{"evidence_type":"focal|multifocal|diffuse|not_localizable","boxes":[{{"x_min":0,"y_min":0,"x_max":1,"y_max":1,"label":"short evidence label"}}],"confidence":0.0,"reason":"one short sentence"}}
"""


def parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        object_start = candidate.find("{")
        if object_start < 0:
            raise
        value, _ = json.JSONDecoder().raw_decode(candidate[object_start:])
    if not isinstance(value, dict):
        raise ValueError("annotation response is not a JSON object")
    return value


def validate_annotation(value: dict[str, Any]) -> dict[str, Any]:
    evidence_type = value.get("evidence_type")
    if evidence_type not in ALLOWED_EVIDENCE_TYPES:
        raise ValueError(f"invalid evidence_type: {evidence_type!r}")
    confidence = value.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        raise ValueError("confidence must be numeric in [0, 1]")
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    boxes = value.get("boxes")
    if not isinstance(boxes, list) or len(boxes) > 6:
        raise ValueError("boxes must be a list containing at most six entries")
    if evidence_type in {"diffuse", "not_localizable"} and boxes:
        raise ValueError(f"{evidence_type} annotations must not contain boxes")
    if evidence_type in {"focal", "multifocal"} and not boxes:
        raise ValueError(f"{evidence_type} annotations must contain at least one box")
    validated_boxes = []
    for box in boxes:
        if not isinstance(box, dict):
            raise ValueError("each box must be an object")
        coordinates = []
        for name in ("x_min", "y_min", "x_max", "y_max"):
            coordinate = box.get(name)
            if not isinstance(coordinate, int) or not 0 <= coordinate <= 1000:
                raise ValueError(f"{name} must be an integer in [0, 1000]")
            coordinates.append(coordinate)
        if coordinates[0] >= coordinates[2] or coordinates[1] >= coordinates[3]:
            raise ValueError(f"invalid box endpoints: {coordinates}")
        label = box.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("each box requires a non-empty label")
        validated_boxes.append(
            dict(zip(("x_min", "y_min", "x_max", "y_max"), coordinates))
            | {"label": label.strip()}
        )
    return {
        "evidence_type": evidence_type,
        "boxes": validated_boxes,
        "confidence": float(confidence),
        "reason": reason.strip(),
    }


def annotate(
    *, key: str, model: str, case: dict[str, Any], timeout: int, max_tokens: int
) -> dict[str, Any]:
    image = Path(case["image"]).resolve()
    data_url, image_sha256 = image_data_url(image)
    if image_sha256 != case["image_sha256"]:
        raise ValueError(f"image hash mismatch for panel case {case['panel_index']}")
    prompt = annotation_prompt(case)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = urllib.request.Request(
        BASE_URL + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(),
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
        status = int(response.status)
    choices = body.get("choices")
    if status != 200 or not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError(f"unexpected annotation response for case {case['panel_index']}")
    message = choices[0].get("message", {})
    raw_response = message.get("content")
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise RuntimeError(f"empty annotation response for case {case['panel_index']}")
    result = {
        "status": "response_received",
        "panel_index": case["panel_index"],
        "source_index": case["index"],
        "source_record_sha256": case["source_record_sha256"],
        "image": str(image),
        "image_sha256": image_sha256,
        "target_choice": case["target_choice"],
        "max_tokens": max_tokens,
        "requested_model": model,
        "served_model": body.get("model"),
        "response_id": body.get("id"),
        "latency_seconds": time.monotonic() - started,
        "finish_reason": choices[0].get("finish_reason"),
        "usage": body.get("usage") if isinstance(body.get("usage"), dict) else {},
        "raw_response": raw_response,
    }
    try:
        result["annotation"] = validate_annotation(parse_json_object(raw_response))
        result["status"] = "validated"
    except (json.JSONDecodeError, ValueError) as error:
        result["annotation"] = None
        result["status"] = "invalid_annotation"
        result["validation_error"] = f"{type(error).__name__}: {error}"
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--panel-index", action="append", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite paid annotation output: {args.output}")
    key = os.getenv("AIGCBEST_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("AIGCBEST_API_KEY is not set")
    panel_raw = args.panel.read_bytes()
    panel = json.loads(panel_raw)
    cases = {case["panel_index"]: case for case in panel["cases"]}
    requested_indices = list(dict.fromkeys(args.panel_index))
    missing = sorted(set(requested_indices) - set(cases))
    if missing:
        raise ValueError(f"panel indices are absent: {missing}")

    results = []
    record = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running_smoke",
        "formal_result": False,
        "annotation_kind": "external_model_pseudo_reference_roi_not_expert_ground_truth",
        "provider_gateway": "aigcbest",
        "panel_path": str(args.panel.resolve()),
        "panel_sha256": hashlib.sha256(panel_raw).hexdigest(),
        "models": args.model,
        "panel_indices": requested_indices,
        "blinded_to_heatmaps": True,
        "results": results,
    }
    for model in args.model:
        for panel_index in requested_indices:
            results.append(
                annotate(
                    key=key,
                    model=model,
                    case=cases[panel_index],
                    timeout=args.timeout,
                    max_tokens=args.max_tokens,
                )
            )
            atomic_json(args.output, record)
    record["status"] = "completed_smoke"
    atomic_json(args.output, record)
    print(
        json.dumps(
            {
                "status": record["status"],
                "output": str(args.output.resolve()),
                "request_count": len(results),
                "served_models": sorted({row["served_model"] for row in results}),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
