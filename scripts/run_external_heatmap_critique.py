#!/usr/bin/env python3
"""Ask an external vision model to critique frozen occlusion heatmaps."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_external_roi_annotations import atomic_json, parse_json_object


BASE_URL = "https://api2.aigcbest.top"
ASSESSMENTS = {"reasonable", "partially_reasonable", "unreasonable"}


def data_url(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}", hashlib.sha256(raw).hexdigest()


def validate(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("assessment") not in ASSESSMENTS:
        raise ValueError("invalid assessment")
    rating = value.get("alignment_rating")
    if not isinstance(rating, int) or not 0 <= rating <= 4:
        raise ValueError("alignment_rating must be an integer in [0, 4]")
    for name in ("direct_visual_evidence", "heatmap_alignment", "patch_resolution_comment", "artifact_risk", "decision_fidelity_caveat"):
        if not isinstance(value.get(name), str) or not value[name].strip():
            raise ValueError(f"missing string field: {name}")
    return value


def prompt(case: dict[str, Any], result: dict[str, Any]) -> str:
    return f"""Act as an independent visual-methodology reviewer. The attached figure has three panels:
left = original pathology image; middle = a 6x6 patch-occlusion importance map; right = overlay.
Red means that masking the patch reduced the normalized A/B/C/D probability of the dataset reference
option; blue means masking increased it. The smooth appearance is only interpolation of 36 values.

Question and choices:
{case['problem']}

Dataset reference option: {case['target_choice']}
Model prediction on the unmasked image: {result['baseline_predicted_choice']}
Prediction correct: {str(result['baseline_correct']).lower()}
Deletion advantage: {result['deletion_auc_advantage']:.6f}
Insertion advantage: {result['insertion_auc_advantage']:.6f}
Top-25% comprehensiveness: {result['top_25pct_comprehensiveness']:.6f}

Judge whether the highlighted regions visually coincide with the image evidence relevant to the
reference answer. Explicitly consider whether the evidence is focal, multifocal, diffuse, or not
spatially localizable; whether a 6x6 grid is adequate; whether mean-color masking can create artifacts;
and, when prediction is wrong, that this target-option map does not explain the actual predicted choice.
Do not assume the heatmap is correct merely because metrics are positive.

Return JSON only:
{{"assessment":"reasonable|partially_reasonable|unreasonable","alignment_rating":0,"direct_visual_evidence":"...","heatmap_alignment":"...","patch_resolution_comment":"...","artifact_risk":"...","decision_fidelity_caveat":"..."}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--case-results", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--panel-index", action="append", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite paid critique output: {args.output}")
    key = os.getenv("AIGCBEST_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("AIGCBEST_API_KEY is not set")
    panel = json.loads(args.panel.read_text())
    cases = {case["panel_index"]: case for case in panel["cases"]}
    results = {row["panel_index"]: row for row in (json.loads(line) for line in args.case_results.open())}
    record = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running_smoke",
        "formal_result": False,
        "provider_gateway": "aigcbest",
        "requested_model": args.model,
        "panel_indices": args.panel_index,
        "results": [],
    }
    for panel_index in args.panel_index:
        figure = args.figures / f"case_{panel_index:02d}_heatmap.png"
        image_url, figure_sha256 = data_url(figure)
        request_payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt(cases[panel_index], results[panel_index])},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]}],
            "temperature": 0,
            "max_tokens": 1200,
            "stream": False,
        }
        request = urllib.request.Request(
            BASE_URL + "/v1/chat/completions",
            data=json.dumps(request_payload, ensure_ascii=False, separators=(",", ":")).encode(),
            method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        choice = body["choices"][0]
        raw = choice.get("message", {}).get("content", "")
        row = {
            "panel_index": panel_index,
            "figure": str(figure.resolve()),
            "figure_sha256": figure_sha256,
            "requested_model": args.model,
            "served_model": body.get("model"),
            "response_id": body.get("id"),
            "latency_seconds": time.monotonic() - started,
            "finish_reason": choice.get("finish_reason"),
            "usage": body.get("usage", {}),
            "raw_response": raw,
        }
        try:
            row["critique"] = validate(parse_json_object(raw))
            row["status"] = "validated"
        except (json.JSONDecodeError, ValueError) as error:
            row["critique"] = None
            row["status"] = "invalid_critique"
            row["validation_error"] = f"{type(error).__name__}: {error}"
        record["results"].append(row)
        atomic_json(args.output, record)
    record["status"] = "completed_smoke"
    atomic_json(args.output, record)
    print(json.dumps({"status": record["status"], "results": len(record["results"]), "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
