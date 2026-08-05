#!/usr/bin/env python3
"""Blindly compare occlusion and attention maps with an external vision model."""

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
BASE_METHODS = ("occlusion_6x6", "raw_attention", "gradient_attention", "relevance_rollout")
BLOG_METHOD = "blog_text_attention"


def data_url(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}", hashlib.sha256(raw).hexdigest()


def validate(
    value: dict[str, Any], expected_methods: tuple[str, ...] = BASE_METHODS
) -> dict[str, Any]:
    methods = value.get("methods")
    if not isinstance(methods, dict) or set(methods) != set(expected_methods):
        raise ValueError(f"methods must contain exactly {expected_methods}")
    for method in expected_methods:
        row = methods[method]
        if not isinstance(row, dict):
            raise ValueError(f"invalid method assessment: {method}")
        rating = row.get("alignment_rating")
        if not isinstance(rating, int) or not 0 <= rating <= 4:
            raise ValueError(f"invalid alignment_rating for {method}")
        if not isinstance(row.get("assessment"), str) or not row["assessment"].strip():
            raise ValueError(f"missing assessment for {method}")
    for name in ("direct_visual_evidence", "best_method", "overall_verdict", "limitations"):
        if not isinstance(value.get(name), str) or not value[name].strip():
            raise ValueError(f"missing field: {name}")
    if value["best_method"] not in {*expected_methods, "none"}:
        raise ValueError("best_method is invalid")
    return value


def prompt(
    case: dict[str, Any], record: dict[str, Any], *, include_blog_attention: bool
) -> str:
    if include_blog_attention:
        explanation_count = "five"
        image_order = """1. High-resolution original image.
2. occlusion_6x6: mean-color single-patch target-option probability-drop map.
3. raw_attention: last query, last four language layers.
4. blog_text_attention: all post-image text queries, averaged across all language layers.
5. gradient_attention: positive attention multiplied by the gradient of the reference-answer option margin.
6. relevance_rollout: gradient-weighted attention propagated through all language layers."""
        methods_schema = (
            '"occlusion_6x6":{"alignment_rating":0,"assessment":"..."},'
            '"raw_attention":{"alignment_rating":0,"assessment":"..."},'
            '"blog_text_attention":{"alignment_rating":0,"assessment":"..."},'
            '"gradient_attention":{"alignment_rating":0,"assessment":"..."},'
            '"relevance_rollout":{"alignment_rating":0,"assessment":"..."}'
        )
        best_choices = (
            "occlusion_6x6|raw_attention|blog_text_attention|gradient_attention|"
            "relevance_rollout|none"
        )
    else:
        explanation_count = "four"
        image_order = """1. High-resolution original image.
2. occlusion_6x6: mean-color single-patch target-option probability-drop map.
3. raw_attention: last query, last four language layers.
4. gradient_attention: positive attention multiplied by the gradient of the reference-answer option margin.
5. relevance_rollout: gradient-weighted attention propagated through all language layers."""
        methods_schema = (
            '"occlusion_6x6":{"alignment_rating":0,"assessment":"..."},'
            '"raw_attention":{"alignment_rating":0,"assessment":"..."},'
            '"gradient_attention":{"alignment_rating":0,"assessment":"..."},'
            '"relevance_rollout":{"alignment_rating":0,"assessment":"..."}'
        )
        best_choices = "occlusion_6x6|raw_attention|gradient_attention|relevance_rollout|none"
    return f"""You are an independent visual-methodology reviewer. Judge spatial alignment from the
attached pathology image and {explanation_count} explanation figures. Do not assume any method is correct, and do not
infer quality from visual smoothness. You are not given any computed overlap metric.

Question and choices:
{case['problem']}

Dataset reference answer: {case['target_choice']}
Model prediction: {record['predicted_choice']}
Prediction correct: {str(record['correct']).lower()}

The images following this prompt are ordered as:
{image_order}

For each method, inspect whether its strongest highlighted regions coincide with the direct visual
evidence needed for the reference answer. Penalize background, blank-space, border, and unrelated-tissue
hotspots. If the evidence is diffuse or non-localizable, say so instead of demanding a compact hotspot.
When the prediction is wrong, distinguish reference-answer evidence from an explanation of the actual
wrong decision.

Return JSON only:
{{"direct_visual_evidence":"brief description and location","methods":{{{methods_schema}}},"best_method":"{best_choices}","overall_verdict":"...","limitations":"..."}}

Use this rating scale: 0=no meaningful overlap, 1=mostly wrong, 2=partial/mixed, 3=mostly aligned,
4=strongly aligned with the relevant visual evidence.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--attention-run", type=Path, action="append", required=True)
    parser.add_argument("--occlusion-figures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--panel-index", type=int, action="append", required=True)
    parser.add_argument("--max-tokens", type=int, default=2200)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--include-blog-attention", action="store_true")
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
    attention_records: dict[int, dict[str, Any]] = {}
    attention_roots: dict[int, Path] = {}
    for run in args.attention_run:
        payload = json.loads((run / "attention_results.json").read_text())
        for row in payload["records"]:
            attention_records[row["panel_index"]] = row
            attention_roots[row["panel_index"]] = run
    record = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running_smoke",
        "formal_result": False,
        "provider_gateway": "aigcbest",
        "requested_model": args.model,
        "blinded_to_computed_overlap_metrics": True,
        "include_blog_attention": args.include_blog_attention,
        "results": [],
    }
    expected_methods = (
        ("occlusion_6x6", "raw_attention", BLOG_METHOD, "gradient_attention", "relevance_rollout")
        if args.include_blog_attention
        else BASE_METHODS
    )
    for panel_index in args.panel_index:
        row = attention_records[panel_index]
        root = attention_roots[panel_index]
        paths = [
            Path(row["image"]),
            args.occlusion_figures / f"case_{panel_index:02d}_heatmap.png",
            root / f"case_{panel_index:02d}_last_query_last4.png",
        ]
        if args.include_blog_attention:
            paths.append(root / f"case_{panel_index:02d}_blog_text_all_layers.png")
        paths.extend(
            [
                root / f"case_{panel_index:02d}_target_grad_attention_last4.png",
                root / f"case_{panel_index:02d}_target_relevance_rollout.png",
            ]
        )
        content: list[dict[str, Any]] = [{
            "type": "text",
            "text": prompt(
                cases[panel_index], row, include_blog_attention=args.include_blog_attention
            ),
        }]
        image_records = []
        for path in paths:
            url, digest = data_url(path)
            content.append({"type": "image_url", "image_url": {"url": url}})
            image_records.append({"path": str(path.resolve()), "sha256": digest})
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "stream": False,
        }
        request = urllib.request.Request(
            BASE_URL + "/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(),
            method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        choice = body["choices"][0]
        raw = choice.get("message", {}).get("content", "")
        result = {
            "panel_index": panel_index,
            "requested_model": args.model,
            "served_model": body.get("model"),
            "response_id": body.get("id"),
            "finish_reason": choice.get("finish_reason"),
            "latency_seconds": time.monotonic() - started,
            "usage": body.get("usage", {}),
            "images": image_records,
            "raw_response": raw,
        }
        try:
            result["critique"] = validate(parse_json_object(raw), expected_methods)
            result["status"] = "validated"
        except (json.JSONDecodeError, ValueError) as error:
            result["critique"] = None
            result["status"] = "invalid_critique"
            result["validation_error"] = f"{type(error).__name__}: {error}"
        record["results"].append(result)
        atomic_json(args.output, record)
    record["status"] = "completed_smoke"
    atomic_json(args.output, record)
    print(json.dumps({"status": record["status"], "results": len(record["results"]), "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
