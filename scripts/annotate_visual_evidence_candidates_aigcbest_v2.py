#!/usr/bin/env python3
"""Annotate a frozen pathology panel for causal-deletion suitability.

The annotation is model-blind with respect to the target PathVLM checkpoint.
It is a reference annotation for later pathology-expert review, not ground
truth.  Compared with v1, this contract explicitly separates mere visual
relevance from suitability for a controlled evidence-deletion experiment.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from annotate_visual_evidence_candidates_aigcbest import (
    API_URL,
    MODEL,
    canonical,
    image_url,
    sha256_file,
)


def schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "evidence_type": {
                "type": "string",
                "enum": ["focal", "multifocal", "diffuse", "not_localizable"],
            },
            "boxes": {
                "type": "array",
                "maxItems": 4,
                "items": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "number", "minimum": 0, "maximum": 1000},
                },
            },
            "diagnostically_relevant": {"type": "boolean"},
            "visual_answerability": {
                "type": "string", "enum": ["high", "medium", "low"]
            },
            "causal_panel_suitability": {
                "type": "string", "enum": ["strong", "moderate", "weak", "exclude"]
            },
            "requires_global_context": {"type": "boolean"},
            "discriminating_feature": {"type": "string"},
            "strongest_distractor": {"type": "string"},
            "rationale": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence_concentration": {"type": "number", "minimum": 0, "maximum": 1},
            "redundancy": {"type": "number", "minimum": 0, "maximum": 1},
            "deletion_prediction": {
                "type": "string",
                "enum": [
                    "answer_flip_likely", "confidence_drop_likely",
                    "little_change_expected", "uncertain",
                ],
            },
        },
        "required": [
            "evidence_type", "boxes", "diagnostically_relevant",
            "visual_answerability", "causal_panel_suitability",
            "requires_global_context", "discriminating_feature",
            "strongest_distractor", "rationale", "confidence",
            "evidence_concentration", "redundancy", "deletion_prediction",
        ],
        "additionalProperties": False,
    }


def annotate(case: dict[str, Any], api_key: str, timeout: float, retries: int) -> dict[str, Any]:
    prompt = (
        "You are an independent pathology-image annotator constructing a controlled "
        "causal evidence-deletion panel. You have not seen and must not speculate about "
        "the target model's outputs, attention, or saliency. Use only the image, question, "
        "options, and reference answer. First decide whether the image itself contains "
        "enough visible information to distinguish the reference answer from the strongest "
        "distractor. Mere relevance is insufficient: a strong case has concentrated, "
        "low-redundancy evidence whose local blur/replacement should predictably reduce "
        "confidence or flip the answer while leaving most of the image intact. Return up "
        "to four tight normalized boxes [x0,y0,x1,y1] on a 0-1000 scale. If evidence is "
        "diffuse, redundant, ambiguous, too small to localize reliably, or requires global "
        "context, return empty boxes or mark the case weak/exclude. Do not assume deletion "
        "must matter; choose little_change_expected or uncertain when appropriate."
        f"\nQUESTION:\n{case['problem']}"
        f"\nREFERENCE ANSWER (target identification only):\n{case['solution']}"
    )
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {
                "url": image_url(Path(case["image"]), case["image_sha256"])
            }},
        ]}],
        "temperature": 0,
        "max_tokens": 900,
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "pathology_causal_evidence_annotation_v2",
            "strict": True,
            "schema": schema(),
        }},
    }
    request = urllib.request.Request(
        API_URL, data=canonical(payload).encode(), method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    last: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode())
            annotation = json.loads(body["choices"][0]["message"]["content"])
            return {
                "panel_index": case["panel_index"],
                "image_sha256": case["image_sha256"],
                "model": MODEL,
                "annotation_contract": "causal_deletion_suitability_v2",
                "annotation": annotation,
                "attempt": attempt + 1,
            }
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError,
                TypeError, json.JSONDecodeError) as exc:
            last = exc
            if attempt < retries:
                time.sleep((2, 5, 12)[min(attempt, 2)])
    raise RuntimeError(f"annotation failed for panel {case['panel_index']}: {last}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()
    key = os.environ.get("AIGCBEST_API_KEY", "")
    if not key:
        raise SystemExit("AIGCBEST_API_KEY is missing")
    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    cases = panel["cases"][args.start:args.stop]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done: dict[int, dict[str, Any]] = {}
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[int(row["panel_index"])] = row
    with args.output.open("a", encoding="utf-8") as handle:
        for case in cases:
            index = int(case["panel_index"])
            if index in done:
                continue
            row = annotate(case, key, timeout=180, retries=3)
            handle.write(canonical(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            done[index] = row
            print(canonical({"completed_total": len(done), "panel_index": index}), flush=True)
            time.sleep(max(0.0, args.sleep))
    summary = {
        "status": "completed", "schema_version": 2,
        "annotation_contract": "causal_deletion_suitability_v2",
        "panel": str(args.panel.resolve()), "panel_sha256": sha256_file(args.panel),
        "annotation_count": len(done), "model": MODEL,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
