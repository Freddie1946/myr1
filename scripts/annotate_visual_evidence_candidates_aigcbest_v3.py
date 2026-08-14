#!/usr/bin/env python3
"""Exhaustively annotate all visible pathology evidence regions with an external VLM.

The target PathVLM predictions, attention maps, saliency maps, and perturbation
results are deliberately unavailable to the annotator.  These annotations are
model-generated pseudo-reference regions for later blinded expert review, not
pathology ground truth.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_URL = "https://api2.aigcbest.top/v1/chat/completions"
EVIDENCE_TYPES = {"focal", "multifocal", "diffuse", "mixed", "not_localizable"}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_data_url(path: Path, expected_sha256: str) -> str:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"image hash mismatch: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def prompt(case: dict[str, Any]) -> str:
    return f"""You are an independent pathology-image annotator. You are blinded to every
evaluated model prediction, attention map, saliency map, and perturbation result.

Question and choices:
{case['problem']}

Dataset reference answer (for identifying the diagnostic target only):
{case['solution']}

Identify ALL distinct visible image regions that materially support, oppose, or help distinguish
the reference answer from plausible distractors. Do not return only the single strongest region.
Survey the entire image systematically and include every non-duplicate diagnostic focus that could
change the answer or confidence if removed. Use a separate tight box for each spatially distinct
focus, up to 12 boxes. Coordinates are normalized integers [x_min,y_min,x_max,y_max] on a 0-1000
scale, origin top-left. Do not merge distant foci into one large box and do not use the whole image
as a box. For diffuse architecture or background evidence that cannot be faithfully boxed, describe
it in diffuse_evidence instead. Every box endpoint must be a finite numeric value; never put null,
text, ranges, or uncertainty markers inside box. Omit an uncertain box and describe it as diffuse
evidence instead. A case may contain both boxed and diffuse evidence (evidence_type
"mixed"). If the question is not visually answerable or evidence cannot be localized, say so rather
than inventing boxes.

For every box, state the visible feature and whether it supports the reference answer, opposes a
distractor, or provides contextual evidence. Finally perform a second whole-image coverage pass and
set coverage_complete=true only if no additional material region was omitted.
Keep each region feature within 24 words, coverage_note within 40 words, and rationale within 80
words. Completeness refers to spatial evidence coverage, not verbose prose.

Return JSON only:
{{"evidence_type":"focal|multifocal|diffuse|mixed|not_localizable",
  "regions":[{{"box":[0,0,1,1],"feature":"...","role":"supports_reference|opposes_distractor|context","importance":0.0}}],
  "diffuse_evidence":["..."],
  "visual_answerability":"high|medium|low",
  "strongest_distractor":"A/B/C/D or short description",
  "coverage_complete":true,
  "coverage_note":"...",
  "confidence":0.0,
  "rationale":"..."}}
"""


def parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        if start < 0:
            raise
        value, _ = json.JSONDecoder().raw_decode(candidate[start:])
    if not isinstance(value, dict):
        raise ValueError("annotation response is not a JSON object")
    return value


def validate(value: dict[str, Any]) -> dict[str, Any]:
    evidence_type = value.get("evidence_type")
    if evidence_type not in EVIDENCE_TYPES:
        raise ValueError(f"invalid evidence_type: {evidence_type!r}")
    regions = value.get("regions")
    if not isinstance(regions, list) or len(regions) > 12:
        raise ValueError("regions must be a list with at most 12 entries")
    validated_regions = []
    for region in regions:
        if not isinstance(region, dict):
            raise ValueError("each region must be an object")
        box = region.get("box")
        if not isinstance(box, list) or len(box) != 4 or any(
            not isinstance(x, (int, float)) or isinstance(x, bool) or not 0 <= float(x) <= 1000
            for x in box
        ):
            raise ValueError("box must contain four finite normalized numeric coordinates")
        # APIs sometimes serialize an integer-scale endpoint as 123.0.  The raw
        # response remains preserved, while validated geometry is canonicalized.
        box = [int(round(float(x))) for x in box]
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError(f"invalid box endpoints: {box}")
        area_fraction = (box[2] - box[0]) * (box[3] - box[1]) / 1_000_000
        if area_fraction >= 0.80:
            raise ValueError("near-whole-image evidence must be represented as diffuse evidence, not a box")
        feature = region.get("feature")
        role = region.get("role")
        importance = region.get("importance")
        if not isinstance(feature, str) or not feature.strip():
            raise ValueError("each region requires a feature")
        if role not in {"supports_reference", "opposes_distractor", "context"}:
            raise ValueError(f"invalid region role: {role!r}")
        if not isinstance(importance, (int, float)) or not 0 <= float(importance) <= 1:
            raise ValueError("importance must be numeric in [0,1]")
        validated_regions.append(
            {"box": box, "feature": feature.strip(), "role": role, "importance": float(importance)}
        )
    diffuse = value.get("diffuse_evidence")
    if not isinstance(diffuse, list) or any(not isinstance(x, str) or not x.strip() for x in diffuse):
        raise ValueError("diffuse_evidence must be a list of non-empty strings")
    if regions and diffuse:
        derived_evidence_type = "mixed"
    elif len(regions) > 1:
        derived_evidence_type = "multifocal"
    elif len(regions) == 1:
        derived_evidence_type = "focal"
    elif diffuse:
        derived_evidence_type = "diffuse"
    else:
        derived_evidence_type = "not_localizable"
    answerability = value.get("visual_answerability")
    if answerability not in {"high", "medium", "low"}:
        raise ValueError("invalid visual_answerability")
    coverage_complete = value.get("coverage_complete")
    if not isinstance(coverage_complete, bool):
        raise ValueError("coverage_complete must be boolean")
    confidence = value.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        raise ValueError("confidence must be numeric in [0,1]")
    for key in ("strongest_distractor", "coverage_note", "rationale"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    return {
        "evidence_type": derived_evidence_type,
        "reported_evidence_type": evidence_type,
        "evidence_type_normalized": evidence_type != derived_evidence_type,
        "regions": validated_regions,
        "diffuse_evidence": [x.strip() for x in diffuse],
        "visual_answerability": answerability,
        "strongest_distractor": value["strongest_distractor"].strip(),
        "coverage_complete": coverage_complete,
        "coverage_note": value["coverage_note"].strip(),
        "confidence": float(confidence),
        "rationale": value["rationale"].strip(),
    }


def request_annotation(case: dict[str, Any], model: str, key: str, timeout: int, max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt(case)},
            {"type": "image_url", "image_url": {"url": image_data_url(Path(case["image"]), case["image_sha256"])}},
        ]}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        API_URL, data=canonical(payload).encode(), method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = json.loads(response.read().decode())
    choice = body["choices"][0]
    raw = choice["message"]["content"]
    return {
        "status": "validated", "panel_index": case["panel_index"],
        "image_sha256": case["image_sha256"], "requested_model": model,
        "served_model": body.get("model"), "response_id": body.get("id"),
        "finish_reason": choice.get("finish_reason"), "usage": body.get("usage", {}),
        "latency_seconds": time.monotonic() - started, "raw_response": raw,
        "annotation": validate(parse_json_object(raw)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--panel-index", action="append", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite paid output: {args.output}")
    key = os.getenv("AIGCBEST_API_KEY", "")
    if not key:
        raise SystemExit("AIGCBEST_API_KEY is missing")
    panel_raw = args.panel.read_bytes()
    panel = json.loads(panel_raw)
    if args.panel_index:
        wanted = set(args.panel_index)
        cases = [x for x in panel["cases"] if int(x["panel_index"]) in wanted]
        if len(cases) != len(wanted):
            raise ValueError("one or more requested panel indices are missing")
    else:
        cases = panel["cases"][args.start:args.stop]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.output.open("w", encoding="utf-8") as handle:
        for model in args.model:
            for case in cases:
                history = []
                for attempt in range(1, args.max_attempts + 1):
                    try:
                        row = request_annotation(case, model, key, args.timeout, args.max_tokens)
                        row["attempt"] = attempt
                        rows.append(row)
                        handle.write(canonical(row) + "\n"); handle.flush(); os.fsync(handle.fileno())
                        print(canonical({"model": model, "panel_index": case["panel_index"], "status": "validated"}), flush=True)
                        break
                    except Exception as exc:
                        history.append(f"{type(exc).__name__}: {exc}")
                        if attempt == args.max_attempts:
                            row = {"status": "failed", "panel_index": case["panel_index"],
                                   "image_sha256": case["image_sha256"], "requested_model": model,
                                   "attempt_history": history}
                            rows.append(row); handle.write(canonical(row) + "\n"); handle.flush(); os.fsync(handle.fileno())
                            print(canonical({"model": model, "panel_index": case["panel_index"], "status": "failed"}), flush=True)
                        else:
                            time.sleep((2, 5, 12)[min(attempt - 1, 2)])
                time.sleep(max(0, args.sleep))
    summary = {
        "schema_version": 3, "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed", "annotation_kind": "external_model_exhaustive_pseudo_reference_not_expert_ground_truth",
        "blinded_to_target_outputs_and_explanations": True,
        "panel": str(args.panel.resolve()), "panel_sha256": hashlib.sha256(panel_raw).hexdigest(),
        "models": args.model, "case_count_per_model": len(cases), "request_count": len(rows),
        "validated_count": sum(x["status"] == "validated" for x in rows),
        "failed_count": sum(x["status"] != "validated" for x in rows),
        "served_models": sorted({str(x.get("served_model")) for x in rows if x.get("served_model")}),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(canonical(summary))


if __name__ == "__main__":
    main()
