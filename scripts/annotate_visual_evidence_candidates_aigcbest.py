#!/usr/bin/env python3
"""Annotate frozen pathology evidence candidates with an external VLM.

The output is reference annotation metadata only.  It is never treated as
pathology ground truth without later expert review.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_URL = "https://api2.aigcbest.top/v1/chat/completions"
MODEL = "gpt-4o-2024-08-06"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def image_url(path: Path, expected_sha: str) -> str:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha:
        raise ValueError(f"image hash mismatch: {path}")
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "evidence_type": {"type": "string", "enum": ["focal", "multifocal", "diffuse", "not_localizable"]},
            "boxes": {"type": "array", "maxItems": 4, "items": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number", "minimum": 0, "maximum": 1000}}},
            "diagnostically_relevant": {"type": "boolean"},
            "discriminating_feature": {"type": "string"},
            "rationale": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "redundancy": {"type": "number", "minimum": 0, "maximum": 1},
            "deletion_prediction": {"type": "string", "enum": ["confidence_drop", "answer_flip_likely", "little_change_expected", "uncertain"]},
        },
        "required": ["evidence_type", "boxes", "diagnostically_relevant", "discriminating_feature", "rationale", "confidence", "redundancy", "deletion_prediction"],
        "additionalProperties": False,
    }


def annotate(case: dict[str, Any], api_key: str, *, timeout: float, retries: int) -> dict[str, Any]:
    path = Path(case["image"])
    prompt = (
        "You are an independent pathology image annotator. Identify a pre-existing "
        "diagnostically relevant visual evidence region for the given multiple-choice "
        "question. Do not infer from any model prediction; use only the image, question, "
        "options, and reference answer. Return up to four normalized boxes [x0,y0,x1,y1] "
        "on a 0-1000 coordinate scale. If evidence is diffuse or not localizable, return "
        "an empty list. Explain which feature distinguishes the gold answer from the most "
        "plausible distractor. Estimate redundancy: 0 means concentrated and non-redundant, "
        "1 means many equivalent regions."
        f"\nQUESTION:\n{case['problem']}\nREFERENCE ANSWER (for target identification only):\n{case['solution']}"
    )
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_url(path, case["image_sha256"])}},
        ]}],
        "temperature": 0,
        "max_tokens": 700,
        "response_format": {"type": "json_schema", "json_schema": {"name": "pathology_evidence_annotation", "strict": True, "schema": schema()}},
    }
    request = urllib.request.Request(API_URL, data=canonical(payload).encode(), method="POST", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    last = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode())
            content = body["choices"][0]["message"]["content"]
            value = json.loads(content)
            return {"panel_index": case["panel_index"], "image_sha256": case["image_sha256"], "model": MODEL, "annotation": value, "attempt": attempt + 1}
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
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
    parser.add_argument("--sleep", type=float, default=1.0)
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
                row = json.loads(line); done[int(row["panel_index"])] = row
    with args.output.open("a", encoding="utf-8") as handle:
        for case in cases:
            if int(case["panel_index"]) in done:
                continue
            row = annotate(case, key, timeout=180, retries=3)
            handle.write(canonical(row) + "\n"); handle.flush(); os.fsync(handle.fileno())
            print(canonical({"completed": len(done) + 1, "requested": len(cases), "panel_index": case["panel_index"]}), flush=True)
            done[int(case["panel_index"])] = row
            time.sleep(max(0.0, args.sleep))
    summary = {"status": "completed", "panel": str(args.panel.resolve()), "panel_sha256": sha256_file(args.panel), "annotation_count": len(done), "model": MODEL}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
