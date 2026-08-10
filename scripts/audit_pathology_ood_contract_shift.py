#!/usr/bin/env python3
"""Reproducible dataset/contract audit for PathMMU training versus PathVQA test."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from PIL import Image

from external_vqa_contract import sha256_file


def image_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def summarize(name: str, rows: list[dict], question_key: str) -> dict:
    image_paths = sorted({row["image"] for row in rows})
    words = [len(str(row[question_key]).split()) for row in rows]
    dimensions, modes, formats = [], Counter(), Counter()
    hashes = set()
    for path in image_paths:
        hashes.add(image_sha256(path))
        with Image.open(path) as image:
            dimensions.append(image.size)
            modes[image.mode] += 1
            formats[str(image.format)] += 1
    widths = [width for width, _ in dimensions]
    heights = [height for _, height in dimensions]
    aspects = [max(width / height, height / width) for width, height in dimensions]
    return {
        "name": name,
        "row_count": len(rows),
        "unique_image_path_count": len(image_paths),
        "unique_image_content_sha256_count": len(hashes),
        "mean_questions_per_unique_image": len(rows) / len(image_paths),
        "question_words": {
            "mean": statistics.fmean(words),
            "median": statistics.median(words),
            "p95": percentile(words, 0.95),
        },
        "images": {
            "median_width": statistics.median(widths),
            "median_height": statistics.median(heights),
            "median_area": statistics.median([w * h for w, h in dimensions]),
            "aspect_ratio_over_2_rate": sum(value > 2 for value in aspects) / len(aspects),
            "modes": dict(sorted(modes.items())),
            "formats": dict(sorted(formats.items())),
        },
        "image_content_hashes": hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathmmu-sft", required=True, type=Path)
    parser.add_argument("--pathmmu-rl", required=True, type=Path)
    parser.add_argument("--pathvqa", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pathmmu = json.loads(args.pathmmu_sft.read_text(encoding="utf-8")) + json.loads(args.pathmmu_rl.read_text(encoding="utf-8"))
    pathvqa_all = json.loads(args.pathvqa.read_text(encoding="utf-8"))
    pathvqa = [row for row in pathvqa_all if row.get("answer_type") == "yes_no"]
    if len(pathmmu) != 4000 or len(pathvqa) != 3362:
        raise ValueError("unexpected frozen dataset sizes")
    mm = summarize("PathMMU SFT3000 + RL1000", pathmmu, "problem")
    pv = summarize("PathVQA test yes/no", pathvqa, "question")
    overlap = mm.pop("image_content_hashes") & pv.pop("image_content_hashes")
    result = {
        "schema_version": 1,
        "status": "completed",
        "purpose": "descriptive OOD contract audit; not a causal attribution by itself",
        "inputs": {
            "pathmmu_sft": str(args.pathmmu_sft.resolve()),
            "pathmmu_sft_sha256": sha256_file(args.pathmmu_sft),
            "pathmmu_rl": str(args.pathmmu_rl.resolve()),
            "pathmmu_rl_sha256": sha256_file(args.pathmmu_rl),
            "pathvqa": str(args.pathvqa.resolve()),
            "pathvqa_sha256": sha256_file(args.pathvqa),
        },
        "pathmmu": mm,
        "pathvqa": {
            **pv,
            "answer_label_counts": dict(sorted(Counter(str(row["answer"]).lower() for row in pathvqa).items())),
        },
        "exact_image_content_overlap_count": len(overlap),
        "contract_contrast": {
            "pathmmu": "four-option MCQ with long question/options and trained <think>...<answer>letter + option text</answer>",
            "pathvqa": "short binary yes/no question with no candidate options; repeated questions per image",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(args.output.resolve()), "overlap": len(overlap)}))


if __name__ == "__main__":
    main()
