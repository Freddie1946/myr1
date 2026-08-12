#!/usr/bin/env python3
"""Freeze a deterministic PathVQA cross-format semantic probe panel.

The panel is selected only from already-scored forced-binary outputs, before any
cross-format generations or judge labels exist.  It deliberately uses target-Yes
cases so the positive statement can be converted into a diagnosis/finding MCQ
without inventing the true diagnosis for a target-No record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


QUOTAS = {
    "regression": {
        "stain_marker": 5,
        "morphology": 8,
        "image_diagnosis": 7,
        "short_presence": 7,
        "other": 5,
    },
    "improvement": {
        "morphology": 8,
        "image_diagnosis": 6,
        "short_presence": 2,
    },
    "stable_correct": {
        "short_presence": 4,
        "image_diagnosis": 4,
        "morphology": 3,
        "stain_marker": 2,
        "other": 3,
    },
}


def read_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row["source_record_sha256"]
        if key in rows:
            raise ValueError(f"duplicate source_record_sha256 in {path}: {key}")
        rows[key] = row
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def category(question: str) -> str:
    q = question.lower().strip()
    if re.fullmatch(r"is [a-z -]+ present\?", q):
        return "short_presence"
    if q.startswith("does this image show"):
        return "image_diagnosis"
    if any(
        token in q
        for token in (
            "stain",
            "positive",
            "negative",
            "expression",
            "marker",
            "cytokeratin",
            " cd",
        )
    ):
        return "stain_marker"
    if any(
        token in q
        for token in (
            "nuclei",
            "nuclear",
            "cytoplas",
            "cell",
            "gland",
            "papill",
            "morpholog",
            "spindle",
            "pleomorph",
            "necrosis",
            "granul",
            "infiltrat",
            "fibro",
            "stroma",
            "mito",
        )
    ):
        return "morphology"
    return "other"


def outcome_group(base: dict, sft: dict) -> str | None:
    base_ok = bool(base["forced_binary_correct"])
    sft_ok = bool(sft["forced_binary_correct"])
    if base_ok and not sft_ok:
        return "regression"
    if not base_ok and sft_ok:
        return "improvement"
    if base_ok and sft_ok:
        return "stable_correct"
    return None


def stable_key(salt: str, group: str, cat: str, row: dict) -> str:
    payload = "\n".join(
        [salt, group, cat, row["source_record_sha256"], row["image"]]
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_panel(base_rows: dict[str, dict], sft_rows: dict[str, dict], salt: str) -> list[dict]:
    if set(base_rows) != set(sft_rows):
        raise ValueError("Base and SFT prediction keys differ")

    candidates: dict[tuple[str, str], list[tuple[str, dict, dict]]] = defaultdict(list)
    for key in sorted(base_rows):
        base = base_rows[key]
        sft = sft_rows[key]
        if base["target"].lower() != "yes":
            continue
        group = outcome_group(base, sft)
        if group not in QUOTAS:
            continue
        cat = category(base["question"])
        if cat not in QUOTAS[group]:
            continue
        candidates[(group, cat)].append((stable_key(salt, group, cat, base), base, sft))

    selected = []
    used_images: set[str] = set()
    for group, category_quotas in QUOTAS.items():
        for cat, quota in category_quotas.items():
            accepted = []
            for _, base, sft in sorted(candidates[(group, cat)], key=lambda item: item[0]):
                if base["image"] in used_images:
                    continue
                accepted.append((base, sft))
                used_images.add(base["image"])
                if len(accepted) == quota:
                    break
            if len(accepted) != quota:
                raise ValueError(
                    f"insufficient unique-image candidates for {group}/{cat}: "
                    f"wanted {quota}, found {len(accepted)}"
                )
            for base, sft in accepted:
                selected.append(
                    {
                        "panel_index": len(selected),
                        "group": group,
                        "category": cat,
                        "source_index": base["source_index"],
                        "source_record_sha256": base["source_record_sha256"],
                        "image": base["image"],
                        "question": base["question"],
                        "target": base["target"],
                        "base_forced_answer": base["forced_binary_answer"],
                        "base_target_margin": base["target_margin"],
                        "sft_forced_answer": sft["forced_binary_answer"],
                        "sft_target_margin": sft["target_margin"],
                    }
                )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--sft-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--salt", default="pathvqa-cross-format-semantic-v1-20260811")
    args = parser.parse_args()

    base_rows = read_jsonl(args.base_predictions)
    sft_rows = read_jsonl(args.sft_predictions)
    panel = build_panel(base_rows, sft_rows, args.salt)
    expected = sum(sum(values.values()) for values in QUOTAS.values())
    if len(panel) != expected or len({row["image"] for row in panel}) != expected:
        raise RuntimeError("panel size or unique-image gate failed")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    panel_path = args.output_dir / "panel.json"
    panel_path.write_text(json.dumps(panel, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "frozen_before_cross_format_generation",
        "formal_result": False,
        "selection_or_tuning_use_forbidden": True,
        "target_filter": "yes_only",
        "salt": args.salt,
        "count": len(panel),
        "unique_images": len({row["image"] for row in panel}),
        "quotas": QUOTAS,
        "base_predictions": str(args.base_predictions.resolve()),
        "base_predictions_sha256": sha256_file(args.base_predictions),
        "sft_predictions": str(args.sft_predictions.resolve()),
        "sft_predictions_sha256": sha256_file(args.sft_predictions),
        "panel": str(panel_path.resolve()),
        "panel_sha256": sha256_file(panel_path),
        "planned_probes": [
            "free_generated_binary_think_answer",
            "open_ended_visible_features",
            "four_choice_diagnosis_or_finding_mcq",
        ],
        "max_new_tokens": 2048,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
