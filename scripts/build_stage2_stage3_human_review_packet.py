#!/usr/bin/env python3
"""Build auditable Stage2-versus-Stage3 human-review packets.

The script creates two deliberately separate products:

1. An output-independent, blinded A/B packet for estimating human preference.
2. Descriptive wrong-to-right case packs for inspecting concrete improvements.

The latter must never be used to estimate an overall preference rate because it
is selected using model correctness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import shutil
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            index = int(row["index"])
            if index in rows:
                raise ValueError(f"duplicate index {index}: {path}")
            rows[index] = row
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def correct(row: dict[str, Any]) -> bool:
    return float(row.get("accuracy_reward", 0.0)) > 0.5


def verify_pair(a: dict[str, Any], b: dict[str, Any], index: int) -> None:
    for key in ("source_record_sha256", "problem", "target_choice"):
        if str(a.get(key)) != str(b.get(key)):
            raise ValueError(f"pair mismatch at index {index}: {key}")
    if Path(str(a["image"])).resolve() != Path(str(b["image"])).resolve():
        raise ValueError(f"image mismatch at index {index}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_image(row: dict[str, Any], directory: Path) -> str:
    source = Path(str(row["image"])).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower() or ".png"
    target = directory / f"image{suffix}"
    shutil.copy2(source, target)
    return target.name


def case_page(
    *, image_name: str, problem: str, reference: str, response_a: str, response_b: str
) -> str:
    esc = lambda value: html.escape(str(value)).replace("\n", "<br>\n")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Blind pairwise review</title>
<style>body{{font-family:sans-serif;max-width:1200px;margin:24px auto;line-height:1.55}}
img{{max-width:100%;max-height:620px}} .pair{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.box{{border:1px solid #bbb;padding:14px;border-radius:8px}} code{{white-space:pre-wrap}}</style></head>
<body><h1>盲化回答比较</h1><img src="{html.escape(image_name)}">
<h2>问题</h2><div class="box">{esc(problem)}</div>
<h2>参考答案（仅供核对医学正确性）</h2><div class="box">{esc(reference)}</div>
<div class="pair"><section><h2>回答 A</h2><div class="box">{esc(response_a)}</div></section>
<section><h2>回答 B</h2><div class="box">{esc(response_b)}</div></section></div>
<p>请在总评分表中填写 A / B / Tie，并分别评价医学事实、视觉依据与推理质量。</p>
</body></html>"""


def build_blind_panel(
    *, stage2: dict[int, dict[str, Any]], stage3: dict[int, dict[str, Any]],
    indices: list[int], output: Path, seed: int, labels: tuple[str, str]
) -> dict[str, Any]:
    review = output / "review_packet"
    key_dir = output / "adjudication_key_keep_separate"
    review.mkdir(parents=True, exist_ok=False)
    key_dir.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    key_rows: list[dict[str, Any]] = []
    ratings: list[dict[str, Any]] = []
    for serial, index in enumerate(indices, 1):
        a, b = stage2[index], stage3[index]
        verify_pair(a, b, index)
        stage2_is_a = bool(rng.getrandbits(1))
        response_a = a["completion"] if stage2_is_a else b["completion"]
        response_b = b["completion"] if stage2_is_a else a["completion"]
        case_id = f"case_{serial:03d}"
        directory = review / case_id
        directory.mkdir()
        image_name = copy_image(a, directory)
        reference = f"正确选项：{a['target_choice']}\n\n{a.get('solution', '')}"
        (directory / "review.html").write_text(
            case_page(image_name=image_name, problem=a["problem"], reference=reference,
                      response_a=response_a, response_b=response_b), encoding="utf-8"
        )
        write_json(directory / "case.json", {
            "case_id": case_id, "problem": a["problem"], "target_choice": a["target_choice"],
            "reference_reasoning": a.get("solution"), "response_a": response_a,
            "response_b": response_b, "image_file": image_name,
        })
        key_rows.append({
            "case_id": case_id, "source_index": index,
            "response_a_model": labels[0] if stage2_is_a else labels[1],
            "response_b_model": labels[1] if stage2_is_a else labels[0],
            "stage2_correct": int(correct(a)), "stage3_correct": int(correct(b)),
            "source_record_sha256": a["source_record_sha256"],
        })
        ratings.append({
            "case_id": case_id, "preferred_response_A_B_Tie": "", "confidence_1_to_5": "",
            "A_medical_factuality_1_to_5": "", "B_medical_factuality_1_to_5": "",
            "A_visual_grounding_1_to_5": "", "B_visual_grounding_1_to_5": "",
            "A_reasoning_quality_1_to_5": "", "B_reasoning_quality_1_to_5": "", "notes": "",
        })
    for path, rows in ((review / "ratings_template.csv", ratings), (key_dir / "answer_key.csv", key_rows)):
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    (review / "README.md").write_text(
        "# Stage2 vs Stage3 盲化人工比较\n\n"
        "逐一打开每个子目录中的 `review.html`，填写 `ratings_template.csv`。"
        "A/B 顺序已随机化；评审者不要查看同级目录之外的 answer key。\n\n"
        "主指标：A/B/Tie 偏好率；次指标：医学事实、视觉依据、推理质量（1–5）。"
        "建议两名评审独立评分，并对分歧项仲裁。\n",
        encoding="utf-8",
    )
    return {"case_count": len(indices), "seed": seed, "selection": "output-independent frozen panel"}


def build_improvements(
    *, stage2: dict[int, dict[str, Any]], stage3: dict[int, dict[str, Any]],
    output: Path, labels: tuple[str, str]
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    common = sorted(set(stage2) & set(stage3))
    strata = {"wrong_to_right": [], "right_to_wrong": [], "both_right": [], "both_wrong": []}
    for index in common:
        a, b = stage2[index], stage3[index]
        verify_pair(a, b, index)
        key = "both_right" if correct(a) and correct(b) else "right_to_wrong" if correct(a) else "wrong_to_right" if correct(b) else "both_wrong"
        strata[key].append(index)
    for serial, index in enumerate(strata["wrong_to_right"], 1):
        a, b = stage2[index], stage3[index]
        directory = output / f"improved_{serial:03d}_source_{index:03d}"
        directory.mkdir()
        image_name = copy_image(a, directory)
        write_json(directory / "case.json", {
            "selection": "Stage2 wrong and Stage3 correct; descriptive case, not an unbiased estimate",
            "source_index": index, "image_file": image_name, "problem": a["problem"],
            "reference": a.get("solution"), "target_choice": a["target_choice"],
            labels[0]: {"completion": a["completion"], "predicted_choice": a.get("predicted_choice")},
            labels[1]: {"completion": b["completion"], "predicted_choice": b.get("predicted_choice")},
        })
        (directory / "README.md").write_text(
            f"# 改善案例 {serial:03d}\n\n![image]({image_name})\n\n"
            f"## 问题\n\n{a['problem']}\n\n## 参考\n\n{a.get('solution','')}\n\n"
            f"## {labels[0]}（错误）\n\n{a['completion']}\n\n"
            f"## {labels[1]}（正确）\n\n{b['completion']}\n",
            encoding="utf-8",
        )
    summary = {"total": len(common), **{key: len(value) for key, value in strata.items()}}
    write_json(output / "summary.json", summary)
    (output / "README.md").write_text(
        "# Stage2 → Stage3 错转对案例\n\n"
        "本目录只包含 Stage2 错、Stage3 对的案例，用于定性分析改善机制。"
        "它是按结果筛选的，不能用于估计 Stage3 的总体胜率或显著性。\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--judge-aligned-stage3", type=Path, required=True)
    parser.add_argument("--current-stage3", type=Path, required=True)
    parser.add_argument("--frozen-panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    stage2 = load_jsonl(args.stage2)
    aligned = load_jsonl(args.judge_aligned_stage3)
    current = load_jsonl(args.current_stage3)
    panel = json.loads(args.frozen_panel.read_text(encoding="utf-8"))
    indices = [int(value) for value in panel["indices"]]
    manifest = {
        "schema_version": 1,
        "purpose": "Stage2-versus-Stage3 human comparison and concrete improvement review",
        "inputs": {
            "stage2": {"path": str(args.stage2.resolve()), "sha256": sha256_file(args.stage2)},
            "judge_aligned_stage3": {"path": str(args.judge_aligned_stage3.resolve()), "sha256": sha256_file(args.judge_aligned_stage3)},
            "current_stage3": {"path": str(args.current_stage3.resolve()), "sha256": sha256_file(args.current_stage3)},
            "frozen_panel": {"path": str(args.frozen_panel.resolve()), "sha256": sha256_file(args.frozen_panel)},
        },
    }
    manifest["blind_pairwise"] = build_blind_panel(
        stage2=stage2, stage3=aligned, indices=indices, output=args.output / "blind_pairwise_multijudge_panel100",
        seed=args.seed, labels=("Stage2-outcome-GRPO", "Stage3-GPT4o-selected"),
    )
    manifest["improvements_multijudge_aligned"] = build_improvements(
        stage2=stage2, stage3=aligned, output=args.output / "improvements_multijudge_aligned_stage3",
        labels=("Stage2-outcome-GRPO", "Stage3-GPT4o-selected"),
    )
    manifest["improvements_current_n8_checkpoint1500"] = build_improvements(
        stage2=stage2, stage3=current, output=args.output / "improvements_current_n8_checkpoint1500",
        labels=("Stage2-outcome-GRPO", "Stage3-GPT4o-n8-checkpoint1500"),
    )
    write_json(args.output / "manifest.json", manifest)
    (args.output / "README.md").write_text(
        "# Stage2 / Stage3 人工复核材料\n\n"
        "- `blind_pairwise_multijudge_panel100/review_packet`：与三模型 Judge 相同的、"
        "不依赖输出正确性的100例盲化人工A/B比较。\n"
        "- `improvements_multijudge_aligned_stage3`：与现有多Judge统计同一Stage3模型的错转对案例。\n"
        "- `improvements_current_n8_checkpoint1500`：当前n=8 Stage3 checkpoint1500的错转对案例。\n\n"
        "改善案例仅用于说明，不得据此计算总体偏好；总体人类偏好应使用盲化100例。\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
