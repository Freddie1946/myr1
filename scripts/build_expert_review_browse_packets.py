#!/usr/bin/env python3
"""Build browser-friendly blinded packets for expert review.

The public reviewer tree deliberately excludes automated rewards, model/judge
identities, training-stage labels, saliency maps, perturbation outcomes and
activation-patching results.  A separate local-only mapping records source
provenance for later audit and must not be sent to reviewers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
body{{font-family:system-ui,-apple-system,sans-serif;max-width:1120px;margin:2rem auto;padding:0 1rem;line-height:1.55;color:#202124}}
img{{max-width:100%;height:auto;border:1px solid #bbb}} pre{{white-space:pre-wrap;background:#f5f5f5;padding:1rem;border-radius:.4rem}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:.45rem;text-align:left}}a{{color:#1457a6}}
.note{{background:#fff8dc;padding:.8rem;border-left:4px solid #d49b00}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:.7rem}}
</style></head><body><h1>{html.escape(title)}</h1>{body}</body></html>"""


def copy_verified(source: Path, target: Path, expected_hash: str | None = None) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, target)
    digest = sha256_file(target)
    if expected_hash and digest != expected_hash:
        raise ValueError(f"image hash mismatch: {source}")
    return digest


def reward_packet(source_dir: Path, output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [json.loads(line) for line in (source_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 60 or len({row["case_id"] for row in rows}) != 60:
        raise ValueError("reward-agreement source must contain 60 unique blinded cases")
    cases_root = output / "cases"
    cases_root.mkdir(parents=True)
    rating_fields = [
        "case_id", "reviewer_id", "image_feature_analysis_present", "option_elimination_present",
        "medical_knowledge_support_present", "histological_definition_error", "logical_contradiction",
        "outdated_or_incorrect_pathology_criterion", "pathology_reasoning_correctness_1_to_5",
        "image_grounding_1_to_5", "logic_consistency_1_to_5", "hallucination_present",
        "reference_answer_ambiguous_or_noisy", "reviewer_confidence_1_to_5", "reviewer_notes",
    ]
    ratings: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    links = []
    for row in sorted(rows, key=lambda item: item["case_id"]):
        case_id = row["case_id"]
        case_dir = cases_root / case_id
        case_dir.mkdir()
        source_image = source_dir / row["image_file"]
        image_name = "image" + source_image.suffix.lower()
        digest = copy_verified(source_image, case_dir / image_name, row.get("image_sha256"))
        rating = {field: "" for field in rating_fields}
        rating["case_id"] = case_id
        ratings.append(rating)
        question = html.escape(row["question_and_options"])
        reference = html.escape(row["dataset_reference_answer"])
        candidate = html.escape(row["candidate_completion"])
        body = f"""<p><a href="../../index.html">返回总目录</a></p><img src="{image_name}" alt="case image">
<h2>题目与选项</h2><pre>{question}</pre><h2>数据集参考答案</h2><pre>{reference}</pre>
<h2>待评分回答</h2><pre>{candidate}</pre><div class="note">请在根目录 ratings_template.csv 中填写本例评分。不要查阅模型身份或自动评分。</div>"""
        (case_dir / "index.html").write_text(page(case_id, body), encoding="utf-8")
        (case_dir / "case.json").write_text(json.dumps({
            "case_id": case_id, "image_file": image_name, "image_sha256": digest,
            "question_and_options": row["question_and_options"], "dataset_reference_answer": row["dataset_reference_answer"],
            "candidate_completion": row["candidate_completion"], "ratings": row.get("ratings", {}),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        links.append(f'<a href="cases/{case_id}/index.html">{case_id}</a>')
        audit.append({"case_id": case_id, "source_case_id": case_id, "source_image": str(source_image.resolve())})
    write_csv(output / "ratings_template.csv", rating_fields, ratings)
    write_csv(output / "reviewer_A_all60.csv", ["case_id"], [{"case_id": row["case_id"]} for row in ratings])
    write_csv(output / "reviewer_B_suggested_overlap30.csv", ["case_id"], [{"case_id": row["case_id"]} for row in ratings[:30]])
    instructions = """<p class="note">这是盲化的奖励可信度复核包。Reviewer A 评全部 60 例；Reviewer B 至少独立评建议的 30 例重叠集。代表性样本与挑战样本必须在解盲后分别统计。</p>
<p>布尔事件填写 true / false / uncertain；三个质量维度和信心填写 1–5。候选回答应同时根据图像、题目、选项与病理知识判断。</p>"""
    (output / "index.html").write_text(page("奖励评分专家复核（60例）", instructions + '<div class="grid">' + "".join(links) + "</div>"), encoding="utf-8")
    return ratings, audit


def scaled_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    if len(box) != 4 or any(not isinstance(value, (int, float)) for value in box):
        raise ValueError(f"invalid box: {box}")
    x1, y1, x2, y2 = box
    # Annotation coordinates are normalized to a 0..1000 canvas.
    values = (round(x1 * width / 1000), round(y1 * height / 1000), round(x2 * width / 1000), round(y2 * height / 1000))
    px1, py1, px2, py2 = values
    px1, px2 = sorted((max(0, min(width - 1, px1)), max(1, min(width, px2))))
    py1, py2 = sorted((max(0, min(height - 1, py1)), max(1, min(height, py2))))
    if px2 <= px1 or py2 <= py1:
        raise ValueError(f"degenerate scaled box: {box} -> {values}")
    return px1, py1, px2, py2


def annotated_image(source: Path, target: Path, boxes: list[list[float]]) -> list[dict[str, Any]]:
    with Image.open(source) as original:
        image = original.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    rendered = []
    width, height = image.size
    for number, box in enumerate(boxes, 1):
        pixels = scaled_box(box, width, height)
        color = (230, 35, 50)
        line_width = max(2, round(min(width, height) / 180))
        draw.rectangle(pixels, outline=color, width=line_width)
        label = f"R{number}"
        label_box = draw.textbbox((pixels[0], pixels[1]), label, font=font)
        draw.rectangle(label_box, fill=(255, 255, 255))
        draw.text((pixels[0], pixels[1]), label, fill=color, font=font)
        rendered.append({"region_id": label, "box_normalized_0_1000": box, "box_pixels": list(pixels)})
    image.save(target, quality=95)
    return rendered


def roi_packet(source_json: Path, output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(source_json.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if len(cases) != 20:
        raise ValueError("interpretability source must contain the frozen 20-case panel")
    cases_root = output / "cases"
    cases_root.mkdir(parents=True)
    case_fields = [
        "case_id", "reviewer_id", "reference_answer_valid_yes_partial_no", "candidate_regions_overall_relevance_1_to_5",
        "candidate_regions_coverage_all_most_some_none", "candidate_regions_specificity_1_to_5",
        "missing_diagnostic_region_yes_no", "roi_edit_required_yes_no", "reviewer_confidence_1_to_5", "reviewer_notes",
    ]
    region_fields = ["case_id", "region_id", "reviewer_id", "diagnostically_relevant_yes_partial_no", "boundary_quality_1_to_5", "region_notes"]
    case_ratings: list[dict[str, Any]] = []
    region_ratings: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    links = []
    for ordinal, row in enumerate(cases, 1):
        case_id = f"EIR-{ordinal:03d}"
        case_dir = cases_root / case_id
        case_dir.mkdir()
        source_image = Path(row["image"])
        original_name = "image_original" + source_image.suffix.lower()
        digest = copy_verified(source_image, case_dir / original_name, row.get("image_sha256"))
        boxes = row["external_annotation"]["boxes"]
        rendered = annotated_image(source_image, case_dir / "image_candidate_regions.jpg", boxes)
        case_rating = {field: "" for field in case_fields}; case_rating["case_id"] = case_id
        case_ratings.append(case_rating)
        for region in rendered:
            rating = {field: "" for field in region_fields}; rating.update({"case_id": case_id, "region_id": region["region_id"]})
            region_ratings.append(rating)
        body = f"""<p><a href="../../index.html">返回总目录</a></p>
<h2>原图</h2><img src="{original_name}" alt="original image"><h2>待复核候选区域</h2>
<img src="image_candidate_regions.jpg" alt="candidate regions"><h2>题目与选项</h2><pre>{html.escape(row['problem'])}</pre>
<h2>数据集参考答案</h2><pre>{html.escape(row['solution'])}</pre>
<div class="note">请判断各候选框是否包含诊断相关证据、边界是否合理、是否漏掉关键区域。此包不展示目标模型预测、热图或删除结果。</div>"""
        (case_dir / "index.html").write_text(page(case_id, body), encoding="utf-8")
        public_case = {
            "case_id": case_id, "image_original": original_name, "image_sha256": digest,
            "image_candidate_regions": "image_candidate_regions.jpg", "question_and_options": row["problem"],
            "dataset_reference_answer": row["solution"], "regions": rendered,
        }
        (case_dir / "case.json").write_text(json.dumps(public_case, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        links.append(f'<a href="cases/{case_id}/index.html">{case_id}</a>')
        audit.append({"case_id": case_id, "source_panel_index": row.get("panel_index"), "source_record_index": row.get("index"), "source_image": str(source_image.resolve())})
    write_csv(output / "case_ratings_template.csv", case_fields, case_ratings)
    write_csv(output / "region_ratings_template.csv", region_fields, region_ratings)
    instructions = """<p class="note">这是可解释性 ROI 专家盲审包。候选区域在目标模型运行前由两个外部闭源视觉模型独立提出并交叉确认，但仍只是 pseudo-reference，不是专家真值。</p>
<p>请先判断参考答案是否成立，再逐框评价诊断相关性和边界质量，并记录遗漏区域或需要编辑的框。不要查阅目标模型热图、预测、置信度或删除效应。</p>"""
    (output / "index.html").write_text(page("可解释性 ROI 专家复核（20例）", instructions + '<div class="grid">' + "".join(links) + "</div>"), encoding="utf-8")
    return case_ratings + region_ratings, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-source", type=Path, required=True, help="existing blinded_packet directory")
    parser.add_argument("--roi-source", type=Path, required=True, help="frozen 20-case JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    reward_out = output / "reward_agreement_blinded"
    roi_out = output / "interpretability_roi_blinded"
    reward_out.mkdir(parents=True)
    roi_out.mkdir(parents=True)
    reward_ratings, reward_audit = reward_packet(args.reward_source.resolve(), reward_out)
    roi_ratings, roi_audit = roi_packet(args.roi_source.resolve(), roi_out)
    internal = output / "INTERNAL_DO_NOT_SEND_TO_REVIEWERS"
    internal.mkdir()
    (internal / "source_mapping.json").write_text(json.dumps({
        "reward_cases": reward_audit, "roi_cases": roi_audit,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    top_body = """<p>本目录包含两个相互独立的专家复核任务。发给专家时只发送相应的 <code>*_blinded</code> 子目录；不要发送 <code>INTERNAL_DO_NOT_SEND_TO_REVIEWERS</code>。</p>
<ul><li><a href="reward_agreement_blinded/index.html">奖励评分复核（60例）</a></li>
<li><a href="interpretability_roi_blinded/index.html">可解释性 ROI 复核（20例）</a></li></ul>"""
    (output / "index.html").write_text(page("PathVLM-R1 专家复核包", top_body), encoding="utf-8")
    file_hashes = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "reward_case_count": 60, "roi_case_count": 20,
        "reward_rating_row_count": len(reward_ratings), "roi_rating_row_count": len(roi_ratings),
        "blinding": {
            "reward": "excludes judge/model identity, automated rewards, stage and sampling stratum",
            "roi": "excludes target prediction, saliency, deletion effect and activation patching",
        },
        "file_count": len(file_hashes), "file_sha256": file_hashes,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **{k: manifest[k] for k in ("reward_case_count", "roi_case_count", "file_count")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
