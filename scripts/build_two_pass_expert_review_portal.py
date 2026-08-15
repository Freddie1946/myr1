#!/usr/bin/env python3
"""Build a local two-pass expert-review portal with optional external advice."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
from typing import Any


STYLE = """body{font-family:system-ui,sans-serif;max-width:1180px;margin:2rem auto;padding:0 1rem;line-height:1.55;color:#202124}
table{border-collapse:collapse;width:100%}th,td{border:1px solid #aaa;padding:.45rem;vertical-align:top}pre{white-space:pre-wrap;background:#f6f6f6;padding:1rem}
.warning{background:#fff2cc;border-left:5px solid #d49b00;padding:1rem}.ref{background:#eef6ff;border-left:5px solid #3977b7;padding:1rem}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:.6rem}a{color:#1457a6}"""


def page(title: str, body: str) -> str:
    return f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{STYLE}</style></head><body><h1>{html.escape(title)}</h1>{body}</body></html>'


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, fields: list[str], ids: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case_id in ids:
            row = {field: "" for field in fields}
            row["case_id"] = case_id
            writer.writerow(row)


def list_regions(regions: list[dict[str, Any]]) -> str:
    if not regions:
        return "<p>未提出区域。</p>"
    rows = []
    for number, region in enumerate(regions, 1):
        rows.append(
            "<tr><td>" + str(number) + "</td><td>" + html.escape(str(region.get("box")))
            + "</td><td>" + html.escape(str(region.get("feature", ""))) + "</td><td>"
            + html.escape(str(region.get("role", ""))) + "</td><td>"
            + html.escape(str(region.get("importance", ""))) + "</td></tr>"
        )
    return "<table><tr><th>#</th><th>box (0–1000)</th><th>feature</th><th>role</th><th>importance</th></tr>" + "".join(rows) + "</table>"


def build_reward(root: Path, references_path: Path) -> None:
    output = root / "reward_reference_after_first_pass"
    output.mkdir(parents=True)
    refs = load_jsonl(references_path)
    links = []
    for row in refs:
        case_id = row["case_id"]
        result = row["reference"]
        table = "".join(
            f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>"
            for key, value in result.items() if key != "overall_reason"
        )
        body = f'''<p><a href="../index.html">返回二阶段目录</a></p>
<div class="warning">仅在已经保存本例独立评分后查看。此处是 Claude Sonnet 4.6 的外部参考，不是病理专家真值，也没有读取训练期 GPT-4o 奖励。</div>
<h2>结构化参考</h2><table>{table}</table><h2>参考理由</h2><div class="ref">{html.escape(result['overall_reason'])}</div>'''
        (output / f"{case_id}.html").write_text(page(f"{case_id} 外部参考", body), encoding="utf-8")
        links.append(f'<a href="{case_id}.html">{case_id}</a>')
    fields = ["case_id", "reviewer_id", "first_pass_saved_yes_no", "changed_any_rating_after_reference_yes_no", "changed_fields", "reference_helpfulness_1_to_5", "post_reference_notes"]
    write_csv(output / "post_reference_change_log.csv", fields, [row["case_id"] for row in refs])
    (output / "index.html").write_text(page("奖励复核：二阶段外部参考", '<div class="warning">先完成并保存第一阶段 CSV，再打开对应病例。主要统计必须使用第一阶段评分；改判另行报告。</div><div class="grid">' + "".join(links) + "</div>"), encoding="utf-8")


def build_roi(root: Path, effective_path: Path, mapping_path: Path, gemini_path: Path, opus_path: Path) -> None:
    output = root / "roi_reference_after_first_pass"
    output.mkdir(parents=True)
    effective = json.loads(effective_path.read_text(encoding="utf-8"))
    by_panel = {int(row["panel_index"]): row for row in effective["cases"]}
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))["roi_cases"]
    gemini = {int(row["panel_index"]): row["annotation"] for row in load_jsonl(gemini_path)}
    opus = {int(row["panel_index"]): row["annotation"] for row in load_jsonl(opus_path)}
    links = []
    for item in mapping:
        case_id = item["case_id"]
        panel_index = int(item["source_panel_index"])
        case = by_panel[panel_index]
        ext = case["external_annotation"]
        g = gemini[panel_index]
        o = opus[panel_index]
        consensus = list_regions(ext["regions"])
        body = f'''<p><a href="../index.html">返回二阶段目录</a></p>
<div class="warning">仅在独立完成整题及逐框评分后查看。两套意见均为外部 VLM pseudo-reference，不是专家标注。</div>
<h2>双模型共识区域</h2>{consensus}
<h2>Gemini 3.1 Pro 独立意见（confidence={g.get('confidence')}）</h2><div class="ref">{html.escape(str(g.get('rationale','')))}</div>{list_regions(g.get('regions', []))}
<h2>Claude Opus 5 独立意见（confidence={o.get('confidence')}）</h2><div class="ref">{html.escape(str(o.get('rationale','')))}</div>{list_regions(o.get('regions', []))}
<p>提示：外部模型未看到目标模型的 attention、删除效应或 activation patching 结果。</p>'''
        (output / f"{case_id}.html").write_text(page(f"{case_id} 区域参考", body), encoding="utf-8")
        links.append(f'<a href="{case_id}.html">{case_id}</a>')
    fields = ["case_id", "reviewer_id", "first_pass_saved_yes_no", "changed_any_rating_after_reference_yes_no", "changed_fields_or_regions", "reference_helpfulness_1_to_5", "post_reference_notes"]
    write_csv(output / "post_reference_change_log.csv", fields, [row["case_id"] for row in mapping])
    (output / "index.html").write_text(page("可解释性区域：二阶段外部参考", '<div class="warning">第一阶段评分保存后再看。主分析采用第一阶段人工评分；参考后的改动作为敏感性/仲裁记录。</div><div class="grid">' + "".join(links) + "</div>"), encoding="utf-8")


def judge_map(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    rows = load_jsonl(path)
    selected = {}
    for row in rows:
        if int(row["run"]) == 0 and row["candidate_label"] in {"Stage2-outcome-GRPO", "Stage3-GPT4o-selected"}:
            selected[(int(row["index"]), row["candidate_label"])] = row["scores"]
    return selected


def score_table(scores: dict[str, Any]) -> str:
    order = ("r_acc", "k_acc", "rigor", "professionalism", "clarity", "conciseness")
    return "<table>" + "".join(f"<tr><th>{key}</th><td>{float(scores[key]):.2f}</td></tr>" for key in order) + "</table><div class=\"ref\">" + html.escape(str(scores["overall_reason"])) + "</div>"


def build_generation(root: Path, answer_key_path: Path, claude_path: Path, gemini_path: Path) -> None:
    output = root / "generation_reference_after_first_pass"
    output.mkdir(parents=True)
    claude = judge_map(claude_path)
    gemini = judge_map(gemini_path)
    with answer_key_path.open(encoding="utf-8-sig") as handle:
        keys = list(csv.DictReader(handle))
    links = []
    for item in keys:
        case_id = item["case_id"]
        source_index = int(item["source_index"])
        cards = []
        for anonymous in ("A", "B"):
            model = item[f"response_{anonymous.lower()}_model"]
            c = claude[(source_index, model)]
            g = gemini[(source_index, model)]
            cards.append(f'''<h2>匿名回答 {anonymous}</h2><h3>Claude Sonnet 4.6 参考</h3>{score_table(c)}<h3>Gemini 3.1 Pro 参考</h3>{score_table(g)}''')
        body = '<p><a href="../index.html">返回二阶段目录</a></p><div class="warning">仅在已经保存 A/B/Tie 与各维度独立评分后查看。响应模型身份仍不显示；以下仅是两个外部 Judge 的建议。</div>' + "".join(cards)
        (output / f"{case_id}.html").write_text(page(f"{case_id} 生成质量参考", body), encoding="utf-8")
        links.append(f'<a href="{case_id}.html">{case_id}</a>')
    fields = ["case_id", "reviewer_id", "first_pass_saved_yes_no", "first_pass_preference_A_B_Tie", "post_reference_preference_A_B_Tie", "preference_changed_yes_no", "reference_helpfulness_1_to_5", "post_reference_notes"]
    write_csv(output / "post_reference_change_log.csv", fields, [row["case_id"] for row in keys])
    (output / "index.html").write_text(page("生成质量盲评：二阶段外部参考", '<div class="warning">必须先冻结第一阶段 A/B/Tie。外部意见不可替代人工评分；请记录是否改判。</div><div class="grid">' + "".join(links) + "</div>"), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-pass", type=Path, required=True)
    parser.add_argument("--reward-references", type=Path, required=True)
    parser.add_argument("--effective-panel", type=Path, required=True)
    parser.add_argument("--source-mapping", type=Path, required=True)
    parser.add_argument("--gemini-roi", type=Path, required=True)
    parser.add_argument("--opus-roi", type=Path, required=True)
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--claude-judgments", type=Path, required=True)
    parser.add_argument("--gemini-judgments", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    relative_target = os.path.relpath(args.first_pass.resolve(), args.output.resolve())
    os.symlink(relative_target, args.output / "first_pass_blinded")
    second = args.output / "second_pass_external_references"
    second.mkdir()
    build_reward(second, args.reward_references)
    build_roi(second, args.effective_panel, args.source_mapping, args.gemini_roi, args.opus_roi)
    build_generation(second, args.answer_key, args.claude_judgments, args.gemini_judgments)
    first_body = '''<div class="warning">这里不显示任何外部参考。请先完成并保存对应 CSV，然后返回总目录进入第二阶段。</div><ol>
<li><a href="first_pass_blinded/reward_agreement_blinded/index.html">任务 A：奖励机制复核（60例）</a></li>
<li><a href="first_pass_blinded/interpretability_roi_blinded/index.html">任务 B：可解释性区域复核（20例）</a></li>
<li><a href="first_pass_generation.html">任务 C：生成质量盲评（100例）</a></li></ol>'''
    (args.output / "first_pass.html").write_text(page("第一阶段：独立盲评", first_body), encoding="utf-8")
    generation_links = "".join(
        f'<a href="first_pass_blinded/stage2_stage3_blind_pairwise_100/case_{number:03d}/review.html">case_{number:03d}</a>'
        for number in range(1, 101)
    )
    (args.output / "first_pass_generation.html").write_text(
        page("第一阶段：生成质量盲评", '<div class="warning">先独立填写盲化 ratings_template.csv；本页不显示外部 Judge 意见。</div><div class="grid">' + generation_links + "</div>"),
        encoding="utf-8",
    )
    body = '''<div class="warning"><strong>强制顺序：</strong>第一阶段独立评分并保存 CSV → 第二阶段查看外部参考 → 填写改判日志。论文主要人工统计只使用第一阶段。</div>
<ol><li><a href="first_pass.html">第一阶段：盲化专家入口</a></li><li><a href="second_pass_external_references/reward_reference_after_first_pass/index.html">奖励机制参考</a></li><li><a href="second_pass_external_references/roi_reference_after_first_pass/index.html">可解释性区域参考</a></li><li><a href="second_pass_external_references/generation_reference_after_first_pass/index.html">生成质量参考</a></li></ol>'''
    (args.output / "index.html").write_text(page("PathVLM-R1 两阶段专家复核入口", body), encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
