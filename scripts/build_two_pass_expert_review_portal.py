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


STYLE = """body{font-family:system-ui,sans-serif;max-width:1380px;margin:2rem auto;padding:0 1rem;line-height:1.55;color:#202124}
table{border-collapse:collapse;width:100%}th,td{border:1px solid #aaa;padding:.45rem;vertical-align:top}pre{white-space:pre-wrap;background:#f6f6f6;padding:1rem}
.warning{background:#fff2cc;border-left:5px solid #d49b00;padding:1rem}.ref{background:#eef6ff;border-left:5px solid #3977b7;padding:1rem}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:.6rem}a{color:#1457a6}.nav{display:flex;justify-content:space-between;gap:1rem;position:sticky;top:0;background:white;padding:.7rem 0;z-index:2}.panes{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:1rem}.panes iframe{width:100%;height:900px;border:1px solid #999}@media(max-width:900px){.panes{grid-template-columns:1fr}}"""


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


def assisted_rubric(task: str) -> str:
    if task == "reward":
        return '''<h2>与训练奖励一致的评分规则</h2>
<p>只对六个奖励事件作布尔判断（true / false / uncertain）：</p>
<ol><li>正向完整性事件：图像特征分析、选项排除、医学知识支持；</li>
<li>负向错误事件：组织学定义错误、逻辑矛盾、错误或过时病理标准。</li></ol>
<p>令 <code>m</code> 为三个正向事件中缺失数，<code>e</code> 为三个负向事件中出现数：</p>
<pre>S_integrity = max(0, 1 - 0.4*m)
S_knowledge = max(0, 1 - 0.4*e)
R_process = (S_integrity + S_knowledge) / 2</pre>
<p>每个子量表的 0/1/2/3 个失败对应 1.0/0.6/0.2/0.0。另填病理推理正确性、图像 grounding、逻辑一致性及信心（1–5）。幻觉和参考答案歧义是补充审计项，不进入上述六事件奖励公式。</p>'''
    if task == "roi":
        return '''<h2>ROI 评分规则</h2><p>整题填写：参考答案有效性（yes/partial/no）、总体病理相关性 1–5、覆盖度（all/most/some/none）、特异性 1–5、是否漏标、是否需要修框、信心 1–5。逐框填写：诊断相关性（yes/partial/no）和边界质量 1–5。外部框仍是 pseudo-reference；模型意见不能替代专家判断。</p>'''
    return '''<h2>生成质量评分规则</h2><p>先给匿名回答 A/B/Tie 偏好及信心 1–5；再分别给 A/B 的医学事实正确性、视觉依据充分性、推理质量各 1–5。不能因回答更长而加分，也不能只看最终选项。两者同样好或同样差应选 Tie。</p>'''


def build_assisted_review(root: Path) -> None:
    assisted = root / "assisted_review_with_external_references"
    assisted.mkdir(exist_ok=True)
    specs = {
        "reward": {
            "title": "奖励机制辅助复核",
            "ids": [f"HRA-{number:03d}" for number in range(1, 61)],
            "blind": lambda case_id: f"../../first_pass_blinded/reward_agreement_blinded/cases/{case_id}/index.html",
            "reference": lambda case_id: f"../../second_pass_external_references/reward_reference_after_first_pass/{case_id}.html",
            "csv": "../../first_pass_blinded/reward_agreement_blinded/ratings_template.csv",
        },
        "roi": {
            "title": "可解释性区域辅助复核",
            "ids": [f"EIR-{number:03d}" for number in range(1, 21)],
            "blind": lambda case_id: f"../../first_pass_blinded/interpretability_roi_blinded/cases/{case_id}/index.html",
            "reference": lambda case_id: f"../../second_pass_external_references/roi_reference_after_first_pass/{case_id}.html",
            "csv": "../../first_pass_blinded/interpretability_roi_blinded/case_ratings_template.csv",
        },
        "generation": {
            "title": "Stage2/Stage3 生成质量辅助盲评",
            "ids": [f"case_{number:03d}" for number in range(1, 101)],
            "blind": lambda case_id: f"../../first_pass_blinded/stage2_stage3_blind_pairwise_100/{case_id}/review.html",
            "reference": lambda case_id: f"../../second_pass_external_references/generation_reference_after_first_pass/{case_id}.html",
            "csv": "../../first_pass_blinded/stage2_stage3_blind_pairwise_100/ratings_template.csv",
        },
    }
    root_links = []
    for task, spec in specs.items():
        task_root = assisted / task
        task_root.mkdir(exist_ok=True)
        ids = spec["ids"]
        links = []
        for position, case_id in enumerate(ids):
            previous_link = f'<a href="{ids[position - 1]}.html">← 上一题</a>' if position else '<span>← 上一题</span>'
            next_link = f'<a href="{ids[position + 1]}.html">下一题 →</a>' if position + 1 < len(ids) else '<span>下一题 →</span>'
            nav = f'<div class="nav">{previous_link}<a href="index.html">题目目录</a>{next_link}</div>'
            body = f'''{nav}<div class="warning"><strong>外部参考辅助模式：</strong>候选模型/训练阶段仍盲化，但专家从一开始就能看到外部模型意见。因此本入口结果必须标为 reference-assisted，不能用于“独立专家评分”主统计。</div>
{assisted_rubric(task)}<p><a href="{spec['csv']}">下载该任务评分 CSV 模板</a></p>
<div class="panes"><section><h2>病例与待评分内容</h2><iframe src="{spec['blind'](case_id)}"></iframe></section><section><h2>外部模型参考意见</h2><iframe src="{spec['reference'](case_id)}"></iframe></section></div>{nav}'''
            (task_root / f"{case_id}.html").write_text(page(f"{spec['title']}：{case_id}", body), encoding="utf-8")
            links.append(f'<a href="{case_id}.html">{case_id}</a>')
        index_body = f'''<p><a href="../index.html">返回辅助评审总目录</a></p><div class="warning">本任务含外部参考，结果属于 reference-assisted expert review。</div>{assisted_rubric(task)}<p><a href="{spec['csv']}">下载评分 CSV 模板</a></p><div class="grid">{"".join(links)}</div>'''
        (task_root / "index.html").write_text(page(spec["title"], index_body), encoding="utf-8")
        root_links.append(f'<li><a href="{task}/index.html">{spec["title"]}</a></li>')
    assisted_body = '<div class="warning">此入口直接显示外部参考，适合辅助作答或仲裁，不适合作为独立盲评主统计。</div><ol>' + "".join(root_links) + "</ol>"
    (assisted / "index.html").write_text(page("外部参考辅助专家评审", assisted_body), encoding="utf-8")


def write_root_index(root: Path) -> None:
    body = '''<div class="warning"><strong>请选择评审模式：</strong>独立盲评用于论文主要人工统计；辅助模式从第一题开始显示外部意见，必须单独标注为 reference-assisted。</div>
<ol><li><a href="first_pass.html">独立盲评入口（不显示外部参考）</a></li><li><a href="assisted_review_with_external_references/index.html">外部参考辅助评审入口（含上一题/下一题）</a></li><li><a href="second_pass_external_references/reward_reference_after_first_pass/index.html">原两阶段：奖励参考</a></li><li><a href="second_pass_external_references/roi_reference_after_first_pass/index.html">原两阶段：ROI 参考</a></li><li><a href="second_pass_external_references/generation_reference_after_first_pass/index.html">原两阶段：生成质量参考</a></li></ol>'''
    (root / "index.html").write_text(page("PathVLM-R1 专家复核入口", body), encoding="utf-8")


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
    parser.add_argument("--augment-existing", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        if not args.augment_existing:
            raise FileExistsError(args.output)
        build_assisted_review(args.output)
        write_root_index(args.output)
        print(json.dumps({"status": "augmented", "output": str(args.output.resolve())}, ensure_ascii=False))
        return
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
    build_assisted_review(args.output)
    write_root_index(args.output)
    print(json.dumps({"status": "completed", "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
