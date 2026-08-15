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

from generate_bilingual_expert_review_translations import source_records


STYLE = """body{font-family:system-ui,sans-serif;max-width:1380px;margin:2rem auto;padding:0 1rem;line-height:1.55;color:#202124}
table{border-collapse:collapse;width:100%}th,td{border:1px solid #aaa;padding:.45rem;vertical-align:top}pre{white-space:pre-wrap;background:#f6f6f6;padding:1rem}
.warning{background:#fff2cc;border-left:5px solid #d49b00;padding:1rem}.ref{background:#eef6ff;border-left:5px solid #3977b7;padding:1rem}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:.6rem}a{color:#1457a6}.nav{display:flex;justify-content:space-between;gap:1rem;position:sticky;top:0;background:white;padding:.7rem 0;z-index:2}.panes{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:1rem}.panes iframe{width:100%;height:900px;border:1px solid #999}.event-card{border:1px solid #bbb;border-radius:8px;padding:1rem;margin:.8rem 0}.choices{display:flex;gap:1.5rem;margin:.7rem 0}.choices label{font-size:1.1rem}.score-box{background:#eef9ef;border-left:5px solid #3a8b46;padding:1rem;margin:1rem 0}button{padding:.7rem 1.2rem;font-size:1rem;cursor:pointer}input[type=text]{padding:.5rem;min-width:260px}.save-status{font-weight:600;margin-left:1rem}@media(max-width:900px){.panes{grid-template-columns:1fr}}"""

REWARD_FIELDS = {
    "image_feature_analysis_present": ("包含图像特征分析", "Image-feature analysis present", "回答指出图像中可见的具体特征，并用于支持判断；仅写泛泛的“图像异常”不够。"),
    "option_elimination_present": ("包含选项排除", "Option elimination present", "回答显式或隐式比较、排除至少一个替代选项；不要求固定排除数量。"),
    "medical_knowledge_support_present": ("包含医学知识支持", "Medical-knowledge support present", "回答使用正确病理知识把图像征象与结论连接起来，而非只堆砌术语。"),
    "histological_definition_error": ("存在组织学定义错误", "Histological-definition error", "错误定义细胞、组织结构、形态或染色等病理概念。"),
    "logical_contradiction": ("存在逻辑矛盾", "Logical contradiction", "推理内部互相矛盾，或所述证据与最终结论直接冲突。"),
    "outdated_or_incorrect_pathology_criterion": ("使用错误或过时病理标准", "Outdated or incorrect pathology criterion", "把错误、已废弃或不适用于本题的诊断标准作为判断依据。"),
    "pathology_reasoning_correctness_1_to_5": ("病理推理正确性（1–5）", "Pathology reasoning correctness (1–5)", "1=根本错误；2=重大错误；3=部分正确但有实质问题；4=基本正确，仅小问题；5=完整且正确。"),
    "image_grounding_1_to_5": ("图像依据充分性（1–5）", "Image grounding (1–5)", "1=无依据或描述错误；3=部分对应图像；5=准确、具体且充分引用可见证据。"),
    "logic_consistency_1_to_5": ("逻辑一致性（1–5）", "Logic consistency (1–5)", "1=严重矛盾；3=基本可跟随但有跳步；5=连贯且结论由证据推出。"),
    "hallucination_present": ("存在幻觉", "Hallucination present", "声称图中或医学背景存在并无依据的事实；该项是补充审计，不进入六事件奖励。"),
    "reference_answer_ambiguous_or_noisy": ("参考答案含歧义或噪声", "Reference answer ambiguous or noisy", "数据集参考答案本身可能不唯一、不充分或错误；该项不进入六事件奖励。"),
    "reviewer_confidence_1_to_5": ("评审信心（1–5）", "Reviewer confidence (1–5)", "1=非常不确定；3=中等；5=非常确定。"),
}

CORE_REWARD_FIELDS = (
    "image_feature_analysis_present",
    "option_elimination_present",
    "medical_knowledge_support_present",
    "histological_definition_error",
    "logical_contradiction",
    "outdated_or_incorrect_pathology_criterion",
)

JUDGE_FIELDS = {
    "r_acc": ("推理对答案的支持度", "Reasoning support for answer"),
    "k_acc": ("医学/病理知识准确性", "Medical/pathology knowledge accuracy"),
    "rigor": ("推理严谨性", "Reasoning rigor"),
    "professionalism": ("专业性", "Professionalism"),
    "clarity": ("清晰度", "Clarity"),
    "conciseness": ("简洁性", "Conciseness"),
}

ROLE_LABELS = {
    "supports_reference": "支持参考答案 / Supports reference",
    "opposes_distractor": "反驳干扰项 / Opposes distractor",
    "context": "背景信息 / Context",
}


def page(title: str, body: str) -> str:
    return f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{STYLE}</style></head><body><h1>{html.escape(title)}</h1>{body}</body></html>'


def bilingual_text(title_zh: str, title_en: str, english: str, chinese: str) -> str:
    return f'''<h2>{html.escape(title_zh)} / {html.escape(title_en)}</h2>
<h3>中文译文 / Chinese translation</h3><pre>{html.escape(chinese)}</pre>
<h3>英文原文 / English source</h3><pre>{html.escape(english)}</pre>'''


def boolean_label(value: Any) -> str:
    if value is True:
        return "是 / True"
    if value is False:
        return "否 / False"
    return "不确定 / Uncertain"


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
        field_rows = "".join(
            f"<tr><th>{html.escape(REWARD_FIELDS[field][0])} / {html.escape(REWARD_FIELDS[field][1])}</th>"
            f"<td>{html.escape(REWARD_FIELDS[field][2])}</td></tr>"
            for field in CORE_REWARD_FIELDS
        )
        return f'''<h2>与训练奖励一致的评分规则 / Scoring rubric aligned with the training reward</h2>
<p><strong>人工只填写下面六个布尔值（是/True 或否/False）。</strong>这六项与训练期 GPT-4o Judge 的事件 schema 完全一致；页面不再要求任何 1–5 分、幻觉、参考答案噪声或总体质量分。Experts enter exactly the same six booleans as the training-time Judge.</p>
<ol><li>正向完整性事件 / Positive integrity events：图像特征分析 / image-feature analysis、选项排除 / option elimination、医学知识支持 / medical-knowledge support；</li>
<li>负向错误事件 / Error events：组织学定义错误 / histological-definition error、逻辑矛盾 / logical contradiction、错误或过时病理标准 / incorrect or outdated criterion。</li></ol>
<p>令 <code>m</code> 为三个正向事件中缺失数，<code>e</code> 为三个负向事件中出现数。Let <code>m</code> be the number of missing positive events and <code>e</code> the number of present error events:</p>
<pre>S_integrity = max(0, 1 - 0.4*m)
S_knowledge = max(0, 1 - 0.4*e)
R_process = (S_integrity + S_knowledge) / 2</pre>
<p>每个子量表的 0/1/2/3 个失败对应 1.0/0.6/0.2/0.0。Zero/one/two/three failures map to 1.0/0.6/0.2/0.0.</p><table><tr><th>指标 / Field</th><th>含义 / Meaning</th></tr>{field_rows}</table>'''
    if task == "roi":
        return '''<h2>ROI 评分规则 / ROI scoring rubric</h2>
<p>整题填写 / Case-level fields：参考答案有效性 / reference validity（是/yes、部分/partial、否/no）、总体病理相关性 / overall relevance 1–5、覆盖度 / coverage（全部/all、大部分/most、部分/some、无/none）、特异性 / specificity 1–5、是否漏标 / missing region、是否需要修框 / edit required、信心 / confidence 1–5。逐框填写 / Per-region fields：诊断相关性 / diagnostic relevance（yes/partial/no）和边界质量 / boundary quality 1–5。</p>
<table><tr><th>外部参考字段 / External-reference field</th><th>含义 / Meaning</th></tr>
<tr><th>坐标 / box</th><td>按图像宽高归一化到 0–1000 的 [x1,y1,x2,y2]，不是原图像素 / [x1,y1,x2,y2] normalized to 0–1000, not source-image pixels.</td></tr>
<tr><th>特征 / feature</th><td>外部模型认为框内包含的病理征象 / Pathologic finding that the external model believes is inside the box.</td></tr>
<tr><th>作用 / role</th><td>该征象被认为支持参考答案、反驳干扰项或仅提供背景 / Whether the feature supports the reference, opposes a distractor, or provides context.</td></tr>
<tr><th>重要度 / importance [0,1]</th><td>外部模型估计该区域对本题判断的重要程度；不是概率，也未经校准 / Model-estimated relevance to this decision; not a probability and not calibrated.</td></tr>
<tr><th>信心 / confidence [0,1]</th><td>外部模型对整份区域标注的自评信心；不是专家一致率或正确概率 / External-model self-confidence for the annotation; not expert agreement or a calibrated correctness probability.</td></tr></table>
<p>外部框仍是伪参考区域 / pseudo-reference；模型意见不能替代专家判断 / model advice does not replace expert judgment.</p>'''
    judge_rows = "".join(
        f"<tr><th>{html.escape(zh)} / {html.escape(en)}</th><td>[0,1]；外部 Judge 对该维度的连续评分，越高越好；不是校准概率 / Continuous external-Judge score; higher is better and it is not a calibrated probability.</td></tr>"
        for zh, en in JUDGE_FIELDS.values()
    )
    return f'''<h2>生成质量评分规则 / Generation-quality scoring rubric</h2><p>先给匿名回答 A/B/平局（Tie）偏好及信心 1–5；再分别给 A/B 的医学事实正确性 / medical factuality、视觉依据充分性 / visual grounding、推理质量 / reasoning quality 各 1–5。不能因回答更长而加分，也不能只看最终选项 / Do not reward verbosity or judge only the final option. 两者同样好或同样差应选 Tie / Choose Tie when equally good or equally poor.</p>
<h3>外部 Judge 结构化参考 / Structured external-Judge reference</h3><table><tr><th>指标 / Metric</th><th>取值和含义 / Range and meaning</th></tr>{judge_rows}</table><p><code>r_acc</code> 专看推理是否支持最终答案；<code>k_acc</code> 专看医学知识是否正确。其余四项依次衡量严谨性、专业性、清晰度和简洁性。These fields separately score answer-supporting reasoning, medical knowledge, rigor, professionalism, clarity, and conciseness.</p>'''


def process_score(events: dict[str, bool]) -> float:
    missing = sum(not events[field] for field in CORE_REWARD_FIELDS[:3])
    errors = sum(events[field] for field in CORE_REWARD_FIELDS[3:])
    return (max(0.0, 1.0 - 0.4 * missing) + max(0.0, 1.0 - 0.4 * errors)) / 2.0


def reward_click_form(case_id: str, external_reference: dict[str, Any]) -> str:
    cards = []
    for field in CORE_REWARD_FIELDS:
        zh, en, explanation = REWARD_FIELDS[field]
        external = bool(external_reference[field])
        external_label = "是 / True" if external else "否 / False"
        cards.append(f'''<div class="event-card" data-field="{html.escape(field)}">
<h3>{html.escape(zh)} / {html.escape(en)}</h3><p>{html.escape(explanation)}</p>
<div class="choices"><label><input type="radio" name="{html.escape(field)}" value="true"> 是 / True</label><label><input type="radio" name="{html.escape(field)}" value="false"> 否 / False</label></div>
<div class="ref"><strong>Claude Sonnet 4.6 外部参考 / External reference：</strong>{external_label}</div></div>''')
    fields_json = json.dumps(list(CORE_REWARD_FIELDS), ensure_ascii=False)
    external_json = json.dumps({field: bool(external_reference[field]) for field in CORE_REWARD_FIELDS})
    external_score = process_score({field: bool(external_reference[field]) for field in CORE_REWARD_FIELDS})
    script = f'''<script>
const CASE_ID = {json.dumps(case_id)};
const FIELDS = {fields_json};
const EXTERNAL_EVENTS = {external_json};
const EXTERNAL_SCORE = {external_score:.12g};
function calculateScore(values) {{
  const missing = FIELDS.slice(0,3).filter(field => !values[field]).length;
  const errors = FIELDS.slice(3).filter(field => values[field]).length;
  return (Math.max(0, 1 - 0.4 * missing) + Math.max(0, 1 - 0.4 * errors)) / 2;
}}
function selectedEvents(requireAll=true) {{
  const values = {{}};
  for (const field of FIELDS) {{
    const selected = document.querySelector(`input[name="${{field}}"]:checked`);
    if (!selected) {{ if (requireAll) throw new Error(`尚未填写：${{field}}`); else return null; }}
    values[field] = selected.value === 'true';
  }}
  return values;
}}
function updateScore() {{
  const values = selectedEvents(false);
  const human = values === null ? null : calculateScore(values);
  document.getElementById('human-score').textContent = human === null ? '待六项填写完整' : human.toFixed(2);
  document.getElementById('external-score').textContent = EXTERNAL_SCORE.toFixed(2);
  document.getElementById('score-difference').textContent = human === null ? '待计算' : (human - EXTERNAL_SCORE).toFixed(2);
}}
async function loadSavedRating() {{
  const reviewer = document.getElementById('reviewer-id').value.trim();
  if (!reviewer) return;
  localStorage.setItem('pathvlm_reviewer_id', reviewer);
  const response = await fetch(`/api/reward-rating?reviewer_id=${{encodeURIComponent(reviewer)}}&case_id=${{encodeURIComponent(CASE_ID)}}`);
  if (response.status === 404) {{ document.getElementById('save-status').textContent = '本题尚未保存 / Not saved yet'; return; }}
  if (!response.ok) throw new Error(await response.text());
  const record = await response.json();
  for (const [field, value] of Object.entries(record.events)) {{
    const radio = document.querySelector(`input[name="${{field}}"][value="${{value ? 'true' : 'false'}}"]`);
    if (radio) radio.checked = true;
  }}
  document.getElementById('reviewer-notes').value = record.reviewer_notes || '';
  document.getElementById('save-status').textContent = `已恢复保存记录：${{record.saved_at}}`;
  updateScore();
}}
async function saveRating() {{
  const reviewer = document.getElementById('reviewer-id').value.trim();
  if (!reviewer) {{ alert('请先填写评审者编号 / Enter reviewer ID'); return; }}
  let events;
  try {{ events = selectedEvents(true); }} catch (error) {{ alert(error.message); return; }}
  localStorage.setItem('pathvlm_reviewer_id', reviewer);
  const response = await fetch('/api/reward-rating', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify({{reviewer_id:reviewer,case_id:CASE_ID,events:events,reviewer_notes:document.getElementById('reviewer-notes').value}})}});
  if (!response.ok) {{ alert(`保存失败 / Save failed: ${{await response.text()}}`); return; }}
  const result = await response.json();
  document.getElementById('save-status').textContent = `已写入文件 / Saved：${{result.saved_at}}；当前完成 ${{result.completed_cases}}/60`;
  updateScore();
}}
function exportRatings() {{
  const reviewer = document.getElementById('reviewer-id').value.trim();
  if (!reviewer) {{ alert('请先填写评审者编号 / Enter reviewer ID'); return; }}
  localStorage.setItem('pathvlm_reviewer_id', reviewer);
  window.location.href = `/api/export-reward-ratings?reviewer_id=${{encodeURIComponent(reviewer)}}`;
}}
window.addEventListener('DOMContentLoaded', () => {{
  document.getElementById('reviewer-id').value = localStorage.getItem('pathvlm_reviewer_id') || '';
  document.querySelectorAll('input[type=radio]').forEach(node => node.addEventListener('change', updateScore));
  updateScore();
  if (document.getElementById('reviewer-id').value) loadSavedRating().catch(error => document.getElementById('save-status').textContent = error.message);
}});
</script>'''
    return f'''<h2>六事件直接评分与保存 / Direct six-event rating</h2>
<div class="warning"><strong>Reviewer ID 使用方法：</strong>每位专家在自己的电脑上使用负责人预先分配的固定匿名编号，例如 <code>pathologist_A</code>、<code>pathologist_B</code>。第一次在第一题录入；后续浏览器会自动带出。中断后输入同一编号并点击“载入已保存评分”。历史评分只从本机解压目录的 <code>expert_submissions/&lt;Reviewer ID&gt;/</code> 读取，不访问远程服务器。不要填写真实姓名或邮箱；编号拼错会创建另一套本地记录。</div>
<p><label>评审者编号 / Reviewer ID：<input id="reviewer-id" type="text" maxlength="64" placeholder="例如 pathologist_A"></label> <button type="button" onclick="loadSavedRating()">载入本题已保存评分 / Load this saved case</button></p>
{''.join(cards)}
<p><label>可选备注（不参与奖励）/ Optional notes (not scored)：<br><textarea id="reviewer-notes" rows="3" style="width:100%" maxlength="2000"></textarea></label></p>
<div class="score-box"><strong>人工过程分 / Human process score：</strong><span id="human-score"></span><br><strong>外部 Claude 过程分 / External Claude process score：</strong><span id="external-score"></span><br><strong>人工−外部模型分差 / Human minus external difference：</strong><span id="score-difference"></span></div>
<button type="button" onclick="saveRating()">保存本题评分到本地文件 / Save this rating locally</button> <button type="button" onclick="exportRatings()">导出我的评分包 / Export my ratings ZIP</button><span id="save-status" class="save-status"></span>{script}'''


def load_translation_maps(directory: Path) -> dict[str, dict[str, dict[str, str]]]:
    maps = {}
    for task in ("reward", "roi", "generation"):
        rows = load_jsonl(directory / f"{task}.jsonl")
        maps[task] = {row["case_id"]: row["translations"] for row in rows}
    if {task: len(rows) for task, rows in maps.items()} != {"reward": 60, "roi": 20, "generation": 100}:
        raise ValueError("incomplete bilingual translation set")
    return maps


def bilingual_region_table(regions: list[dict[str, Any]], texts: dict[str, str], translations: dict[str, str], prefix: str) -> str:
    rows = []
    for number, region in enumerate(regions, 1):
        key = f"{prefix}_region_{number:02d}_feature"
        rows.append(
            f"<tr><td>{number}</td><td>{html.escape(str(region.get('box')))}</td>"
            f"<td><strong>中文：</strong>{html.escape(translations[key])}<br><strong>English：</strong>{html.escape(texts[key])}</td>"
            f"<td>{html.escape(ROLE_LABELS.get(str(region.get('role')), str(region.get('role'))))}</td>"
            f"<td>{html.escape(str(region.get('importance', '')))}</td></tr>"
        )
    return '<table><tr><th>编号 / No.</th><th>坐标 / Box (0–1000)</th><th>特征 / Feature</th><th>作用 / Role</th><th>重要度 / Importance</th></tr>' + "".join(rows) + "</table>"


def bilingual_score_table(scores: dict[str, Any], reason_en: str, reason_zh: str) -> str:
    rows = []
    for field, (zh, en) in JUDGE_FIELDS.items():
        rows.append(f"<tr><th>{html.escape(zh)} / {html.escape(en)}</th><td>{float(scores[field]):.2f}</td></tr>")
    return "<table>" + "".join(rows) + "</table>" + bilingual_text("总体理由", "Overall reason", reason_en, reason_zh)


def build_bilingual_views(root: Path, translations_dir: Path) -> None:
    translations = load_translation_maps(translations_dir)
    sources = source_records()
    sources_by_task = {task: {row["case_id"]: row for row in rows} for task, rows in sources.items()}
    first = root / "first_pass_blinded"
    views = root / "bilingual_views"
    for kind in ("cases", "references"):
        for task in ("reward", "roi", "generation"):
            (views / kind / task).mkdir(parents=True, exist_ok=True)

    reward_refs = {row["case_id"]: row for row in load_jsonl(Path("/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/human_review/external_reference_reward_review_20260815/references.jsonl"))}
    for case_id, source in sources_by_task["reward"].items():
        texts, zh = source["texts"], translations["reward"][case_id]
        case_json = json.loads((first / "reward_agreement_blinded/cases" / case_id / "case.json").read_text(encoding="utf-8"))
        image_url = f"../../../first_pass_blinded/reward_agreement_blinded/cases/{case_id}/{case_json['image_file']}"
        body = f'<img src="{image_url}" alt="病理图像 / pathology image">' + bilingual_text("题干与选项", "Question and options", texts["question_and_options"], zh["question_and_options"]) + bilingual_text("数据集参考答案", "Dataset reference answer", texts["dataset_reference_answer"], zh["dataset_reference_answer"]) + bilingual_text("待评分回答", "Candidate response to review", texts["candidate_completion"], zh["candidate_completion"])
        (views / "cases/reward" / f"{case_id}.html").write_text(page(f"{case_id} 病例 / Case", body), encoding="utf-8")
        reference = reward_refs[case_id]["reference"]
        rows = []
        for field in CORE_REWARD_FIELDS:
            value = reference[field]
            zh_label, en_label, explanation = REWARD_FIELDS[field]
            display = boolean_label(value) if isinstance(value, bool) or value == "uncertain" else str(value)
            rows.append(f"<tr><th>{html.escape(zh_label)} / {html.escape(en_label)}</th><td>{html.escape(display)}</td><td>{html.escape(explanation)}</td></tr>")
        ref_body = '<div class="warning">Claude Sonnet 4.6 外部参考 / External reference；不是病理专家真值 / not pathologist ground truth. 仅六事件进入奖励；总体理由只帮助理解，不是评分项 / Only the six events enter the reward; the overall reason is explanatory only.</div><table><tr><th>指标 / Field</th><th>参考值 / Value</th><th>含义 / Meaning</th></tr>' + "".join(rows) + "</table>" + bilingual_text("补充参考理由（不评分）", "Explanatory rationale (not rated)", texts["external_overall_reason"], zh["external_overall_reason"])
        (views / "references/reward" / f"{case_id}.html").write_text(page(f"{case_id} 外部参考 / External reference", ref_body), encoding="utf-8")

    effective = json.loads(Path("/home/dataset-assist-0/czy/wjy/myr1/protocol/effective_interpretability_panel_20case_20260814.json").read_text(encoding="utf-8"))
    effective_by_panel = {int(row["panel_index"]): row for row in effective["cases"]}
    mapping = json.loads(Path("/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/human_review/expert_review_browse_packets_20260815/INTERNAL_DO_NOT_SEND_TO_REVIEWERS/source_mapping.json").read_text(encoding="utf-8"))["roi_cases"]
    gemini = {int(row["panel_index"]): row["annotation"] for row in load_jsonl(Path("/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/exhaustive_reference_annotation_gemini31pro_formal_20260814/all160_validated.jsonl"))}
    opus = {int(row["panel_index"]): row["annotation"] for row in load_jsonl(Path("/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/dual_model_roi_formal_20260814/opus_annotations/all120_validated.jsonl"))}
    for item in mapping:
        case_id, panel = item["case_id"], int(item["source_panel_index"])
        source, case = sources_by_task["roi"][case_id], effective_by_panel[panel]
        texts, zh = source["texts"], translations["roi"][case_id]
        base_url = f"../../../first_pass_blinded/interpretability_roi_blinded/cases/{case_id}"
        case_body = f'<h2>原图 / Original image</h2><img src="{base_url}/image_original.png"><h2>候选区域 / Candidate regions</h2><img src="{base_url}/image_candidate_regions.jpg">' + bilingual_text("题干与选项", "Question and options", texts["question_and_options"], zh["question_and_options"]) + bilingual_text("数据集参考答案", "Dataset reference answer", texts["dataset_reference_answer"], zh["dataset_reference_answer"])
        (views / "cases/roi" / f"{case_id}.html").write_text(page(f"{case_id} 区域病例 / ROI case", case_body), encoding="utf-8")
        g, o = gemini[panel], opus[panel]
        ext = case["external_annotation"]
        ref_body = '<div class="warning">外部 VLM pseudo-reference；需由病理专家独立确认 / External VLM pseudo-reference requiring pathologist verification.</div><h2>双模型共识区域 / Dual-model consensus regions</h2>' + bilingual_region_table(ext["regions"], texts, zh, "consensus") + bilingual_text("Gemini 理由", "Gemini rationale", texts["gemini_rationale"], zh["gemini_rationale"]) + bilingual_region_table(g.get("regions", []), texts, zh, "gemini") + bilingual_text("Claude Opus 理由", "Claude Opus rationale", texts["opus_rationale"], zh["opus_rationale"]) + bilingual_region_table(o.get("regions", []), texts, zh, "opus")
        (views / "references/roi" / f"{case_id}.html").write_text(page(f"{case_id} 区域参考 / ROI reference", ref_body), encoding="utf-8")

    answer_key_path = Path("/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/stage2_stage3_human_review_packet_20260815/blind_pairwise_multijudge_panel100/adjudication_key_keep_separate/answer_key.csv")
    with answer_key_path.open(encoding="utf-8-sig") as handle:
        answer_keys = {row["case_id"]: row for row in csv.DictReader(handle)}
    claude = judge_map(Path("/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/multijudge_reasoning_eval_20260810/formal_claude/judgments.jsonl"))
    gemini_judge = judge_map(Path("/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/multijudge_reasoning_eval_20260810/formal_gemini/judgments.jsonl"))
    for case_id, source in sources_by_task["generation"].items():
        texts, zh = source["texts"], translations["generation"][case_id]
        case_json = json.loads((first / "stage2_stage3_blind_pairwise_100" / case_id / "case.json").read_text(encoding="utf-8"))
        image_url = f"../../../first_pass_blinded/stage2_stage3_blind_pairwise_100/{case_id}/{case_json['image_file']}"
        case_body = f'<img src="{image_url}" alt="病理图像 / pathology image">' + bilingual_text("题干与选项", "Question and options", texts["question_and_options"], zh["question_and_options"]) + bilingual_text("数据集参考答案", "Dataset reference answer", texts["dataset_reference_answer"], zh["dataset_reference_answer"]) + '<div class="panes"><section>' + bilingual_text("匿名回答 A", "Anonymous response A", texts["response_a"], zh["response_a"]) + '</section><section>' + bilingual_text("匿名回答 B", "Anonymous response B", texts["response_b"], zh["response_b"]) + "</section></div>"
        (views / "cases/generation" / f"{case_id}.html").write_text(page(f"{case_id} 生成质量病例 / Generation-quality case", case_body), encoding="utf-8")
        key = answer_keys[case_id]
        source_index = int(key["source_index"])
        cards = []
        for anonymous in ("a", "b"):
            model = key[f"response_{anonymous}_model"]
            c_scores, g_scores = claude[(source_index, model)], gemini_judge[(source_index, model)]
            cards.append(f'<h2>匿名回答 {anonymous.upper()} / Anonymous response {anonymous.upper()}</h2><h3>Claude Sonnet 4.6 参考 / Reference</h3>' + bilingual_score_table(c_scores, texts[f"claude_response_{anonymous}_reason"], zh[f"claude_response_{anonymous}_reason"]) + '<h3>Gemini 3.1 Pro 参考 / Reference</h3>' + bilingual_score_table(g_scores, texts[f"gemini_response_{anonymous}_reason"], zh[f"gemini_response_{anonymous}_reason"]))
        (views / "references/generation" / f"{case_id}.html").write_text(page(f"{case_id} 生成质量参考 / Generation reference", '<div class="warning">外部 Judge 仅供辅助，不显示回答模型身份 / Advisory only; response-model identity remains hidden.</div>' + "".join(cards)), encoding="utf-8")


def build_assisted_review(root: Path, translations_dir: Path | None = None) -> None:
    if translations_dir is not None:
        build_bilingual_views(root, translations_dir)
    reward_references = {
        row["case_id"]: row["reference"]
        for row in load_jsonl(Path("/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/human_review/external_reference_reward_review_20260815/references.jsonl"))
    }
    assisted = root / "assisted_review_with_external_references"
    assisted.mkdir(exist_ok=True)
    specs = {
        "reward": {
            "title": "奖励机制辅助复核",
            "ids": [f"HRA-{number:03d}" for number in range(1, 61)],
            "blind": lambda case_id: f"../../bilingual_views/cases/reward/{case_id}.html" if translations_dir else f"../../first_pass_blinded/reward_agreement_blinded/cases/{case_id}/index.html",
            "reference": lambda case_id: f"../../bilingual_views/references/reward/{case_id}.html" if translations_dir else f"../../second_pass_external_references/reward_reference_after_first_pass/{case_id}.html",
            "csv": "../../first_pass_blinded/reward_agreement_blinded/ratings_template.csv",
        },
        "roi": {
            "title": "可解释性区域辅助复核",
            "ids": [f"EIR-{number:03d}" for number in range(1, 21)],
            "blind": lambda case_id: f"../../bilingual_views/cases/roi/{case_id}.html" if translations_dir else f"../../first_pass_blinded/interpretability_roi_blinded/cases/{case_id}/index.html",
            "reference": lambda case_id: f"../../bilingual_views/references/roi/{case_id}.html" if translations_dir else f"../../second_pass_external_references/roi_reference_after_first_pass/{case_id}.html",
            "csv": "../../first_pass_blinded/interpretability_roi_blinded/case_ratings_template.csv",
        },
        "generation": {
            "title": "Stage2/Stage3 生成质量辅助盲评",
            "ids": [f"case_{number:03d}" for number in range(1, 101)],
            "blind": lambda case_id: f"../../bilingual_views/cases/generation/{case_id}.html" if translations_dir else f"../../first_pass_blinded/stage2_stage3_blind_pairwise_100/{case_id}/review.html",
            "reference": lambda case_id: f"../../bilingual_views/references/generation/{case_id}.html" if translations_dir else f"../../second_pass_external_references/generation_reference_after_first_pass/{case_id}.html",
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
            direct_form = reward_click_form(case_id, reward_references[case_id]) if task == "reward" else ""
            csv_link = "" if task == "reward" else f'<p><a href="{spec["csv"]}">下载该任务评分 CSV 模板</a></p>'
            if task == "reward":
                body = f'''{nav}<div class="warning"><strong>外部参考辅助模式：</strong>候选模型/训练阶段仍盲化，但专家从一开始就能看到外部模型意见。因此本入口结果必须标为 reference-assisted，不能用于“独立专家评分”主统计。</div>
<section><h2>先阅读病例、图像与待评分回答 / First review the case, image, and candidate response</h2><iframe style="width:100%;height:1050px;border:1px solid #999" src="{spec['blind'](case_id)}"></iframe></section>
{assisted_rubric(task)}{direct_form}{nav}'''
            else:
                body = f'''{nav}<div class="warning"><strong>外部参考辅助模式：</strong>候选模型/训练阶段仍盲化，但专家从一开始就能看到外部模型意见。因此本入口结果必须标为 reference-assisted，不能用于“独立专家评分”主统计。</div>
{assisted_rubric(task)}{csv_link}
<div class="panes"><section><h2>病例与待评分内容 / Case and response</h2><iframe src="{spec['blind'](case_id)}"></iframe></section><section><h2>外部模型参考意见 / External-model reference</h2><iframe src="{spec['reference'](case_id)}"></iframe></section></div>{nav}'''
            (task_root / f"{case_id}.html").write_text(page(f"{spec['title']}：{case_id}", body), encoding="utf-8")
            links.append(f'<a href="{case_id}.html">{case_id}</a>')
        csv_link = "" if task == "reward" else f'<p><a href="{spec["csv"]}">下载评分 CSV 模板</a></p>'
        index_body = f'''<p><a href="../index.html">返回辅助评审总目录</a></p><div class="warning">本任务含外部参考，结果属于 reference-assisted expert review。奖励任务的六事件可在每题页面直接点击并保存到本地文件。</div>{assisted_rubric(task)}{csv_link}<div class="grid">{"".join(links)}</div>'''
        (task_root / "index.html").write_text(page(spec["title"], index_body), encoding="utf-8")
        root_links.append(f'<li><a href="{task}/index.html">{spec["title"]}</a></li>')
    assisted_body = '<div class="warning">此入口直接显示外部参考，适合辅助作答或仲裁，不适合作为独立盲评主统计。</div><ol>' + "".join(root_links) + "</ol>"
    (assisted / "index.html").write_text(page("外部参考辅助专家评审", assisted_body), encoding="utf-8")


def write_root_index(root: Path) -> None:
    body = '''<div class="warning"><strong>请选择评审模式：</strong>独立盲评用于论文主要人工统计；辅助模式从第一题开始显示外部意见，必须单独标注为 reference-assisted。</div>
<ol><li><a href="first_pass.html">独立盲评入口（不显示外部参考）</a></li><li><a href="assisted_review_with_external_references/index.html">外部参考辅助评审入口（含上一题/下一题）</a></li><li><a href="second_pass_external_references/reward_reference_after_first_pass/index.html">原两阶段：奖励参考</a></li><li><a href="second_pass_external_references/roi_reference_after_first_pass/index.html">原两阶段：ROI 参考</a></li><li><a href="second_pass_external_references/generation_reference_after_first_pass/index.html">原两阶段：生成质量参考</a></li></ol>'''
    (root / "index.html").write_text(page("PathVLM-R1 专家复核入口", body), encoding="utf-8")


def write_portable_server(root: Path) -> None:
    source = Path(__file__).with_name("serve_expert_review_app.py")
    if not source.is_file():
        raise FileNotFoundError(source)
    bundled = root / "serve_expert_review_app.py"
    bundled.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    bundled.chmod(0o755)
    launcher = root / "serve_review.sh"
    launcher.write_text('''#!/usr/bin/env bash
set -euo pipefail
PORT="${1:-8765}"
ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
echo "Portal: http://127.0.0.1:${PORT}/"
echo "Ratings: ${ROOT_DIR}/expert_submissions"
exec python3 "${ROOT_DIR}/serve_expert_review_app.py" \\
  --root "${ROOT_DIR}" --output "${ROOT_DIR}/expert_submissions" \\
  --host 127.0.0.1 --port "${PORT}"
''', encoding="utf-8")
    launcher.chmod(0o755)
    quick_guide = Path(__file__).resolve().parents[1] / "docs/20260815_EXPERT_REVIEWER_QUICK_START.md"
    if not quick_guide.is_file():
        raise FileNotFoundError(quick_guide)
    (root / "EXPERT_REVIEWER_QUICK_START.md").write_text(
        quick_guide.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (root / "PACKAGE_README.md").write_text('''# PathVLM-R1 分布式专家复核包

专家收到本包后，请先阅读 `EXPERT_REVIEWER_QUICK_START.md`。

每位专家在自己的电脑上解压一份本包，运行 `bash serve_review.sh 8765`，再访问
`http://127.0.0.1:8765/`。负责人应预先分配匿名 Reviewer ID；不要填写姓名或邮箱。

奖励复核的历史记录只从本包内 `expert_submissions/<Reviewer ID>/` 读取，不访问 HF、
负责人主机或其他远程服务。完成后点击“导出我的评分包”，将生成的
`pathvlm_reward_review_<Reviewer ID>_complete60.zip` 交还负责人。`partialN` 只用于断点备份。

负责人收到 ZIP 后，先用仓库中的 `scripts/import_expert_reward_exports.py` 校验 SHA-256、
Reviewer ID、60 例完整性和六布尔字段，再用 `scripts/analyze_human_reward_score_differences.py`
比较人工过程分与训练期 GPT-4o 奖励；Claude 只作为第二参考。当前直接显示外部参考的入口
属于 reference-assisted review，不能宣称为独立盲评。

ROI 结果用于确认 pseudo-reference evidence region 是否具有病理相关性，并在确认后比较
reference deletion 与 area-matched random deletion 的决策损失。Stage2/Stage3 生成质量结果
应在 A/B 解盲后比较人工偏好，Claude/Gemini 只作外部一致性参照。
''', encoding="utf-8")


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
    parser.add_argument("--translations-dir", type=Path)
    parser.add_argument("--augment-existing", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        if not args.augment_existing:
            raise FileExistsError(args.output)
        build_assisted_review(args.output, args.translations_dir)
        write_root_index(args.output)
        write_portable_server(args.output)
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
    build_assisted_review(args.output, args.translations_dir)
    write_root_index(args.output)
    write_portable_server(args.output)
    print(json.dumps({"status": "completed", "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
