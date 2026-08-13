#!/usr/bin/env python3
"""Build auditable Markdown/JSON tables from explicit final-result paths.

The registry deliberately separates the current fixed-A/B generation contract from
legacy baseline contracts. Missing/running cells are rendered as NA instead of being
silently borrowed from an incompatible evaluation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORK = Path("/home/dataset-assist-0/czy/wjy")
EVAL = WORK / "pathvlm_revision_eval_a100/runs"


def load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric(path: Path, keys: tuple[str, ...]) -> tuple[float | None, dict[str, Any] | None]:
    data = load(path)
    if data is None:
        return None, None
    for key in keys:
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value), data
    return None, data


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.2f}"


def model_row(name: str, paths: dict[str, Path]) -> dict[str, Any]:
    val, _ = metric(paths["pathmmu_val"], ("accuracy",))
    test, _ = metric(paths["pathmmu_test"], ("accuracy",))
    pv, pv_data = metric(paths["pathvqa"], ("contract_aligned_exact_accuracy", "accuracy"))
    omni, omni_data = metric(paths["omni"], ("contract_aligned_accuracy",))
    mmmu, _ = metric(paths["mmmu"], ("accuracy",))
    return {
        "model": name,
        "pathmmu_val": val,
        "pathmmu_test": test,
        "pathvqa_fixed_ab": pv,
        "pathvqa_parse_rate": None if pv_data is None else pv_data.get("ab_parseable_rate", pv_data.get("parseable_rate")),
        "pathvqa_cap_rate": None if pv_data is None else pv_data.get("generation_cap_hit_rate", 0.0 if pv_data.get("generation_cap_hit_count") == 0 else None),
        "omnimed_contract_aligned": omni,
        "omnimed_official": None if omni_data is None else omni_data.get("official_accuracy"),
        "mmmu_nonmedical": mmmu,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def render_table(rows: list[dict[str, Any]]) -> list[str]:
    out = [
        "| Model | PathMMU Val | PathMMU Test999 | PathVQA Test A/B | Parse | Cap hit | Omni aligned | Omni official | MMMU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        out.append(
            "| {model} | {val} | {test} | {pv} | {parse} | {cap} | {omni} | {official} | {mmmu} |".format(
                model=row["model"],
                val=pct(row["pathmmu_val"]),
                test=pct(row["pathmmu_test"]),
                pv=pct(row["pathvqa_fixed_ab"]),
                parse=pct(row["pathvqa_parse_rate"]),
                cap=pct(row["pathvqa_cap_rate"]),
                omni=pct(row["omnimed_contract_aligned"]),
                official=pct(row["omnimed_official"]),
                mmmu=pct(row["mmmu_nonmedical"]),
            )
        )
    return out


def registry() -> list[tuple[str, dict[str, Path]]]:
    empty = Path("/nonexistent")
    selected = EVAL / "formal_selected_sft3000_20260811"
    current = EVAL / "full_language_rule_rl_clean_n4_n8_step1000_20260812"
    historical = EVAL / "core_historical_corrected_eval_20260813"
    stage3_compare = EVAL / "stage3_gpt4o_n8_checkpoint_ood_comparison_20260813"
    stage3_final = EVAL / "stage3_gpt4o_n8_checkpoint1500_final_eval_20260813"
    legacy_test = EVAL / "pathmmu_sft4000_stage2_20260729_221640"
    lora4000 = EVAL / "lora_sft4000_control_20260814/formal_8gpu_gbs96"
    base_mmmu = EVAL / "visual_adaptation_architecture_validation_20260811/mmmu_nonmedical_dev116/base_max4096/metrics.json"
    rows = [
        ("Base Qwen2.5-VL-7B", {
            "pathmmu_val": historical / "base/pathmmu_val385/metrics.json",
            "pathmmu_test": selected / "pathmmu_test999/base/metrics.json",
            "pathvqa": historical / "base/pathvqa_test3362_ab/metrics.json",
            "omni": historical / "base/omnimedvqa_8518/metrics.json",
            "mmmu": base_mmmu,
        }),
        ("Historical full-SFT3000", {
            "pathmmu_val": historical / "sft3000/pathmmu_val385/metrics.json",
            "pathmmu_test": legacy_test / "sft3000_parent_test999/metrics.json",
            "pathvqa": historical / "sft3000/pathvqa_test3362_ab/metrics.json",
            "omni": historical / "sft3000/omnimedvqa_8518/metrics.json",
            "mmmu": historical / "sft3000/mmmu_nonmedical116/metrics.json",
        }),
        ("Historical Stage2 outcome-RL", {
            "pathmmu_val": historical / "stage2/pathmmu_val385/metrics.json",
            "pathmmu_test": legacy_test / "stage2_rl_test999/metrics.json",
            "pathvqa": historical / "stage2/pathvqa_test3362_ab/metrics.json",
            "omni": historical / "stage2/omnimedvqa_8518/metrics.json",
            "mmmu": historical / "stage2/mmmu_nonmedical116/metrics.json",
        }),
        ("Current L-r16 SFT3000 step80", {
            "pathmmu_val": selected / "pathmmu/step080/metrics.json",
            "pathmmu_test": selected / "pathmmu_test999/step080/metrics.json",
            "pathvqa": EVAL / "current_lineage_pathvqa_test3362_ab_20260813/l_r16_sft80/metrics.json",
            "omni": current / "candidate_ood_eval/omnimedvqa/sft_parent/metrics.json",
            "mmmu": current / "candidate_ood_eval/mmmu/sft_parent/metrics.json",
        }),
        ("Full rule-RL n4 step1000", {
            "pathmmu_val": current / "n4_fresh_step1000_pathmmu_eval/val/metrics.json",
            "pathmmu_test": current / "n4_fresh_step1000_pathmmu_eval/test999/metrics.json",
            "pathvqa": EVAL / "current_lineage_pathvqa_test3362_ab_20260813/full_rule_rl_n4_step1000/metrics.json",
            "omni": current / "candidate_ood_eval/omnimedvqa/full_rl_n4_step1000/metrics.json",
            "mmmu": current / "candidate_ood_eval/mmmu/full_rl_n4_step1000/metrics.json",
        }),
        ("Full rule-RL n8 step1000", {
            "pathmmu_val": current / "n8_fresh_step1000_pathmmu_eval/val/metrics.json",
            "pathmmu_test": current / "n8_fresh_step1000_pathmmu_eval/test999/metrics.json",
            "pathvqa": EVAL / "current_lineage_pathvqa_test3362_ab_20260813/full_rule_rl_n8_step1000/metrics.json",
            "omni": current / "candidate_ood_eval/omnimedvqa/full_rl_n8_step1000/metrics.json",
            "mmmu": current / "candidate_ood_eval/mmmu/full_rl_n8_step1000/metrics.json",
        }),
        ("GPT-4o Stage3 step500", {
            "pathmmu_val": stage3_compare / "checkpoint500/pathmmu_val385/metrics.json",
            "pathmmu_test": EVAL / "stage3_gpt4o_n8_checkpoint500_test999_20260813_v2/metrics.json",
            "pathvqa": stage3_compare / "checkpoint500/pathvqa_test3362_ab/metrics.json",
            "omni": stage3_compare / "checkpoint500/omnimedvqa_8518/metrics.json",
            "mmmu": stage3_compare / "checkpoint500/mmmu_nonmedical116/metrics.json",
        }),
        ("GPT-4o Stage3 step1000", {
            "pathmmu_val": stage3_compare / "checkpoint1000/pathmmu_val385/metrics.json",
            "pathmmu_test": EVAL / "stage3_gpt4o_n8_checkpoint1000_test999_20260813_v2/metrics.json",
            "pathvqa": stage3_compare / "checkpoint1000/pathvqa_test3362_ab/metrics.json",
            "omni": stage3_compare / "checkpoint1000/omnimedvqa_8518/metrics.json",
            "mmmu": stage3_compare / "checkpoint1000/mmmu_nonmedical116/metrics.json",
        }),
        ("GPT-4o Stage3 step1500", {
            "pathmmu_val": stage3_final / "pathmmu_val385/metrics.json",
            "pathmmu_test": stage3_final / "pathmmu_test999/metrics.json",
            "pathvqa": stage3_final / "pathvqa_test3362_ab/metrics.json",
            "omni": stage3_final / "omnimedvqa_8518/metrics.json",
            "mmmu": stage3_final / "mmmu_nonmedical116/metrics.json",
        }),
        ("LoRA-SFT4000 control", {
            "pathmmu_val": lora4000 / "pathmmu_val385/metrics.json",
            "pathmmu_test": lora4000 / "pathmmu_test999/metrics.json",
            "pathvqa": lora4000 / "pathvqa_test3362_ab/metrics.json",
            "omni": lora4000 / "omnimedvqa_8518/metrics.json",
            "mmmu": lora4000 / "mmmu_nonmedical116/metrics.json",
        }),
    ]
    return rows


def ratio_rows() -> list[dict[str, Any]]:
    root = EVAL / "data_ratio_uniform_eval_20260814"
    specs = [(250, 750, "sft0250_rl0750"), (500, 500, "sft0500_rl0500"), (750, 250, "sft0750_rl0250")]
    rows = []
    for sft, rl, label in specs:
        base = root / label
        paths = {
            "pathmmu_val": base / "pathmmu_val385/metrics.json",
            "pathmmu_test": base / "pathmmu_test999/metrics.json",
            "pathvqa": base / "pathvqa_test3362_ab/metrics.json",
            "omni": base / "omnimedvqa_8518/metrics.json",
            "mmmu": base / "mmmu_nonmedical116/metrics.json",
        }
        row = model_row(f"{sft} SFT + {rl} rule-RL", paths)
        row.update({"sft_samples": sft, "rl_samples": rl})
        rows.append(row)
    return rows


def repeat_rows() -> list[dict[str, Any]]:
    root = EVAL / "repeated_inference_pathmmu_test999_20260814"
    labels = ["l_r16_sft80", "full_rule_rl_n8_step1000", "stage3_gpt4o_step1500"]
    rows = []
    for label in labels:
        path = root / label / "summary.json"
        rows.append({"model": label, "summary": load(path), "path": str(path)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    core = [model_row(name, paths) for name, paths in registry()]
    ratios = ratio_rows()
    repeats = repeat_rows()
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "core_current_contract": core,
        "ratio_ablation": ratios,
        "repeated_inference": repeats,
        "human_expert_scoring": "excluded_pending_user",
        "contract_note": "Current PathVQA table uses fixed A=Yes/B=No generation. Legacy external baselines remain separately stratified.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 最终统一结果表（自动生成）",
        "",
        "> 百分数均由明确的 `metrics.json` 路径生成。`NA` 表示该合同下尚未完成，绝不从不兼容的旧评测借值。PathMMU Test999 是开发诊断测试，不能表述为未触碰最终测试。",
        "",
        "## 当前核心主线（固定 A=Yes/B=No PathVQA 合同）",
        "",
        *render_table(core),
        "",
        "## 固定总池 1000 的 SFT/RL 配比筛查",
        "",
        *render_table(ratios),
        "",
        "## Test999 五次随机推理重复",
        "",
        "合同：seeds 42–46，temperature=0.7，top-p=0.9，top-k disabled，max_new_tokens=1024。主结果仍以 greedy 为准；这里的 t 区间衡量解码随机性，不替代 case-bootstrap。",
        "",
        "| Model | Runs | Mean | SD | t 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in repeats:
        s = row["summary"] or {}
        count = s.get("run_count", s.get("completed_runs"))
        mean = s.get("mean_accuracy")
        sd = s.get("sample_sd", s.get("sd_accuracy"))
        ci = s.get("student_t_95_ci", s.get("ci95"))
        ci_text = "NA" if not ci else f"[{100*ci[0]:.2f}, {100*ci[1]:.2f}]"
        lines.append(f"| {row['model']} | {count if count is not None else 'NA'} | {pct(mean)} | {pct(sd)} | {ci_text} |")
    lines.extend([
        "",
        "## 外部基线（历史统一 short-answer / Yes-No 合同）",
        "",
        "> 以下结果已经全量跑完，保留原合同用于可复现对比；不能解释为已使用当前 A/B prompt 重跑。",
        "",
        "| Model | PathMMU Test999 | PathVQA yes/no3362 | OmniMedVQA aligned |",
        "|---|---:|---:|---:|",
        "| Qwen2.5-VL-3B | 44.94 | 58.06 | 60.72 |",
        "| Lingshu-7B | 58.36 | 85.22 | 80.37 |",
        "| InternVL3-8B | 55.36 | 65.50 | 72.58 |",
        "| HuatuoGPT-Vision-7B | 53.15 | 64.78 | 71.32 |",
        "| MedGemma-4B-IT | 40.04 | 62.17 | 71.62 |",
        "| MedVLM-R1 | 41.34 | 56.28 | 50.89 |",
        "| Llama-3.2-Vision-90B | 57.16 | 62.49 | 68.09 |",
        "| Llama-3.2-Vision-11B | 22.22 | 55.98 | 53.90 |",
        "| DeepSeek-VL2 | 42.94 | 59.52 | 51.56 |",
        "| ScaleReasoner-R1 | 64.86 | 57.53 | 43.30 |",
        "| Qwen-VL-Plus | 54.75 | 67.58 | 72.22 |",
        "| Claude Haiku 4.5 | 46.65 | 40.30 | 56.22 |",
        "",
        "完整分子/分母、official sensitivity 与 cap caveat：`docs/20260810_final_closed_benchmark_and_multijudge_results.md`。",
        "",
        "## 已完成的异构 Judge 稳健性",
        "",
        "GPT-4o、Claude Sonnet 4.6、Gemini 3.1 Pro 已对冻结的 100-case、6-model panel 完成 2,880 个盲评判断。两条 Stage3 臂相对 Stage2 的宏平均点估计在三个 Judge 下均为正，但六个 95% CI 均跨 0；因此只能写成方向一致的小幅提升，不能宣称统计显著。人工专家评分仍由用户后补。",
        "",
        "## 合同边界",
        "",
        "- 外部 PathVQA 基线已经有全量结果，但大多使用旧 short-answer/自由 Yes-No 合同；详见 `docs/20260810_final_closed_benchmark_and_multijudge_results.md`，不与上表伪装成同 prompt 比较。",
        "- 人工病理专家 ROI/生成质量评分由用户后续完成，本表显式排除该唯一人工缺口。",
        "- 不扩展全模型配对反转分析；仅保留最终核心对照所需的最小 paired bootstrap/McNemar。",
        "",
        f"机器可读数据：`{args.output_json}`",
    ])
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
