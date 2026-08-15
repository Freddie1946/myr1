#!/usr/bin/env python3
"""Build an auditable result-lineage catalog and revision-table data pack."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORK = Path("/home/dataset-assist-0/czy/wjy")
REPO = WORK / "myr1"
EVAL = WORK / "pathvlm_revision_eval_a100/runs"
TRAIN = WORK / "pathvlm_r1_v1_a100/runs"
FINAL_JSON = REPO / "protocol/final_unified_results_tables_20260814.json"


def load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def relative(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(WORK))
    except ValueError:
        return str(path.resolve())


def text_hash(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def first(*values: Any) -> Any:
    return next((value for value in values if value not in (None, "", [])), None)


def lineage_entry(directory: Path) -> dict[str, Any]:
    metrics_path = directory / "metrics.json"
    config_path = directory / "run_config.json"
    predictions_path = directory / "predictions.jsonl"
    metrics = load(metrics_path) or {}
    config = load(config_path) or {}
    merged = {**metrics, **config}
    model = first(
        merged.get("model_path"), merged.get("model"), merged.get("model_label"),
        merged.get("adapter_path"), merged.get("adapter"),
    )
    model_hash = first(
        merged.get("adapter_model_sha256"), merged.get("model_config_sha256"),
    )
    data = first(
        merged.get("data_path"), merged.get("data"), merged.get("dataset"),
        merged.get("panel_path"),
    )
    data_hash = first(
        merged.get("data_sha256"), merged.get("dataset_sha256"),
        merged.get("panel_sha256"),
    )
    contract = first(
        merged.get("generation_contract"), merged.get("prompt_contract"),
        merged.get("matching_contract"),
    )
    task = merged.get("task")
    if not task and "omnimedvqa" in str(directory).lower():
        task = "omnimedvqa"
    elif not task and "pathvqa" in str(directory).lower():
        task = "pathvqa"
    prompt_hash = text_hash(merged.get("prompt_template"))
    has_prompt = bool(contract or prompt_hash)
    has_decoder = merged.get("max_new_tokens") is not None and merged.get("do_sample") is not None
    has_prediction_integrity = bool(
        merged.get("predictions_sha256") or
        (predictions_path.is_file() and predictions_path.stat().st_size > 0)
    )
    checks = {
        "model_identity": bool(model),
        "model_hash": bool(model_hash),
        "data_identity": bool(data),
        "data_hash": bool(data_hash),
        "prompt_or_generation_contract": has_prompt,
        "decoder_contract": has_decoder,
        "per_case_predictions": predictions_path.is_file(),
        "prediction_integrity": has_prediction_integrity,
        "aggregate_metrics": metrics_path.is_file(),
    }
    if all(checks[key] for key in (
        "model_identity", "model_hash", "data_identity", "data_hash",
        "prompt_or_generation_contract", "decoder_contract",
        "per_case_predictions", "aggregate_metrics",
    )):
        level = "complete"
    elif checks["model_identity"] and checks["data_identity"] and has_prompt:
        level = "strong_partial"
    elif checks["model_identity"] and checks["data_identity"]:
        level = "legacy_partial"
    else:
        level = "ambiguous_nonstandard"
    metric_keys = (
        "count", "accuracy", "correct", "contract_aligned_accuracy",
        "contract_aligned_correct", "official_accuracy", "official_correct",
        "paper_yes_no_contract_aligned_accuracy", "mean_generated_tokens",
        "generation_cap_hit_count", "empty_completion_count", "primary_metric",
    )
    return {
        "artifact_directory": str(directory.resolve()),
        "workspace_relative_directory": relative(directory),
        "traceability_level": level,
        "traceability_checks": checks,
        "model": model,
        "model_config_sha256": merged.get("model_config_sha256"),
        "adapter": first(merged.get("adapter_path"), merged.get("adapter")),
        "adapter_config_sha256": merged.get("adapter_config_sha256"),
        "adapter_model_sha256": merged.get("adapter_model_sha256"),
        "task": task,
        "split_role": merged.get("split_role"),
        "backend": merged.get("backend"),
        "data": data,
        "data_sha256": data_hash,
        "generation_contract": contract,
        "prompt_template_sha256": prompt_hash,
        "do_sample": merged.get("do_sample"),
        "seed": merged.get("seed"),
        "temperature": merged.get("temperature"),
        "top_p": merged.get("top_p"),
        "top_k": merged.get("top_k"),
        "max_new_tokens": merged.get("max_new_tokens"),
        "metrics": {key: metrics[key] for key in metric_keys if key in metrics},
        "files": {
            "run_config": relative(config_path) if config_path.is_file() else None,
            "metrics": relative(metrics_path) if metrics_path.is_file() else None,
            "predictions": relative(predictions_path) if predictions_path.is_file() else None,
            "predictions_bytes": predictions_path.stat().st_size if predictions_path.is_file() else None,
        },
        "paper_reporting_status": (
            "eligible_corrected_omni"
            if task == "omnimedvqa" and contract == "omnimed_domain_think_answer_v4_1024"
            else "excluded_legacy_omni"
            if task == "omnimedvqa"
            else "requires_table_registry_review"
        ),
    }


def build_lineage() -> list[dict[str, Any]]:
    directories: set[Path] = set()
    for root in (EVAL, TRAIN):
        for name in ("metrics.json", "run_config.json", "predictions.jsonl"):
            directories.update(path.parent for path in root.rglob(name))
    return [lineage_entry(path) for path in sorted(directories)]


BASELINE_PATHMMU = (
    ("Qwen2.5-VL-3B", EVAL / "pathmmu_diagnostic/test999_development_20260728_215918/qwen3/metrics.json"),
    ("Qwen2.5-VL-7B", EVAL / "pathmmu_diagnostic/test999_development_20260728_215918/qwen7/metrics.json"),
    ("Lingshu-7B", EVAL / "pathmmu_diagnostic/test999_development_20260728_215918/lingshu/metrics.json"),
    ("InternVL3-8B", EVAL / "pathmmu_diagnostic/test999_development_20260728_220821/internvl/metrics.json"),
    ("HuatuoGPT-Vision-7B", EVAL / "pathmmu_diagnostic/test999_development_20260728_221830/huatuo/metrics.json"),
    ("MedGemma-4B-IT", EVAL / "pathmmu_diagnostic/test999_development_20260728_220617/medgemma/metrics.json"),
    ("MedVLM-R1", EVAL / "pathmmu_diagnostic/test999_development_20260728_215918/medvlm/metrics.json"),
    ("Llama-3.2-Vision-11B", EVAL / "local_gpu_baselines_20260803/llama32_11b/pathmmu_test999/metrics.json"),
    ("Llama-3.2-Vision-90B", EVAL / "local_gpu_baselines_20260803/llama32_90b/pathmmu_test999/metrics.json"),
    ("DeepSeek-VL2", EVAL / "local_gpu_baseline_gap_completion_20260810/deepseek_vl2/pathmmu_letter_only_v2/test999/metrics.json"),
    ("ScaleReasoner-R1", EVAL / "pathology_baselines_contemporary_20260810/scalereasoner_r1/pathmmu_test999/metrics.json"),
    ("LLaVA-Med-v1.5", EVAL / "llava_med_formal_20260814/pathmmu_option_text_test999/metrics.json"),
    ("Qwen-VL-Plus", EVAL / "hosted_baseline_full_20260801/qwen-vl-plus/pathmmu/metrics.json"),
    ("Claude Haiku 4.5", EVAL / "hosted_baseline_full_20260801/claude-haiku-4-5-20251001/pathmmu/metrics.json"),
    ("PLIP (image-option matching)", EVAL / "pathmmu_diagnostic/test999_matching_20260728_223308/plip/metrics.json"),
    ("CONCH (image-option matching)", EVAL / "pathmmu_diagnostic/test999_matching_20260728_223308/conch/metrics.json"),
)


CORRECTED_OMNI_OUTPUTS = (
    ("Qwen2.5-VL-3B", EVAL / "omnimedvqa_per_model_gated_v2_20260815/qwen2_5_vl_3b/full8518/metrics.json"),
    ("Lingshu-7B", EVAL / "omnimedvqa_per_model_gated_v2_20260815/lingshu_7b/full8518/metrics.json"),
    ("MedVLM-R1", EVAL / "omnimedvqa_per_model_gated_v2_20260815/medvlm_r1/full8518/metrics.json"),
    ("MedGemma-4B-IT", EVAL / "omnimedvqa_per_model_gated_v2_20260815/medgemma_4b_it/full8518/metrics.json"),
    ("ScaleReasoner-R1", EVAL / "omnimedvqa_per_model_gated_v2_20260815/scalereasoner_r1/full8518/metrics.json"),
    ("Llama-3.2-Vision-11B", EVAL / "omnimedvqa_per_model_gated_v2_20260815/llama3_2_vision_11b/full8518/metrics.json"),
    ("HuatuoGPT-Vision-7B", EVAL / "omnimedvqa_per_model_gated_v2_20260815/huatuogpt_vision_7b/full8518/metrics.json"),
    ("InternVL3-8B", EVAL / "omnimedvqa_per_model_gated_v2_20260815/internvl3_8b/full8518/metrics.json"),
    ("DeepSeek-VL2", EVAL / "omnimedvqa_per_model_gated_v2_20260815/deepseek_vl2/full8518/metrics.json"),
    ("Llama-3.2-Vision-90B", EVAL / "omnimedvqa_per_model_gated_v2_20260815/llama3_2_vision_90b/full8518/metrics.json"),
    ("LLaVA-Med-v1.5", EVAL / "omnimedvqa_per_model_gated_v2_20260815/llava_med_7b/full8518/metrics.json"),
)


PATHOLOGY_MATCHING_OMNI_OUTPUTS = (
    ("PLIP (image-option matching)", EVAL / "external_vqa_full_20260729_224503/plip/omnimedvqa/metrics.json"),
    ("CONCH (image-option matching)", EVAL / "external_vqa_full_20260729_224503/conch/omnimedvqa/metrics.json"),
)


def pct(value: Any) -> str:
    return "NA" if not isinstance(value, (int, float)) else f"{100 * value:.2f}"


def source_values(metrics: dict[str, Any] | None) -> list[str]:
    if not metrics:
        return ["RUNNING"] * 6
    by = metrics.get("by_source", {})
    names = ("Chest CT Scan", "ISIC2020", "Retinal OCT-C8", "Diabetic Retinopathy")
    return [
        pct(by.get(name, {}).get("contract_aligned_accuracy")) for name in names
    ] + [pct(metrics.get("contract_aligned_accuracy")), pct(metrics.get("official_accuracy"))]


def matching_source_values(metrics: dict[str, Any] | None) -> list[str]:
    """Read image-option matching metrics without relabelling them as generation scores."""
    if not metrics:
        return ["NA"] * 6
    by = metrics.get("by_source", {})
    names = ("Chest CT Scan", "ISIC2020", "Retinal OCT-C8", "Diabetic Retinopathy")
    return [pct(by.get(name, {}).get("accuracy")) for name in names] + [pct(metrics.get("accuracy")), "NA"]


def core_rows() -> list[dict[str, Any]]:
    payload = load(FINAL_JSON) or {}
    return list(payload.get("core_current_contract", []))


def repeat_rows() -> list[dict[str, Any]]:
    roots = (
        EVAL / "repeated_inference_pathmmu_test999_20260814",
        EVAL / "repeated_inference_pathmmu_test999_20260815",
    )
    result = []
    seen: set[Path] = set()
    for root in roots:
        for path in sorted(root.glob("*/summary.json")):
            # Several short aliases point at canonical result directories.  Do
            # not turn those aliases into duplicate paper rows.
            if path.parent.is_symlink():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            value = load(path)
            if value:
                result.append({"model": path.parent.name, "path": relative(path), **value})
    return result


def paper_tables() -> str:
    core = core_rows()
    lines = [
        "# 论文修订表格数据包（2026-08-15）", "",
        "> 所有数值均绑定到明确 JSON 路径。`RUNNING` 不借用旧 64-token OmniMedVQA 分数；`NA` 表示当前合同没有可用结果。", "",
        "## 修订 Table I：数据划分", "",
        "| Item | Count | Protocol note |", "|---|---:|---|",
        "| Formal image-disjoint pool | 5,385 | PubMed + EduContent rewritten pool |",
        "| SFT training | 3,000 | image-disjoint from validation/test |",
        "| RL training | 1,000 | disjoint from SFT/validation/test |",
        "| Validation | 385 | checkpoint/recipe selection only |",
        "| Test999 | 999 | full held-out engineering diagnostic; already accessed |", "",
        "## 修订 Table II-A：完整 PathMMU Test999 外部基线", "",
        "| Model | Accuracy (%) | N | Exact metrics source |", "|---|---:|---:|---|",
    ]
    for name, path in BASELINE_PATHMMU:
        value = load(path)
        count = first(value.get("count"), value.get("completed"), value.get("expected_count")) if value else None
        lines.append(f"| {name} | {pct(None if value is None else value.get('accuracy'))} | {count if count is not None else 'NA'} | `{relative(path)}` |")
    lines += [
        "", "说明：这些是完整 Test999 准确率表；PLIP/CONCH 使用图像与原始选项文本的余弦匹配，不是生成式问答。原稿 Table II 的 500-case GPT-4o 对话质量维度必须保留为独立的生成质量表，不能与本表 accuracy 混成同一统计口径。", "",
        "### 审稿人指定病理基础模型的可比性边界", "",
        "| Model | PathMMU | PathVQA | OmniMedVQA | Why the cell is or is not available |", "|---|---:|---:|---:|---|",
        "| CONCH | 33.83 matching | NA | 38.71 matching | 可运行的图像—文本编码器；仅以每题候选答案作 image-option matching，不是生成式 VQA |",
        "| UNI | NA | NA | NA | 视觉表征编码器，没有文本/问题/答案接口；应作任务定位讨论或另设线性探针，不应伪装成零样本 VQA |",
        "| PathChat | NA | NA | NA | 未核验到可复现的官方端到端权重与推理发布；按审稿意见作训练数据、监督和临床定位讨论 |",
        "| PLIP | 33.43 matching | NA | 23.78 matching | 与 CONCH 相同的受限 image-option matching 对照 |", "",
        "## 修订 Table III：主线模型完整 Test999 平均生成长度", "",
        "| Model/stage | Mean generated tokens | N | Exact metrics source |", "|---|---:|---:|---|",
    ]
    for row in core:
        path = Path(row["paths"]["pathmmu_test"])
        value = load(path) or {}
        mean_tokens = value.get("mean_generated_tokens")
        count = first(value.get("count"), value.get("completed"), value.get("expected_count"))
        token_text = "NA" if not isinstance(mean_tokens, (int, float)) else f"{mean_tokens:.2f}"
        lines.append(f"| {row['model']} | {token_text} | {count if count is not None else 'NA'} | `{relative(path)}` |")
    lines += [
        "", "说明：该表使用各模型完整 Test999 的实际生成记录重新计算；不沿用原稿中无法绑定到当前 checkpoint/合同的旧 token 数。", "",
        "## 修订 Table IV：统一 1024-token OmniMedVQA 分源结果", "",
        "合同：`omnimed_domain_think_answer_v4_1024`；主指标为 target-blind final-answer contract-aligned accuracy。", "",
        "| Model | Chest CT | ISIC2020 | OCT-C8 | DR | Micro aligned | Official sensitivity | Source |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in core:
        omni_path = Path(row["paths"]["omni"])
        values = source_values(load(omni_path))
        lines.append(f"| {row['model']} | {' | '.join(values)} | `{relative(omni_path)}` |")
    for name, path in CORRECTED_OMNI_OUTPUTS:
        values = source_values(load(path))
        lines.append(f"| {name} | {' | '.join(values)} | `{relative(path)}` |")
    for name, path in PATHOLOGY_MATCHING_OMNI_OUTPUTS:
        values = matching_source_values(load(path))
        lines.append(f"| {name} | {' | '.join(values)} | `{relative(path)}` |")
    lines += [
        "", "PLIP/CONCH 两行的五个 accuracy 单元格是 `image_to_raw_option_text_cosine_similarity`，与上方生成式 `omnimed_domain_think_answer_v4_1024` 合同不同；`Official sensitivity` 不适用。UNI 与 PathChat 没有兼容的逐题 VQA 输出，因此不进入该分源数值表。", "",
        "", "## 修订 Table V：当前分阶段训练主线", "",
        "| Model/stage | PathMMU Val | Test999 | PathVQA A/B diagnostic | Omni aligned | MMMU |", "|---|---:|---:|---:|---:|---:|",
    ]
    for row in core:
        lines.append(
            f"| {row['model']} | {pct(row.get('pathmmu_val'))} | {pct(row.get('pathmmu_test'))} | "
            f"{pct(row.get('pathvqa_fixed_ab'))} | {pct(row.get('omnimed_contract_aligned'))} | {pct(row.get('mmmu_nonmedical'))} |"
        )
    lines += [
        "", "## 多次随机推理统计", "",
        "合同：seeds 42–46，temperature=0.7，top-p=0.9，top-k disabled；t 区间衡量随机解码波动，不替代 case/image-cluster bootstrap。", "",
        "| Model | Runs | Mean (%) | SD (pp) | Student-t 95% CI (%) | Source |", "|---|---:|---:|---:|---:|---|",
    ]
    for row in repeat_rows():
        ci = row.get("student_t_95_ci", [None, None])
        lines.append(
            f"| {row['model']} | {row.get('run_count', 'NA')} | {pct(row.get('mean_accuracy'))} | "
            f"{pct(row.get('sample_standard_deviation'))} | [{pct(ci[0])}, {pct(ci[1])}] | `{row['path']}` |"
        )
    lines += [
        "", "## 原稿表格到修订数据的对应关系", "",
        "| Original table | Revision data | Current boundary |", "|---|---|---|",
        "| Table I | image-disjoint 3000/1000/385/999 split | replaces ambiguous 1385-only performance split description |",
        "| Table II | full Test999 accuracy + separate multi-Judge quality table | do not mix 500-case quality scores with full-test accuracy |",
        "| Table III | token lengths from each exact metrics JSON | regenerate after final Stage3 checkpoint is frozen |",
        "| Table IV | corrected OmniMedVQA source-wise table above | old 64-token values are historical only |",
        "| Table V | staged main-line table above | Stage3 500/1000/1500 remain visible until final checkpoint is declared |", "",
    ]
    return "\n".join(lines)


def readme(counts: Counter[str], total: int) -> str:
    return f"""# 评测结果追溯目录（2026-08-15）

本目录解决两个问题：一个 JSON 是哪个模型产生的，以及它使用了哪套评测协议。

## 判定规则

不要只依赖目录名。每个正式结果以三件套为准：

1. `run_config.json`：模型/adapter 路径及 hash、数据路径及 hash、prompt/generation contract、解码参数；
2. `predictions.jsonl`：逐题原始生成和目标盲解析结果；
3. `metrics.json`：聚合指标与 predictions SHA-256。

`result_lineage.json` 对两个 runs 根目录中的每个结果目录建立了机器可读记录，并给出 `traceability_level`：

- `complete`：模型、模型 hash、数据、数据 hash、prompt/合同、解码参数、逐题预测和汇总均存在；
- `strong_partial`：模型/数据/合同明确，但缺少部分 hash 或逐题文件；
- `legacy_partial`：模型和数据可定位，但旧结果缺少明确 prompt/generation contract；
- `ambiguous_nonstandard`：非标准诊断/汇总，不能仅凭名字写入论文主表。

当前共索引 {total} 个结果目录：{dict(counts)}。

## 关键协议边界

- OmniMedVQA 正式修订口径：`omnimed_domain_think_answer_v4_1024`；旧 64-token 结果不可混入。
- PathVQA 主报告为原始自由 Yes/No；固定 `A=Yes/B=No` 仅作接口敏感性诊断。
- PathMMU Test999 已用于工程诊断，不能再声称是未触碰测试集。
- `paper_tables.md` 只从显式 JSON 路径取值；运行中的格子写 `RUNNING`，不借用不兼容历史分数。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    entries = build_lineage()
    counts = Counter(entry["traceability_level"] for entry in entries)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(WORK),
        "entry_count": len(entries),
        "traceability_counts": dict(counts),
        "entries": entries,
    }
    (out / "result_lineage.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out / "README.md").write_text(readme(counts, len(entries)), encoding="utf-8")
    (out / "paper_tables.md").write_text(paper_tables(), encoding="utf-8")
    print(json.dumps({"output": str(out), "entries": len(entries), "counts": dict(counts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
