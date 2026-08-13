# 最终统一结果表（自动生成）

> 百分数均由明确的 `metrics.json` 路径生成。`NA` 表示该合同下尚未完成，绝不从不兼容的旧评测借值。PathMMU Test999 是开发诊断测试，不能表述为未触碰最终测试。

## 当前核心主线（固定 A=Yes/B=No PathVQA 合同）

| Model | PathMMU Val | PathMMU Test999 | PathVQA Test A/B | Parse | Cap hit | Omni aligned | Omni official | MMMU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen2.5-VL-7B | 50.91 | 48.65 | 60.56 | 97.35 | 0.00 | 59.00 | 57.38 | 56.03 |
| Historical full-SFT3000 | 59.74 | 58.26 | 56.25 | 99.91 | 0.00 | 50.97 | 48.45 | 43.10 |
| Historical Stage2 outcome-RL | 61.04 | 60.96 | 55.59 | 99.91 | 0.00 | NA | NA | 47.41 |
| Current L-r16 SFT3000 step80 | 54.55 | 56.66 | 61.99 | 99.88 | 0.00 | 61.66 | 57.42 | 49.14 |
| Full rule-RL n4 step1000 | 59.74 | 61.16 | 62.61 | 99.79 | 0.00 | 69.70 | 57.20 | 53.45 |
| Full rule-RL n8 step1000 | 64.16 | 63.46 | 62.61 | 99.79 | 0.00 | 69.50 | 56.80 | 54.31 |
| GPT-4o Stage3 step500 | 65.71 | 64.36 | 60.44 | 99.82 | 0.00 | 74.34 | 54.16 | 58.62 |
| GPT-4o Stage3 step1000 | 66.75 | 63.96 | 62.28 | 99.41 | 0.00 | 73.76 | 55.87 | 60.34 |
| GPT-4o Stage3 step1500 | 65.19 | 63.86 | 61.36 | 98.48 | 0.00 | 75.01 | 57.18 | 54.31 |
| LoRA-SFT4000 control | 52.99 | 56.56 | 63.12 | 100.00 | 0.00 | NA | NA | 56.03 |

## 固定总池 1000 的 SFT/RL 配比筛查

| Model | PathMMU Val | PathMMU Test999 | PathVQA Test A/B | Parse | Cap hit | Omni aligned | Omni official | MMMU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 SFT + 750 rule-RL | 50.39 | 47.25 | 57.76 | 99.94 | 0.00 | NA | NA | 54.31 |
| 500 SFT + 500 rule-RL | 51.95 | 51.75 | 57.20 | 99.97 | 0.00 | NA | NA | 45.69 |
| 750 SFT + 250 rule-RL | 50.91 | 49.75 | 55.62 | 99.94 | NA | NA | NA | 43.97 |

## Test999 五次随机推理重复

合同：seeds 42–46，temperature=0.7，top-p=0.9，top-k disabled，max_new_tokens=1024。主结果仍以 greedy 为准；这里的 t 区间衡量解码随机性，不替代 case-bootstrap。

| Model | Runs | Mean | SD | t 95% CI |
|---|---:|---:|---:|---:|
| l_r16_sft80 | NA | NA | NA | NA |
| full_rule_rl_n8_step1000 | NA | NA | NA | NA |
| stage3_gpt4o_step1500 | NA | NA | NA | NA |

## 外部基线（历史统一 short-answer / Yes-No 合同）

> 以下结果已经全量跑完，保留原合同用于可复现对比；不能解释为已使用当前 A/B prompt 重跑。

| Model | PathMMU Test999 | PathVQA yes/no3362 | OmniMedVQA aligned |
|---|---:|---:|---:|
| Qwen2.5-VL-3B | 44.94 | 58.06 | 60.72 |
| Lingshu-7B | 58.36 | 85.22 | 80.37 |
| InternVL3-8B | 55.36 | 65.50 | 72.58 |
| HuatuoGPT-Vision-7B | 53.15 | 64.78 | 71.32 |
| MedGemma-4B-IT | 40.04 | 62.17 | 71.62 |
| MedVLM-R1 | 41.34 | 56.28 | 50.89 |
| Llama-3.2-Vision-90B | 57.16 | 62.49 | 68.09 |
| Llama-3.2-Vision-11B | 22.22 | 55.98 | 53.90 |
| DeepSeek-VL2 | 42.94 | 59.52 | 51.56 |
| ScaleReasoner-R1 | 64.86 | 57.53 | 43.30 |
| Qwen-VL-Plus | 54.75 | 67.58 | 72.22 |
| Claude Haiku 4.5 | 46.65 | 40.30 | 56.22 |

完整分子/分母、official sensitivity 与 cap caveat：`docs/20260810_final_closed_benchmark_and_multijudge_results.md`。

## 已完成的异构 Judge 稳健性

GPT-4o、Claude Sonnet 4.6、Gemini 3.1 Pro 已对冻结的 100-case、6-model panel 完成 2,880 个盲评判断。两条 Stage3 臂相对 Stage2 的宏平均点估计在三个 Judge 下均为正，但六个 95% CI 均跨 0；因此只能写成方向一致的小幅提升，不能宣称统计显著。人工专家评分仍由用户后补。

## 合同边界

- 外部 PathVQA 基线已经有全量结果，但大多使用旧 short-answer/自由 Yes-No 合同；详见 `docs/20260810_final_closed_benchmark_and_multijudge_results.md`，不与上表伪装成同 prompt 比较。
- 人工病理专家 ROI/生成质量评分由用户后续完成，本表显式排除该唯一人工缺口。
- 不扩展全模型配对反转分析；仅保留最终核心对照所需的最小 paired bootstrap/McNemar。

机器可读数据：`protocol/final_unified_results_tables_20260814.json`
