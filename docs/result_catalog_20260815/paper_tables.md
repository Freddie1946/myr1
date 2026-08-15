# 论文修订表格数据包（2026-08-15）

> 所有数值均绑定到明确 JSON 路径。`RUNNING` 不借用旧 64-token OmniMedVQA 分数；`NA` 表示当前合同没有可用结果。

## 修订 Table I：数据划分

| Item | Count | Protocol note |
|---|---:|---|
| Formal image-disjoint pool | 5,385 | PubMed + EduContent rewritten pool |
| SFT training | 3,000 | image-disjoint from validation/test |
| RL training | 1,000 | disjoint from SFT/validation/test |
| Validation | 385 | checkpoint/recipe selection only |
| Test999 | 999 | full held-out engineering diagnostic; already accessed |

## 修订 Table II-A：完整 PathMMU Test999 外部基线

| Model | Accuracy (%) | N | Exact metrics source |
|---|---:|---:|---|
| Qwen2.5-VL-3B | 44.94 | 999 | `pathvlm_revision_eval_a100/runs/pathmmu_diagnostic/test999_development_20260728_215918/qwen3/metrics.json` |
| Qwen2.5-VL-7B | 48.55 | 999 | `pathvlm_revision_eval_a100/runs/pathmmu_diagnostic/test999_development_20260728_215918/qwen7/metrics.json` |
| Lingshu-7B | 58.36 | 999 | `pathvlm_revision_eval_a100/runs/pathmmu_diagnostic/test999_development_20260728_215918/lingshu/metrics.json` |
| InternVL3-8B | 55.36 | 999 | `pathvlm_revision_eval_a100/runs/pathmmu_diagnostic/test999_development_20260728_220821/internvl/metrics.json` |
| HuatuoGPT-Vision-7B | 53.15 | 999 | `pathvlm_revision_eval_a100/runs/pathmmu_diagnostic/test999_development_20260728_221830/huatuo/metrics.json` |
| MedGemma-4B-IT | 40.04 | 999 | `pathvlm_revision_eval_a100/runs/pathmmu_diagnostic/test999_development_20260728_220617/medgemma/metrics.json` |
| MedVLM-R1 | 41.34 | 999 | `pathvlm_revision_eval_a100/runs/pathmmu_diagnostic/test999_development_20260728_215918/medvlm/metrics.json` |
| Llama-3.2-Vision-11B | 22.22 | 999 | `pathvlm_revision_eval_a100/runs/local_gpu_baselines_20260803/llama32_11b/pathmmu_test999/metrics.json` |
| Llama-3.2-Vision-90B | 57.16 | 999 | `pathvlm_revision_eval_a100/runs/local_gpu_baselines_20260803/llama32_90b/pathmmu_test999/metrics.json` |
| DeepSeek-VL2 | 42.94 | 999 | `pathvlm_revision_eval_a100/runs/local_gpu_baseline_gap_completion_20260810/deepseek_vl2/pathmmu_letter_only_v2/test999/metrics.json` |
| ScaleReasoner-R1 | 64.86 | 999 | `pathvlm_revision_eval_a100/runs/pathology_baselines_contemporary_20260810/scalereasoner_r1/pathmmu_test999/metrics.json` |
| LLaVA-Med-v1.5 | 35.54 | 999 | `pathvlm_revision_eval_a100/runs/llava_med_formal_20260814/pathmmu_option_text_test999/metrics.json` |
| Qwen-VL-Plus | 54.75 | 999 | `pathvlm_revision_eval_a100/runs/hosted_baseline_full_20260801/qwen-vl-plus/pathmmu/metrics.json` |
| Claude Haiku 4.5 | 46.69 | 999 | `pathvlm_revision_eval_a100/runs/hosted_baseline_full_20260801/claude-haiku-4-5-20251001/pathmmu/metrics.json` |

说明：这些是完整 Test999 准确率表；原稿 Table II 的 500-case GPT-4o 对话质量维度必须保留为独立的生成质量表，不能与本表 accuracy 混成同一统计口径。

## 修订 Table III：主线模型完整 Test999 平均生成长度

| Model/stage | Mean generated tokens | N | Exact metrics source |
|---|---:|---:|---|
| Base Qwen2.5-VL-7B | 130.56 | 999 | `pathvlm_revision_eval_a100/runs/formal_selected_sft3000_20260811/pathmmu_test999/base/metrics.json` |
| Historical full-SFT3000 | 71.72 | 999 | `pathvlm_revision_eval_a100/runs/pathmmu_sft4000_stage2_20260729_221640/sft3000_parent_test999/metrics.json` |
| Historical Stage2 outcome-RL | 71.96 | 999 | `pathvlm_revision_eval_a100/runs/pathmmu_sft4000_stage2_20260729_221640/stage2_rl_test999/metrics.json` |
| Current L-r16 SFT3000 step80 | 61.08 | 999 | `pathvlm_revision_eval_a100/runs/formal_selected_sft3000_20260811/pathmmu_test999/step080/metrics.json` |
| Full rule-RL n4 step1000 | 63.05 | 999 | `pathvlm_revision_eval_a100/runs/full_language_rule_rl_clean_n4_n8_step1000_20260812/n4_fresh_step1000_pathmmu_eval/test999/metrics.json` |
| Full rule-RL n8 step1000 | 62.23 | 999 | `pathvlm_revision_eval_a100/runs/full_language_rule_rl_clean_n4_n8_step1000_20260812/n8_fresh_step1000_pathmmu_eval/test999/metrics.json` |
| GPT-4o Stage3 step500 | 71.85 | 999 | `pathvlm_revision_eval_a100/runs/stage3_gpt4o_n8_checkpoint500_test999_20260813_v2/metrics.json` |
| GPT-4o Stage3 step1000 | 72.98 | 999 | `pathvlm_revision_eval_a100/runs/stage3_gpt4o_n8_checkpoint1000_test999_20260813_v2/metrics.json` |
| GPT-4o Stage3 step1500 | 75.30 | 999 | `pathvlm_revision_eval_a100/runs/stage3_gpt4o_n8_checkpoint1500_final_eval_20260813/pathmmu_test999/metrics.json` |
| LoRA-SFT4000 control | 57.84 | 999 | `pathvlm_revision_eval_a100/runs/lora_sft4000_control_20260814/formal_8gpu_gbs96/pathmmu_test999/metrics.json` |

说明：该表使用各模型完整 Test999 的实际生成记录重新计算；不沿用原稿中无法绑定到当前 checkpoint/合同的旧 token 数。

## 修订 Table IV：统一 1024-token OmniMedVQA 分源结果

合同：`omnimed_domain_think_answer_v4_1024`；主指标为 target-blind final-answer contract-aligned accuracy。

| Model | Chest CT | ISIC2020 | OCT-C8 | DR | Micro aligned | Official sensitivity | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| Base Qwen2.5-VL-7B | 49.14 | 45.95 | 61.11 | 69.14 | 59.00 | 57.38 | `pathvlm_revision_eval_a100/runs/core_historical_corrected_eval_20260813/base/omnimedvqa_8518/metrics.json` |
| Historical full-SFT3000 | 36.74 | 71.08 | 45.59 | 52.07 | 50.97 | 48.45 | `pathvlm_revision_eval_a100/runs/core_historical_corrected_eval_20260813/sft3000/omnimedvqa_8518/metrics.json` |
| Historical Stage2 outcome-RL | 36.05 | 72.97 | 47.63 | 54.41 | 52.78 | 49.20 | `pathvlm_revision_eval_a100/runs/core_historical_corrected_eval_20260813/stage2/omnimedvqa_8518/metrics.json` |
| Current L-r16 SFT3000 step80 | 44.89 | 60.76 | 64.67 | 63.58 | 61.66 | 57.42 | `pathvlm_revision_eval_a100/runs/full_language_rule_rl_clean_n4_n8_step1000_20260812/candidate_ood_eval/omnimedvqa/sft_parent/metrics.json` |
| Full rule-RL n4 step1000 | 47.07 | 68.54 | 71.41 | 76.84 | 69.70 | 57.20 | `pathvlm_revision_eval_a100/runs/full_language_rule_rl_clean_n4_n8_step1000_20260812/candidate_ood_eval/omnimedvqa/full_rl_n4_step1000/metrics.json` |
| Full rule-RL n8 step1000 | 47.42 | 71.01 | 71.17 | 74.45 | 69.50 | 56.80 | `pathvlm_revision_eval_a100/runs/full_language_rule_rl_clean_n4_n8_step1000_20260812/candidate_ood_eval/omnimedvqa/full_rl_n8_step1000/metrics.json` |
| GPT-4o Stage3 step500 | 49.25 | 78.67 | 75.45 | 79.47 | 74.34 | 54.16 | `pathvlm_revision_eval_a100/runs/stage3_gpt4o_n8_checkpoint_ood_comparison_20260813/checkpoint500/omnimedvqa_8518/metrics.json` |
| GPT-4o Stage3 step1000 | 47.42 | 74.18 | 76.59 | 79.08 | 73.76 | 55.87 | `pathvlm_revision_eval_a100/runs/stage3_gpt4o_n8_checkpoint_ood_comparison_20260813/checkpoint1000/omnimedvqa_8518/metrics.json` |
| GPT-4o Stage3 step1500 | 46.04 | 77.78 | 77.71 | 79.86 | 75.01 | 57.18 | `pathvlm_revision_eval_a100/runs/stage3_gpt4o_n8_checkpoint1500_final_eval_20260813/omnimedvqa_8518/metrics.json` |
| LoRA-SFT4000 control | 43.17 | 51.33 | 67.38 | 59.24 | 59.97 | 57.60 | `pathvlm_revision_eval_a100/runs/lora_sft4000_control_20260814/formal_8gpu_gbs96/omnimedvqa_8518/metrics.json` |
| Qwen2.5-VL-3B | 45.46 | 59.56 | 67.03 | 54.12 | 60.33 | 52.42 | `pathvlm_revision_eval_a100/runs/omnimedvqa_per_model_gated_v2_20260815/qwen2_5_vl_3b/full8518/metrics.json` |
| Lingshu-7B | 47.42 | 89.05 | 80.03 | 76.69 | 77.57 | 51.53 | `pathvlm_revision_eval_a100/runs/omnimedvqa_per_model_gated_v2_20260815/lingshu_7b/full8518/metrics.json` |
| MedVLM-R1 | 43.40 | 50.44 | 72.31 | 70.60 | 64.89 | 43.26 | `pathvlm_revision_eval_a100/runs/omnimedvqa_per_model_gated_v2_20260815/medvlm_r1/full8518/metrics.json` |
| MedGemma-4B-IT | 43.28 | 81.08 | 76.42 | 83.08 | 75.50 | 50.52 | `pathvlm_revision_eval_a100/runs/omnimedvqa_per_model_gated_v2_20260815/medgemma_4b_it/full8518/metrics.json` |
| ScaleReasoner-R1 | 50.40 | 56.08 | 66.36 | 72.65 | 64.33 | 39.86 | `pathvlm_revision_eval_a100/runs/omnimedvqa_per_model_gated_v2_20260815/scalereasoner_r1/full8518/metrics.json` |
| Llama-3.2-Vision-11B | 32.03 | 72.66 | 71.74 | 43.25 | 60.99 | 31.85 | `pathvlm_revision_eval_a100/runs/omnimedvqa_per_model_gated_v2_20260815/llama3_2_vision_11b/full8518/metrics.json` |
| HuatuoGPT-Vision-7B | 43.40 | 38.67 | 75.72 | 68.41 | 63.78 | 57.27 | `pathvlm_revision_eval_a100/runs/omnimedvqa_per_model_gated_v2_20260815/huatuogpt_vision_7b/full8518/metrics.json` |
| InternVL3-8B | 47.30 | 49.62 | 81.42 | 66.80 | 68.51 | 61.22 | `pathvlm_revision_eval_a100/runs/omnimedvqa_per_model_gated_v2_20260815/internvl3_8b/full8518/metrics.json` |
| DeepSeek-VL2 | 32.72 | 60.82 | 52.07 | 55.49 | 52.54 | 9.46 | `pathvlm_revision_eval_a100/runs/omnimedvqa_per_model_gated_v2_20260815/deepseek_vl2/full8518/metrics.json` |
| Llama-3.2-Vision-90B | 40.99 | 82.78 | 78.91 | 55.73 | 70.17 | 39.00 | `pathvlm_revision_eval_a100/runs/omnimedvqa_per_model_gated_v2_20260815/llama3_2_vision_90b/full8518/metrics.json` |
| LLaVA-Med-v1.5 | 28.47 | 57.22 | 43.68 | 38.66 | 43.43 | 41.11 | `pathvlm_revision_eval_a100/runs/omnimedvqa_per_model_gated_v2_20260815/llava_med_7b/full8518/metrics.json` |

## 修订 Table V：当前分阶段训练主线

| Model/stage | PathMMU Val | Test999 | PathVQA A/B diagnostic | Omni aligned | MMMU |
|---|---:|---:|---:|---:|---:|
| Base Qwen2.5-VL-7B | 50.91 | 48.65 | 60.56 | 59.00 | 56.03 |
| Historical full-SFT3000 | 59.74 | 58.26 | 56.25 | 50.97 | 43.10 |
| Historical Stage2 outcome-RL | 61.04 | 60.96 | 55.59 | 52.78 | 47.41 |
| Current L-r16 SFT3000 step80 | 54.55 | 56.66 | 61.99 | 61.66 | 49.14 |
| Full rule-RL n4 step1000 | 59.74 | 61.16 | 62.61 | 69.70 | 53.45 |
| Full rule-RL n8 step1000 | 64.16 | 63.46 | 62.61 | 69.50 | 54.31 |
| GPT-4o Stage3 step500 | 65.71 | 64.36 | 60.44 | 74.34 | 58.62 |
| GPT-4o Stage3 step1000 | 66.75 | 63.96 | 62.28 | 73.76 | 60.34 |
| GPT-4o Stage3 step1500 | 65.19 | 63.86 | 61.36 | 75.01 | 54.31 |
| LoRA-SFT4000 control | 52.99 | 56.56 | 63.12 | 59.97 | 56.03 |

## 多次随机推理统计

合同：seeds 42–46，temperature=0.7，top-p=0.9，top-k disabled；t 区间衡量随机解码波动，不替代 case/image-cluster bootstrap。

| Model | Runs | Mean (%) | SD (pp) | Student-t 95% CI (%) | Source |
|---|---:|---:|---:|---:|---|
| full_rule_rl_n8_step1000 | 5 | 61.16 | 1.05 | [59.85, 62.47] | `pathvlm_revision_eval_a100/runs/repeated_inference_pathmmu_test999_20260814/full_rule_rl_n8_step1000/summary.json` |
| l_r16_sft80 | 5 | 52.65 | 1.09 | [51.30, 54.01] | `pathvlm_revision_eval_a100/runs/repeated_inference_pathmmu_test999_20260814/l_r16_sft80/summary.json` |
| stage3_gpt4o_step1500 | 5 | 63.10 | 0.81 | [62.10, 64.11] | `pathvlm_revision_eval_a100/runs/repeated_inference_pathmmu_test999_20260814/stage3_gpt4o_step1500/summary.json` |
| base_qwen2_5_vl_7b | 5 | 49.31 | 0.48 | [48.71, 49.91] | `pathvlm_revision_eval_a100/runs/repeated_inference_pathmmu_test999_20260815/base_qwen2_5_vl_7b/summary.json` |
| rule_rl4000_checkpoint2500_provisional | 5 | 60.26 | 1.53 | [58.37, 62.16] | `pathvlm_revision_eval_a100/runs/repeated_inference_pathmmu_test999_20260815/rule_rl4000_checkpoint2500_provisional/summary.json` |
| stage2_continued_rule_rl1000 | 5 | 60.92 | 0.72 | [60.02, 61.82] | `pathvlm_revision_eval_a100/runs/repeated_inference_pathmmu_test999_20260815/stage2_continued_rule_rl1000/summary.json` |
| stage2_outcome_grpo | 5 | 58.72 | 0.89 | [57.61, 59.83] | `pathvlm_revision_eval_a100/runs/repeated_inference_pathmmu_test999_20260815/stage2_outcome_grpo/summary.json` |
| stage3_gpt4o_step1000 | 5 | 63.62 | 0.98 | [62.41, 64.84] | `pathvlm_revision_eval_a100/runs/repeated_inference_pathmmu_test999_20260815/stage3_gpt4o_step1000/summary.json` |
| stage3_gpt4o_step500 | 5 | 63.66 | 0.86 | [62.60, 64.73] | `pathvlm_revision_eval_a100/runs/repeated_inference_pathmmu_test999_20260815/stage3_gpt4o_step500/summary.json` |

## 原稿表格到修订数据的对应关系

| Original table | Revision data | Current boundary |
|---|---|---|
| Table I | image-disjoint 3000/1000/385/999 split | replaces ambiguous 1385-only performance split description |
| Table II | full Test999 accuracy + separate multi-Judge quality table | do not mix 500-case quality scores with full-test accuracy |
| Table III | token lengths from each exact metrics JSON | regenerate after final Stage3 checkpoint is frozen |
| Table IV | corrected OmniMedVQA source-wise table above | old 64-token values are historical only |
| Table V | staged main-line table above | Stage3 500/1000/1500 remain visible until final checkpoint is declared |
