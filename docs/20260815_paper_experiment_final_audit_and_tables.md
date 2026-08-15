# 论文实验最终审计与可直接制表数据（2026-08-15）

> 状态更新：本文件生成后列出的 Stage3 自由 Yes/No 与 Stage2-continued OmniMedVQA 缺口已于
> 2026-08-15 补齐。请以
> `20260815_reviewer_response_manuscript_revision_and_complete_tables.md` 和
> `protocol/final_revision_assets_audit_20260815.json` 为最新权威入口；本文件保留为补跑前审计快照。

## 结论先行

除人工病理复核外，主论文训练链、主要域内/域外评测、外部基线、数据配比筛查、重复随机推理、异构 Judge、bad-case 与计算型可解释性实验已经完成。严格意义上仍有三个非人工边界，不能写成“全部完成”：

1. `0 SFT + 4000 rule-RL` 仅到 checkpoint-2500。它已覆盖 4,000 条至少一次并完成统一评测和五次随机推理，但未到原计划的 6,000 steps，因此只可报告为 provisional 极端对照。
2. 当前 n4/n8 step1000 与 GPT-4o Stage3 500/1500 的 PathVQA 已完成固定 `A=Yes/B=No` 生成评测，但尚未全部按原始自由 Yes/No、2048-token、实际生成合同统一重跑。A/B 是接口敏感性补充结果，不能替代原始 Yes/No 主指标。
3. `Stage2 后继续 1000 rule-RL` 已有 PathMMU、PathVQA 与五次随机推理，但没有统一 OmniMedVQA/MMMU 端点。

人工部分仍包括：60 例奖励可信度评分，以及 20 例可解释性候选 ROI 的病理专家复核。自动评分和外部闭源模型框不能冒充人工专家结果。

## Table I：修订后的数据划分

| Item | Count | Reporting boundary |
|---|---:|---|
| PathMMU total images | 24,067 | 原始数据集规模 |
| PathMMU total Q&A | 33,428 | 原始数据集规模 |
| Formal image-disjoint pool | 5,385 | PubMed + EduContent rewritten pool |
| SFT train | 3,000 | 与 RL/validation/test 图像隔离 |
| RL train | 1,000 | 与 SFT/validation/test 图像隔离 |
| Validation | 385 | 仅用于 checkpoint/recipe 选择 |
| Test999 | 999 | 已用于工程诊断，不能称为 untouched final test |

原稿中的“performance testing 1,385”必须拆成 validation385 + Test999，并披露 Test999 已被访问。

## Table II-A：当前主线完整性能

百分数；Omni 主指标为统一 1024-token target-blind contract-aligned accuracy，official 仅作敏感性分析。PathVQA A/B 是补充接口诊断。

| Model/stage | PathMMU Val385 | PathMMU Test999 | PathVQA A/B Test3362 | Parse | Omni aligned | Omni official | MMMU non-medical |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen2.5-VL-7B | 50.91 | 48.65 | 60.56 | 97.35 | 59.00 | 57.38 | 56.03 |
| Historical full-SFT3000 | 59.74 | 58.26 | 56.25 | 99.91 | 50.97 | 48.45 | 43.10 |
| Historical Stage2 outcome-RL | 61.04 | 60.96 | 55.59 | 99.91 | 52.78 | 49.20 | 47.41 |
| Current L-r16 SFT3000 step80 | 54.55 | 56.66 | 61.99 | 99.88 | 61.66 | 57.42 | 49.14 |
| Full rule-RL n4 step1000 | 59.74 | 61.16 | 62.61 | 99.79 | 69.70 | 57.20 | 53.45 |
| Full rule-RL n8 step1000 | 64.16 | 63.46 | 62.61 | 99.79 | 69.50 | 56.80 | 54.31 |
| GPT-4o Stage3 step500 | 65.71 | **64.36** | 60.44 | 99.82 | 74.34 | 54.16 | 58.62 |
| GPT-4o Stage3 step1000 | **66.75** | 63.96 | 62.28 | 99.41 | 73.76 | 55.87 | **60.34** |
| GPT-4o Stage3 step1500 | 65.19 | 63.86 | 61.36 | 98.48 | **75.01** | 57.18 | 54.31 |
| LoRA-SFT4000 control | 52.99 | 56.56 | **63.12** | 100.00 | 59.97 | 57.60 | 56.03 |

Stage3 没有一个 checkpoint 在所有指标上同时占优：step500 的 Test999 最高，step1000 的 Val/MMMU 最高，step1500 的 Omni aligned 最高。因此应展示早/中/晚三点，不能事后只挑某一数据集最优点宣称全局最佳。

### PathVQA 主合同完备度

原始自由 Yes/No 是跨模型主合同；固定 A/B 只解释输出接口漂移。

| Model group | Original generated Yes/No | Fixed generated A/B | Boundary |
|---|---|---|---|
| 历史 Base/full-SFT/Stage2 | 已有：66.42 / 51.40 / 49.64 | 已统一重评 | 原 Yes/No 为旧合同；A/B 为新补充诊断 |
| L-r16 SFT3000 | 仅有 62.52% forced-logit 敏感性结果 | 61.99%，完整 | forced logits 不能当自由生成主结果 |
| Full rule-RL n4/n8 step1000 | 未按统一 2048-token 自由生成补齐 | 均为 62.61%，完整 | 原始主合同缺失 |
| GPT-4o Stage3 500/1000/1500 | 仅旧 step1000 有 52.26%；500/1500 未补齐 | 60.44 / 62.28 / 61.36，完整 | 需要时统一重跑自由 Yes/No |
| LoRA-SFT4000 | 未补齐 | 63.12%，完整 | A/B 仅补充 |

## Table II-B：外部模型基线

PathVQA 为原始 Yes/No；Omni 为 2026-08-15 统一 1024-token 重跑。Hosted 模型尚无同合同统一 Omni 时不借用旧值。

| Model | PathMMU Test999 | PathVQA Yes/No3362 | Omni aligned | Omni official |
|---|---:|---:|---:|---:|
| Qwen2.5-VL-3B | 44.94 | 58.06 | 60.33 | 52.42 |
| Lingshu-7B | 58.36 | **85.22** | **77.57** | 51.53 |
| InternVL3-8B | 55.36 | 65.50 | 68.51 | **61.22** |
| HuatuoGPT-Vision-7B | 53.15 | 64.78 | 63.78 | 57.27 |
| MedGemma-4B-IT | 40.04 | 62.17 | 75.50 | 50.52 |
| MedVLM-R1 | 41.34 | 56.28 | 64.89 | 43.26 |
| Llama-3.2-Vision-90B | 57.16 | 62.49 | 70.17 | 39.00 |
| Llama-3.2-Vision-11B | 22.22 | 55.98 | 60.99 | 31.85 |
| DeepSeek-VL2 | 42.94 | 59.52 | 52.54 | 9.46 |
| ScaleReasoner-R1 | **64.86** | 57.53 | 64.33 | 39.86 |
| LLaVA-Med-v1.5 | 35.54 | 56.57 | 43.43 | 41.11 |
| Qwen-VL-Plus | 54.75 | 67.58 | NA | NA |
| Claude Haiku 4.5 | 46.69 | 40.30 | NA | NA |

旧 64-token Omni 分数仅作历史记录，不与上表统一结果混用。Grok hosted baseline 仅部分完成，不能进入完整基线表。

## Table II-C：盲化异构 Judge 生成质量

冻结 100-case、6-model panel；GPT-4o、Claude Sonnet 4.6、Gemini 3.1 Pro 共完成 2,880 个判断。宏平均为六个 `[0,1]` 维度的平均。

| Candidate | GPT-4o macro | Claude macro | Gemini macro |
|---|---:|---:|---:|
| Base Qwen2.5-VL-7B | 0.7922 | 0.5987 | 0.6335 |
| SFT3000 | 0.7717 | 0.5939 | 0.6213 |
| SFT4000 control | 0.7634 | 0.5873 | 0.6177 |
| Stage2 outcome-RL | 0.7804 | 0.5952 | 0.6126 |
| Stage3 GPT-4o | 0.7937 | 0.6123 | 0.6209 |
| Stage3 Grok 4.3 | **0.7993** | **0.6268** | 0.6223 |

| Judge | GPT-4o Stage3 − Stage2 | Grok Stage3 − Stage2 |
|---|---:|---:|
| GPT-4o | +0.0132, 95% CI [-0.0262, +0.0526] | +0.0189 [-0.0207, +0.0576] |
| Claude Sonnet 4.6 | +0.0170 [-0.0227, +0.0555] | +0.0315 [-0.0107, +0.0726] |
| Gemini 3.1 Pro | +0.0082 [-0.0105, +0.0276] | +0.0096 [-0.0120, +0.0313] |

所有点估计方向为正，但六个 CI 均跨 0；只能写“方向一致的小幅改善”，不能写统计显著。人工评分用于验证自动过程奖励可信度，而不是补造显著性。

## Table III：完整 Test999 平均生成长度

| Model/stage | Mean generated tokens |
|---|---:|
| Base | 130.56 |
| Historical full-SFT3000 | 71.72 |
| Historical Stage2 outcome-RL | 71.96 |
| L-r16 SFT3000 step80 | 61.08 |
| Full rule-RL n4 step1000 | 63.05 |
| Full rule-RL n8 step1000 | 62.23 |
| GPT-4o Stage3 step500 | 71.85 |
| GPT-4o Stage3 step1000 | 72.98 |
| GPT-4o Stage3 step1500 | 75.30 |
| LoRA-SFT4000 | 57.84 |

## Table IV：统一 OmniMedVQA 分源结果

| Model | Chest CT | ISIC2020 | OCT-C8 | DR | Micro aligned | Official sensitivity |
|---|---:|---:|---:|---:|---:|---:|
| Base | 49.14 | 45.95 | 61.11 | 69.14 | 59.00 | 57.38 |
| Historical full-SFT3000 | 36.74 | 71.08 | 45.59 | 52.07 | 50.97 | 48.45 |
| Historical Stage2 outcome-RL | 36.05 | 72.97 | 47.63 | 54.41 | 52.78 | 49.20 |
| L-r16 SFT3000 step80 | 44.89 | 60.76 | 64.67 | 63.58 | 61.66 | 57.42 |
| Full rule-RL n4 step1000 | 47.07 | 68.54 | 71.41 | 76.84 | 69.70 | 57.20 |
| Full rule-RL n8 step1000 | 47.42 | 71.01 | 71.17 | 74.45 | 69.50 | 56.80 |
| GPT-4o Stage3 step500 | 49.25 | 78.67 | 75.45 | 79.47 | 74.34 | 54.16 |
| GPT-4o Stage3 step1000 | 47.42 | 74.18 | 76.59 | 79.08 | 73.76 | 55.87 |
| GPT-4o Stage3 step1500 | 46.04 | 77.78 | 77.71 | 79.86 | **75.01** | 57.18 |
| LoRA-SFT4000 | 43.17 | 51.33 | 67.38 | 59.24 | 59.97 | 57.60 |

统一基线的完整分源表见 `docs/result_catalog_20260815/paper_tables.md`。总体上 Stage3 的增益主要来自 ISIC/OCT/DR，Chest CT 没有同步改善；不能写成所有域外模态普遍提升。

## Table V-A：固定 1,000 总池的数据配比筛查

| SFT | Rule-RL | PathMMU Val | Test999 | PathVQA A/B | Omni aligned | MMMU |
|---:|---:|---:|---:|---:|---:|---:|
| 250 | 750 | 50.39 | 47.25 | **57.76** | 58.03 | **54.31** |
| 500 | 500 | **51.95** | **51.75** | 57.20 | 57.75 | 45.69 |
| 750 | 250 | 50.91 | 49.75 | 55.62 | **60.00** | 43.97 |

这是低成本配比筛查，不是完整 4,000 样本规模曲线。三组没有单调关系，说明结果依赖 SFT/RL 分配和任务分布，而非“RL 比例越高越好”。

## Table V-B：关键训练消融与边界

| Arm | Status | PathMMU Val | Test999 | PathVQA Yes/No | PathVQA A/B | Omni aligned | MMMU |
|---|---|---:|---:|---:|---:|---:|---:|
| LoRA-SFT4000 | complete | 52.99 | 56.56 | NA | 63.12 | 59.97 | 56.03 |
| 3000 LoRA-SFT + full rule-RL n4 | complete | 59.74 | 61.16 | NA | 62.61 | 69.70 | 53.45 |
| 3000 LoRA-SFT + full rule-RL n8 | complete | 64.16 | 63.46 | NA | 62.61 | 69.50 | 54.31 |
| 0 SFT + 4000 rule-RL checkpoint2500 | provisional | 63.38 | 63.76 | 65.20 | 62.85 | 67.59 | NA |
| Stage2 后继续 1000 rule-RL checkpoint1500 | partial | 59.74 | 61.86 | 59.31 | 56.19 | NA | NA |

LoRA-SFT4000 相比 LoRA-SFT3000 的 Test999 基本没有提高，而 n4/n8 rule-RL 相对 LoRA-SFT4000 分别提高 +4.60/+6.91 pp；配对 bootstrap CI 分别为 `[+2.06,+7.20]`、`[+4.15,+9.65]`，McNemar p=`0.000860`、`1.46e-6`。这支持“增益不只是继续 SFT exposure”。

## Table V-C：RL 机制筛查

| Arm | Exposure | Train probe | PathMMU Val | Interpretation |
|---|---:|---:|---:|---|
| R0 accuracy + format | step50 | 51.95 | 55.06 | matched reward arm |
| R1 accuracy only | step50 | 53.52 | 56.36 | 一致但不显著的优势；格式未退化 |
| R1 G20 LR=3e-6 | step25 / 0.5 epoch | 52.34 | 56.36 | 适度提高 LR 有帮助 |
| R1 G2 LR=1e-6 | step250 / 0.5 epoch | 52.34 | 55.32 | 增加 update density 未改善 Val |

SFT-r32 的 Val 最佳点为 step64 的 56.36%，但后期 PathVQA/MMMU retention 更差，因此主要增加 specialization capacity，未被选为主 parent。

## Table S1：五次随机推理置信区间

统一合同：seeds 42–46，temperature=0.7，top-p=0.9，top-k disabled，max_new_tokens=1024。主结果仍使用 greedy；下表衡量随机解码波动。

| Model | Mean | SD | Student-t 95% CI |
|---|---:|---:|---:|
| Base | 49.31 | 0.48 | [48.71, 49.91] |
| L-r16 SFT80 | 52.65 | 1.09 | [51.30, 54.01] |
| Historical Stage2 outcome-RL | 58.72 | 0.89 | [57.61, 59.83] |
| Stage2 continued rule-RL1000 | 60.92 | 0.72 | [60.02, 61.82] |
| Full rule-RL n8 step1000 | 61.16 | 1.05 | [59.85, 62.47] |
| GPT-4o Stage3 step500 | 63.66 | 0.86 | [62.60, 64.73] |
| GPT-4o Stage3 step1000 | 63.62 | 0.98 | [62.41, 64.84] |
| GPT-4o Stage3 step1500 | 63.10 | 0.81 | [62.10, 64.11] |
| 4000 rule-RL checkpoint2500 provisional | 60.26 | 1.53 | [58.37, 62.16] |

这些 t 区间不能替代逐题 paired bootstrap。Stage3 三个 checkpoint 的区间高度重叠，不能声称其中一个稳定显著优于另两个。

## Table S2：可解释性计算结果

| Funnel / statistic | Result |
|---|---:|
| 预先冻结候选 | 160 |
| 双外部模型独立标注后可解析 | 120/123 |
| 双模型 ROI 共识病例 | 94 |
| Target clean-correct | 51/94 |
| clean-correct 且生成/score 一致 | 47 |
| reference deletion margin drop > 0 | 34 |
| strict effective cases | 20（A5/B7/C3/D5） |
| Reference deletion mean margin drop | 0.4542, 95% CI [0.1520, 0.8079] |
| Random matched deletion | 0.3452 [0.1550, 0.5744] |
| Neighbor matched deletion | 0.2456 [0.0661, 0.4520] |
| Reference − random extra drop | 0.1090 [-0.0648, 0.2989] |

当前结果支持 20 个强个例，但 reference-minus-random 的总体 CI 跨 0，尚不支持群体层面的显著 ROI 优势。专家验证后可进一步报告病理学定位有效性；未复核前统一称 `predefined pseudo-reference evidence regions`。

已有 activation patching 辅助结果显示：在正 deletion-gap 的 19 例中，layer-0 clean visual-token patch 的平均恢复比例约 0.971；相对 opposite-direction control 的平均 margin recovery 为 0.783；query-position patch 的 raw recovery 主要在后层上升（layer20/24/27 约 0.345/0.486/0.645）。这些是模型内部因果传播证据，不等同于病理专家定位正确性。

## 审稿意见完成矩阵

| Reviewer request | Status | Claim boundary |
|---|---|---|
| 完整 held-out Test999 | complete | 已被开发访问，不能称 untouched |
| 随机重复推理与 CI | complete for selected main stages | 不要求每个外部 baseline 重复五次 |
| 配对统计/McNemar/bootstrap | complete for key comparisons | 多重探索不夸大显著性 |
| 多种 SFT/RL 数据配比 | complete screening + main extremes | 4000 pure-RL 仍是 provisional |
| 仅 SFT / outcome / rule / process reward 消融 | complete at key endpoints | Stage3 macro quality CI 跨 0，弱化显著性表述 |
| 独立 Judge，避免 GPT-4o 循环 | complete | 三个家族均完成；不 post-hoc 选最有利 Judge |
| 更多病理/医学 VLM 基线 | complete | 统一 Omni 11 个本地基线；hosted 模型合同单列 |
| 域外评测与 bad-case | complete for core routes | PathVQA A/B 与原 Yes/No 必须分开 |
| 定量可解释性 | computational part complete | ROI population claim 仍需专家验证 |
| GPT-4o 奖励可信度人工复核 | pending human | 60 例盲包已冻结并可直接评分 |
| 病理专家 ROI 复核 | pending human | 20 例盲包已冻结并可直接评分 |

## 权威数据入口

- 当前表格生成结果：`docs/result_catalog_20260815/paper_tables.md`
- 机器可读核心结果：`protocol/final_unified_results_tables_20260814.json`
- 统一 Omni 基线：`pathvlm_revision_eval_a100/runs/omnimedvqa_per_model_gated_v2_20260815/*/full8518/metrics.json`
- 多 Judge：`docs/20260810_final_closed_benchmark_and_multijudge_results.md`
- 最新可解释性/人工面板：`docs/INTERPRETABILITY_AND_HUMAN_RATING_CASE_SELECTION_20260814.md`
- 专家浏览包：`pathvlm_revision_eval_a100/human_review/expert_review_browse_packets_20260815`
