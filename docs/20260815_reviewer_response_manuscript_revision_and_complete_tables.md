# PathVLM-R1 审稿回复、正文修订稿与完整实验表（2026-08-15）

> 本文件是统一修订工作底稿，包含可直接用于回复信的文字、可粘贴进英文正文的替换段落、失败案例分析和全部关键数据表。数值均来自本机冻结 JSON；人工专家评分尚未完成，相关位置明确保留为待填项。不得把本文件中的计算型结果或外部模型伪标注表述成人工病理结论。

## 1. 当前完成状态与审计结论

2026-08-15 的 fail-closed 审计检查了 85 项论文相关资产，失败项为 0，状态为 `complete_except_human_ratings`。机器可读审计见 `protocol/final_revision_assets_audit_20260815.json`。

已完成的非人工部分包括：完整 PathMMU Test999、主线与数据配比消融、五随机推理种子及置信区间、11 个本地基线的统一 OmniMedVQA 重跑、Stage3 三个 checkpoint 的 PathVQA 自由 Yes/No、Stage2 后继续 1000 rule-RL 的 OmniMedVQA、三种独立 Judge 的盲化生成质量评价、失败案例分析，以及反事实删除、RISE 和 activation patching 等计算型可解释性实验。

仍待人工完成的只有三类评分：

1. 60 例 GPT-4o 过程奖励与病理专家评分一致性；
2. 20 例 pseudo-reference ROI 的病理相关性复核；
3. 冻结 100 例 Stage2/Stage3 匿名 A/B 回答质量比较。

两个科学边界必须保留：`0 SFT + 4000 rule-RL` 仅运行到 checkpoint-2500，虽已覆盖全部 4,000 条训练题至少一次，但未完成原计划 6,000 steps，因此只能称 provisional extreme control；PathMMU Test999 已在工程阶段被访问，不能再称为 untouched final test。

## 2. 给审稿人的综合回复（可直接改写进 response letter）

### 2.1 完整测试、重复推理与统计不确定性

**Response.** We thank the reviewer for requesting complete held-out evaluation and uncertainty estimates. We now report all 999 PathMMU test questions rather than the former 500-case subset. For selected principal stages, we additionally performed five stochastic inference runs with fixed seeds 42–46 under a frozen decoding contract (temperature 0.7, top-p 0.9, top-k disabled, and a 1,024-token generation cap). We report the mean, standard deviation, and Student-t 95% confidence interval across runs, together with paired case-level analyses for key model comparisons. Because the test split was accessed during engineering diagnostics, we no longer describe it as untouched and use validation385—not test999—for checkpoint and recipe selection.

正文应同时加入 Table S1，并删除任何暗示“单次点估计足以证明稳定优势”的表述。五次推理刻画的是解码随机性，不替代训练多种子的可重复性，也不替代逐病例 paired bootstrap。

### 2.2 数据规模与 SFT/RL 配比

**Response.** We added a low-cost allocation screen with a fixed total budget of 1,000 examples: 250 SFT + 750 rule-RL, 500 + 500, and 750 + 250. We also report the principal 3,000 SFT + 1,000 RL route, the 4,000-SFT continuation control, the provisional 0-SFT + 4,000-rule-RL extreme, and continued rule-RL from the historical Stage2 checkpoint. The small-scale allocation screen is explicitly presented as a screening experiment rather than a complete factorial data-scaling law. Its non-monotonic results show that performance depends on sample allocation and task distribution rather than simply increasing the RL proportion.

### 2.3 GPT-4o 同时参与训练与评价的循环性

**Response.** We separated training-time reward construction from final response-quality evaluation. The final blinded 100-case panel was independently scored by GPT-4o, Claude Sonnet 4.6, and Gemini 3.1 Pro without revealing candidate identity. All three judges produced a positive mean Stage3-minus-Stage2 difference, but all confidence intervals included zero; we therefore report a consistent small positive trend rather than a statistically significant superiority claim. A frozen human-review packet is additionally provided for pathology-expert verification. The visual-evidence conclusions do not rely on any LLM judge; they are supported separately by image substitution, controlled deletion/retention, and activation restoration experiments.

### 2.4 0.4 惩罚系数的依据

**Response.** The coefficient 0.4 was selected from the geometry of the predeclared three-event rubric, not by choosing the best validation score after inspection. For either subscale, `max(0, 1 - 0.4k)` for `k=0,1,2,3` produces four distinguishable severity levels: 1.0, 0.6, 0.2, and 0.0. A coefficient of 0.3 leaves a nonzero score of 0.1 even when all three required components fail, whereas 0.5 collapses two and three failures to the same zero score. The 0.4 setting therefore preserves ordinal resolution and reaches zero only at complete subscale failure. In a bounded, matched 100-step single-seed sensitivity pilot, coefficients 0.3, 0.4, and 0.5 achieved 60.78%, 61.56%, and 62.34% on validation385. None of the paired differences was statistically distinguishable from zero. We therefore describe 0.4 as a conservative rubric-based default and report 0.3/0.5 as sensitivity arms; we do not claim that the coefficient is proven insensitive or that 0.5 is inferior.

### 2.5 泛化能力表述收缩

**Response.** We have narrowed all claims of broad generalization. Transfer was heterogeneous across datasets, modalities, and output interfaces. The final Stage3 model improved selected OmniMedVQA subsets, particularly ISIC2020, OCT-C8, and diabetic retinopathy, but did not produce a comparable improvement on Chest CT and remained sensitive to the PathVQA Yes/No response interface. We therefore interpret the results as limited cross-dataset transfer under specified evaluation contracts, not as evidence of a generally applicable pathology foundation model or broad clinical generalization.

### 2.6 失败案例与可解释性

**Response.** We added paired in-domain and out-of-domain failure analyses. In-domain Stage3 gains coexist with regressions in quantitative scale estimation, cell identity, tissue architecture, staining recognition, and nuclear morphology. On PathVQA, the decline contains a large binary decision-interface component: the same Stage3-1500 checkpoint obtains 57.73% under free Yes/No generation and 61.36% when the identical semantic task is expressed as `A=Yes, B=No`. However, controlled case review also identifies genuine fine-grained visual-semantic failures, so the decline cannot be attributed only to parsing or formatting. For interpretability, we replaced the single-attention-map argument with quantitative image substitution, option-conditioned deletion/retention, predefined-region deletion with matched controls, and activation patching. External regions remain pseudo-reference annotations until expert validation.

## 3. 建议替换到英文正文的段落

### 3.1 Abstract/Contributions：替换宽泛的“基础模型与强泛化”表述

建议用下段替换 “remarkable cross-modal transferability”, “foundational model for clinical applications” 等强结论：

> PathVLM-R1 is a 7B pathology-oriented visual-language model trained with a staged SFT, outcome-reward GRPO, and AI-assisted process-reward GRPO pipeline. On the image-disjoint PathMMU split, the staged models improve the in-domain multiple-choice accuracy relative to the base model. Transfer to external datasets is heterogeneous: gains are observed on selected medical-imaging subsets and evaluation contracts, whereas PathVQA binary question answering and Chest CT remain challenging. These findings support the feasibility of sample-efficient, pathology-oriented post-training, but do not establish broad clinical generalization or a universally applicable pathology foundation model.

贡献点建议改为：

> We introduce and audit an AI-assisted cross-modal process reward that explicitly evaluates image-feature use, differential-option reasoning, medical support, and three error events. Its behavior is examined through coefficient sensitivity, independent multi-judge evaluation, and a frozen pathology-expert review protocol.

不要再写 “the first”, “clinically interpretable”, “glass-box partner”, “fundamental pathological principles”, “poised for diverse clinical applications”，除非另有系统文献检索或临床验证支持。

### 3.2 Methods：数据划分与测试边界

> We constructed an image-disjoint formal pool of 5,385 PathMMU records and allocated 3,000 records to SFT, 1,000 to RL, 385 to validation, and 999 to test. No exact image was shared across these four splits. Validation385 was used for checkpoint and recipe selection. Test999 was evaluated in full for reporting and engineering diagnosis; because it was accessed during model development, we do not characterize it as an untouched final test set.

### 3.3 Methods：奖励机制完整说明

> Stage3 used three reward components: final-answer accuracy, output-format compliance, and an online cross-modal process reward. For each sampled trajectory, the evaluator received the pathology image, question, reference answer, and candidate completion. The reference answer was supplied only as context for checking medical correctness; the evaluator was explicitly instructed not to award process credit merely because the final answer matched the reference. The evaluator returned six schema-constrained Boolean events: presence of image-feature analysis, option elimination, and medical-knowledge support; and presence of a histological-definition error, logical contradiction, or outdated/incorrect pathology criterion.
>
> Let `m` be the number of missing positive-integrity events and `e` the number of present error events. We computed `S_integrity=max(0,1-0.4m)`, `S_knowledge=max(0,1-0.4e)`, and `R_process=(S_integrity+S_knowledge)/2`. Stage3 used the accuracy, format, and process components as separate GRPO reward functions. The 0.4 coefficient produces the levels 1.0/0.6/0.2/0.0 for zero to three failures, preserving severity resolution while assigning zero only to complete failure.
>
> The evaluator was `gpt-4o-2024-08-06`, accessed through the AIGCBest OpenAI-compatible endpoint at temperature 0 and seed 42 using strict JSON Schema. The initial response cap was 320 tokens, with bounded validation retries at 512 and 768 tokens. Transient transport errors used delays of 15, 45, and 90 seconds. The formal n=8 run made 12,000 successful logical judgments in 12,034 physical HTTP attempts; 34 ambiguous transport failures were conservatively billed and then successfully retried. No rule fallback was used. The recorded cost was USD 62.4368 at USD 2.50/M input tokens and USD 10.00/M output tokens. The ledger ended with no unresolved reservations.

训练时 Judge 的实际 system prompt 应作为补充材料原样公开：

```text
You are a pathology reasoning-process auditor.
Evaluate only whether the candidate reasoning is a medically sound, image-grounded process for
the supplied multiple-choice pathology question. The reference answer is context for checking
medical correctness; never award process credit merely because the candidate's final answer
matches it.

Return the six required booleans and one short evidence span for each event. For the first three
events, true means the component is genuinely present in the candidate reasoning. Option
elimination may be explicit or implicit and no fixed number of options must be eliminated. For
the last three events, true means the named error is present. Evidence must quote or tightly
point to the candidate reasoning; use an empty string when no supporting span exists. Keep each
evidence span within 160 characters whenever possible; the absolute schema ceiling is 512
characters. Do not add fields, scores, commentary, or a corrected answer.
```

用户消息由以下三段文字和原图组成：`QUESTION`、`REFERENCE ANSWER (context only; do not infer process quality from answer match)`、`CANDIDATE COMPLETION TO AUDIT`。

### 3.4 Results：域内主结果与不确定性

> On PathMMU Test999, the selected full rule-RL n=8 parent achieved 63.46%. Subsequent GPT-4o process-reward Stage3 checkpoints at steps 500, 1,000, and 1,500 achieved 64.36%, 63.96%, and 63.86%, respectively. Relative to the exact Stage3 parent, the paired wrong-to-right/right-to-wrong counts were 72/63, 63/58, and 66/62; none was statistically significant by exact McNemar testing (`p=0.491`, `0.716`, and `0.791`). Thus, Stage3 changed decisions and produced a small positive point estimate, but the current accuracy evidence does not establish a significant in-domain improvement over its immediate parent. Independent response-quality judges showed directionally positive mean differences relative to historical Stage2, with uncertainty intervals crossing zero.

### 3.5 Results：外部迁移与接口敏感性

> External performance varied substantially by task and interface. Stage3-1500 obtained 75.01% contract-aligned accuracy on the unified OmniMedVQA aggregate, with 46.04% on Chest CT, 77.78% on ISIC2020, 77.71% on retinal OCT, and 79.86% on diabetic retinopathy. On the full 3,362-question PathVQA binary set, the same checkpoint achieved 57.73% under free Yes/No generation and 61.36% when the identical semantic decision was encoded as `A=Yes, B=No`. The latter is an interface diagnostic rather than a replacement primary metric. These results indicate heterogeneous transfer and material output-interface sensitivity rather than uniform cross-modal generalization.

### 3.6 Results：可解释性替换段落

> We do not treat a single attention heatmap as proof of interpretability. We instead evaluated whether predictions changed under controlled visual interventions and whether clean internal representations causally restored corrupted decisions. On a frozen 96-case panel, appearance-matched wrong-image substitution reduced the Stage3 correct-option margin by 0.451 (95% CI [0.215, 0.710]) and accuracy by 10.4 percentage points. Option-conditioned RISE regions produced stronger deletion and retention effects than shape-matched random regions. A separate predefined evidence-region funnel yielded 20 strict cases, but the aggregate reference-minus-random deletion difference was 0.109 with a 95% CI of [-0.065, 0.299]; therefore, these external regions remain pseudo-reference regions pending pathology-expert review. Activation restoration provides evidence of model-internal visual-to-language causal propagation, but it does not by itself validate the pathological correctness of the selected region.

### 3.7 Discussion/Conclusion：替换结论

> The experiments support a narrower conclusion than the original manuscript. Staged post-training can improve a pathology-oriented multiple-choice model with limited task-specific data, and the final model demonstrably uses image-specific evidence in a subset of controlled cases. However, external transfer is heterogeneous, output-interface calibration materially affects PathVQA, and neither automatic reasoning-quality scores nor pseudo-reference regions substitute for expert clinical validation. The present model should therefore be viewed as a research prototype for pathology-oriented visual question answering rather than a validated clinical foundation model or a generally reliable diagnostic system.

### 3.8 Related Work：UNI、CONCH 与 PathChat 的定位边界

> UNI and CONCH are representative pathology foundation encoders developed for transferable visual representation and image-text alignment, respectively, whereas PathChat targets pathology-specific interactive visual-language assistance. Our method addresses a different question: staged post-training of an instruction-following VLM for image-conditioned multiple-choice reasoning with outcome and process rewards. For the available frozen encoders, we report PLIP/CONCH image-to-candidate-text matching as a restricted auxiliary comparison; these scores do not measure generated reasoning. UNI has no native question-answering interface and therefore is not assigned a zero-shot VQA accuracy. We did not identify a verified, reproducible official end-to-end PathChat release for direct execution in the frozen environment, so PathChat is discussed by task scope, supervision, capabilities, clinical intent, and limitations rather than represented by an unverifiable numerical cell. These systems are consequently complementary reference points, not strictly interchangeable baselines under one evaluation contract.

## 4. 失败案例分析

### 4.1 域内 PathMMU：真实改善与真实回退并存

当前 Stage3-1500 相对其实际 n=8 rule-RL parent：634/999 → 638/999，仅净增 4 题；66 题错转对、62 题对转错，McNemar `p=0.7910`。因此不能用“所有阶段持续显著提高”描述这段训练。

典型回退：

| Index | 错误类型 | Parent → Stage3 | 具体表现 |
|---:|---|---|---|
| 4 | 尺度/数量估计 | B→A，gold B | 25 μm 标尺下中性粒细胞数量由约10个误估为5个。 |
| 31 | 细胞身份 | B→D，gold B | 将真皮内色素负载巨噬细胞误判为黑素细胞。 |
| 70 | 染色识别 | A→C，gold A | 将 May–Grünwald–Giemsa 误判为 H&E。 |
| 92 | 核形态 | B→A，gold B | 将 hyperchromasia 与 anisokaryosis 混淆。 |

典型改善：

| Index | 修复类型 | Parent → Stage3 | 具体表现 |
|---:|---|---|---|
| 32 | 组织层次 | C→D，gold D | 正确识别 Pagetoid spread 遍及上皮全层。 |
| 75 | 炎症细胞 | A→D，gold D | 将嗜酸细胞错误修正为淋巴细胞为主。 |
| 109 | 肾小球结构 | B→A，gold A | 修复 mesangial expansion 判断。 |
| 36 | 染色与定位 | B→C，gold C | 修复棕色染色区域的病理含义。 |

机制解释：Stage3 的过程奖励能修复部分视觉特征、医学知识和选项映射错误，但 outcome/process reward 仍是稀疏且有噪声的轨迹级信号；训练会改变不少局部决策，却没有形成足够大的净泛化增益。该结果更符合 `decision churn with small positive point estimate`，而不是“模型完全没有学习”或“全面提升”。

### 4.2 域外 PathVQA：接口漂移是主因之一，但不是全部

Stage3 三个 checkpoint 的完整自由 Yes/No 结果为 54.43%、57.85%、57.73%；相同语义改成固定 `A=Yes/B=No` 后为 60.44%、62.28%、61.36%。以 Stage3-1500 为例：

- 自由 Yes/No 错而 A/B 对：355 题；
- 自由 Yes/No 对而 A/B 错：233 题；
- A/B 净恢复 122 题（+3.63 pp）；
- 自由生成预测 Yes 率仅 24.12%，A/B 接口预测 Yes 率为 38.46%，而目标 Yes 占 54.02%。

多条错误显示推理语义与最终答案直接矛盾。例如 source index 10 的推理明确写出病灶“begins as a focus of microabscess ... then expands”，但自由接口结尾输出 `<answer>No</answer>`；A/B 接口在同题输出 `A) Yes` 并得分。indices 12、14、24、26、36 也出现正文肯定图中征象、最终自由答案却为 No 的现象。这不是截断：Stage3-1500 只有 2/3362 触及 2048-token cap，空输出为 0。

另一方面，不能把全部域外损失归为格式问题。Base→L-SFT3000 的 forced-binary 分析中，220 题由对转错、90 题由错转对，净损失 130；其中 173/220 回退来自 `is X present?` 和 `does this image show X?` 两类模板，显示二元决策边界压缩，但仍存在明确的视觉语义回退：

- source 3079：软骨肉瘤包埋原有板层骨的 cartilage/bone interface 判断错误；
- source 5668：表层异型、显著多形性和有丝分裂判断错误；
- source 1725：低对比铁染色判断错误；
- source 1623：图中仍有甲状腺滤泡与淋巴滤泡，却将 Hashimoto thyroiditis 从 Yes 改为 No。

同时也有真实改善，如 source 14 的 AIDS-associated *M. avium*、source 827 的冠状动脉粥样硬化伴急性血栓、source 6547 的横行溃疡/狭窄和 source 4447 的嗜碱性颗粒沉积。正确结论是：窄域训练重新分配了能力，既修复部分病理识别，也造成更多二元接口和细粒度视觉语义错误，而不是“病理知识整体消失”。

### 4.3 OmniMedVQA：按子域解释，不报一个笼统的“域外泛化”

OmniMedVQA 的四个来源是 CT、皮肤镜、视网膜 OCT 和糖网，并不等同于独立病理切片基准。当前 Stage3 的提升主要来自 ISIC/OCT/DR，Chest CT 没有同步提高。Stage2 后继续 1000 rule-RL 的统一结果为 54.29%，其中 CT 36.05%、ISIC 78.10%、OCT 48.43%、DR 55.14%，进一步说明一个总体均值会遮蔽模态间相反变化。正文必须报告分源结果。

## 5. 奖励系数 0.3/0.4/0.5 敏感性及交叉表

### 5.1 验证集结果

| Penalty | Steps | Judge calls | Cost | Val correct | Val accuracy |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 100 | 800 | $4.1221 | 234/385 | 60.7792% |
| 0.4 | 100 | 800 logical / 800 physical | $4.1308 | 237/385 | 61.5584% |
| 0.5 fresh | 100 | 800 logical / 803 physical | $4.1278 | 240/385 | 62.3377% |

| Comparison | Wrong→right | Right→wrong | Net | Δ accuracy | Exact p | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| 0.3→0.4 | 11 | 8 | +3 | +0.7792 pp | 0.6476 | [-1.2987, 3.1169] |
| 0.4→0.5 | 14 | 11 | +3 | +0.7792 pp | 0.6900 | [-1.8182, 3.3766] |
| 0.3→0.5 | 15 | 9 | +6 | +1.5584 pp | 0.3075 | [-0.7792, 4.1558] |

应写成“bounded pilot 中未检测到统计可区分的差异”，不能写成“已经证明完全不敏感”。

### 5.2 800 个训练 rollout 的交叉行为

| Measure | 0.3 | 0.4 | 0.5 |
|---|---:|---:|---:|
| Answer accuracy | 68.625% | **69.000%** | 68.375% |
| Format accuracy | 100.000% | 100.000% | 99.875% |
| Image-feature event | 94.375% | **94.994%** | 93.875% |
| Option-elimination event | **78.500%** | 78.223% | 77.125% |
| Medical-knowledge support | 94.125% | 93.617% | **94.250%** |
| Histological-definition error | 35.125% | **34.293%** | 37.250% |
| Logical contradiction | 5.500% | 4.631% | **3.750%** |
| Incorrect/outdated criterion | **5.250%** | 6.383% | 5.625% |

三模型在 351/385 题上正确性一致，仅 34 题不同；0.3/0.4/0.5 正确性模式为 `001:7, 011:8, 010:3, 110:8, 101:7, 100:1`。0.5 减少显式矛盾，但增加 histological-definition error；0.4 的训练答案准确率最高且 histological-definition error 最低。该交叉表支持 0.4 是保守折中，不支持某一系数全面优越。

## 6. 修订数据大表

### Table I. Image-disjoint data allocation

| Item | Count | Reporting boundary |
|---|---:|---|
| PathMMU total images | 24,067 | 原始数据集规模 |
| PathMMU total Q&A | 33,428 | 原始数据集规模 |
| Formal image-disjoint pool | 5,385 | PubMed + EduContent rewritten pool |
| SFT | 3,000 | 与 RL/validation/test 图像隔离 |
| RL | 1,000 | 与 SFT/validation/test 图像隔离 |
| Validation | 385 | checkpoint/recipe 选择 |
| Test | 999 | 完整报告；已用于工程诊断 |

### Table II-A. Main staged models under current contracts

百分数。`Yes/No` 为真实自由生成；`A/B` 是补充接口诊断；`forced` 只能作机制诊断。

| Model/stage | Val385 | Test999 | PathVQA free Yes/No | PathVQA A/B | Omni aligned | Omni official | MMMU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen2.5-VL-7B | 50.91 | 48.65 | 66.42 (legacy generated) | 60.56 | 59.00 | 57.38 | 56.03 |
| Historical full-SFT3000 | 59.74 | 58.26 | 51.40 (legacy generated) | 56.25 | 50.97 | 48.45 | 43.10 |
| Historical Stage2 outcome-RL | 61.04 | 60.96 | 49.64 (legacy generated) | 55.59 | 52.78 | 49.20 | 47.41 |
| L-r16 SFT3000 step80 | 54.55 | 56.66 | 62.52 (forced diagnostic) | 61.99 | 61.66 | 57.42 | 49.14 |
| Full rule-RL n4 step1000 | 59.74 | 61.16 | NA | 62.61 | 69.70 | 57.20 | 53.45 |
| Full rule-RL n8 step1000 | 64.16 | 63.46 | NA | 62.61 | 69.50 | 56.80 | 54.31 |
| GPT-4o Stage3 step500 | 65.71 | **64.36** | 54.43 | 60.44 | 74.34 | 54.16 | 58.62 |
| GPT-4o Stage3 step1000 | **66.75** | 63.96 | **57.85** | **62.28** | 73.76 | 55.87 | **60.34** |
| GPT-4o Stage3 step1500 | 65.19 | 63.86 | 57.73 | 61.36 | **75.01** | 57.18 | 54.31 |
| LoRA-SFT4000 control | 52.99 | 56.56 | NA | **63.12** | 59.97 | 57.60 | 56.03 |
| Stage2 + continued rule-RL1000 | 59.74 | 61.86 | 59.31 | 56.19 | 54.29 | 51.82 | NA |
| 0-SFT + rule-RL4000 ckpt2500 | 63.38 | 63.76 | 65.20 | 62.85 | 67.59 | NA | NA |

`legacy generated` 与 2026-08-15 的 `domain_think_answer_v5_2048` 不是完全相同 prompt 版本，应按合同分层解释，不能直接作纯模型差异归因。

### Table II-B. External baselines

| Model | PathMMU Test999 | PathVQA free Yes/No | Omni aligned | Omni official |
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
| PLIP (image-text matching) | 33.43 | 48.30 statement matching | 23.78 option matching | NA |
| CONCH (image-text matching) | 33.83 | 53.87 statement matching | 38.71 option matching | NA |
| UNI (vision encoder) | NA | NA | NA | NA |
| PathChat (no verified runnable release) | NA | NA | NA | NA |

这里的 `NA` 不是漏测或按零分处理。PLIP/CONCH 不能生成推理：PathMMU/Omni 将图像与候选项文本作余弦匹配；PathVQA 在 3,362 道 Yes/No test 上比较 `Answer: Yes. Question: {question}` 与 `Answer: No. Question: {question}` 两条陈述。答案词置前以避免 CLIP 上下文截断，但该值仍是探索性 retrieval-style statement matching，不能与生成式 VQA 按同一合同排序。PLIP 的 accuracy/balanced accuracy 为 48.30%/48.94%，CONCH 为 53.87%/52.53%。UNI 是视觉表征编码器，没有文本问答接口；PathChat 尚未核验到可复现的官方端到端权重与推理发布。后二者按审稿意见在 Related Work/Discussion 中作任务范围、训练数据、监督方式、能力和临床场景的定位比较，而不伪造 accuracy。

### Table III. Blind multi-Judge response quality

分数范围 `[0,1]`，冻结 100 例、每个 Judge 960 次，共 2,880 个判断。

| Candidate | GPT-4o | Claude Sonnet 4.6 | Gemini 3.1 Pro | Three-judge mean |
|---|---:|---:|---:|---:|
| Base | 0.7922 | 0.5987 | 0.6335 | 0.6748 |
| SFT3000 | 0.7717 | 0.5939 | 0.6213 | 0.6623 |
| SFT4000 | 0.7634 | 0.5873 | 0.6177 | 0.6561 |
| Stage2 | 0.7804 | 0.5952 | 0.6126 | 0.6628 |
| Stage3 GPT-4o | 0.7937 | 0.6123 | 0.6209 | 0.6756 |
| Stage3 Grok-4.3 | **0.7993** | **0.6268** | 0.6223 | **0.6828** |

| Judge | GPT-4o Stage3 − Stage2 | Grok Stage3 − Stage2 |
|---|---:|---:|
| GPT-4o | +0.0132, 95% CI [-0.0262, +0.0526] | +0.0189 [-0.0207, +0.0576] |
| Claude | +0.0171 [-0.0227, +0.0555] | +0.0315 [-0.0107, +0.0726] |
| Gemini | +0.0082 [-0.0105, +0.0276] | +0.0096 [-0.0120, +0.0313] |

三个 GPT-4o Stage3 点估计均高于 Stage2，但 CI 均跨 0。结论是“一致的小幅正向趋势”，不是“显著优于”。人工盲评仍需完成。

### Table IV. Unified OmniMedVQA source-wise results

合同为 `omnimed_domain_think_answer_v4_1024`；主指标是 target-blind contract-aligned accuracy。

| Model | Chest CT | ISIC2020 | OCT-C8 | DR | Micro aligned | Official |
|---|---:|---:|---:|---:|---:|---:|
| Base | 49.14 | 45.95 | 61.11 | 69.14 | 59.00 | 57.38 |
| Historical full-SFT3000 | 36.74 | 71.08 | 45.59 | 52.07 | 50.97 | 48.45 |
| Historical Stage2 | 36.05 | 72.97 | 47.63 | 54.41 | 52.78 | 49.20 |
| L-r16 SFT3000 | 44.89 | 60.76 | 64.67 | 63.58 | 61.66 | 57.42 |
| Full rule-RL n4 | 47.07 | 68.54 | 71.41 | 76.84 | 69.70 | 57.20 |
| Full rule-RL n8 | 47.42 | 71.01 | 71.17 | 74.45 | 69.50 | 56.80 |
| Stage3-500 | 49.25 | 78.67 | 75.45 | 79.47 | 74.34 | 54.16 |
| Stage3-1000 | 47.42 | 74.18 | 76.59 | 79.08 | 73.76 | 55.87 |
| Stage3-1500 | 46.04 | 77.78 | 77.71 | 79.86 | **75.01** | 57.18 |
| LoRA-SFT4000 | 43.17 | 51.33 | 67.38 | 59.24 | 59.97 | 57.60 |
| Stage2 + continued rule-RL1000 | 36.05 | 78.10 | 48.43 | 55.14 | 54.29 | 51.82 |
| Qwen2.5-VL-3B | 45.46 | 59.56 | 67.03 | 54.12 | 60.33 | 52.42 |
| Lingshu-7B | 47.42 | 89.05 | 80.03 | 76.69 | **77.57** | 51.53 |
| MedVLM-R1 | 43.40 | 50.44 | 72.31 | 70.60 | 64.89 | 43.26 |
| MedGemma-4B | 43.28 | 81.08 | 76.42 | 83.08 | 75.50 | 50.52 |
| ScaleReasoner-R1 | 50.40 | 56.08 | 66.36 | 72.65 | 64.33 | 39.86 |
| Llama-3.2-Vision-11B | 32.03 | 72.66 | 71.74 | 43.25 | 60.99 | 31.85 |
| HuatuoGPT-Vision-7B | 43.40 | 38.67 | 75.72 | 68.41 | 63.78 | 57.27 |
| InternVL3-8B | 47.30 | 49.62 | **81.42** | 66.80 | 68.51 | **61.22** |
| DeepSeek-VL2 | 32.72 | 60.82 | 52.07 | 55.49 | 52.54 | 9.46 |
| Llama-3.2-Vision-90B | 40.99 | 82.78 | 78.91 | 55.73 | 70.17 | 39.00 |
| LLaVA-Med-v1.5 | 28.47 | 57.22 | 43.68 | 38.66 | 43.43 | 41.11 |
| PLIP (matching only) | 26.29 | 38.61 | 15.31 | 27.89 | 23.78 | NA |
| CONCH (matching only) | 35.82 | 92.85 | 10.68 | 53.10 | 38.71 | NA |

PLIP/CONCH 两行使用 `image_to_raw_option_text_cosine_similarity`，不是上方生成式 `omnimed_domain_think_answer_v4_1024`。它们用于补充病理图文编码器对照，不能据此声称生成推理能力；UNI 和 PathChat 没有兼容的逐题 VQA 输出。

### Table V-A. Fixed 1,000-example SFT/RL allocation screen

| SFT | Rule-RL | Val385 | Test999 | PathVQA A/B | Omni aligned | MMMU |
|---:|---:|---:|---:|---:|---:|---:|
| 250 | 750 | 50.39 | 47.25 | **57.76** | 58.03 | **54.31** |
| 500 | 500 | **51.95** | **51.75** | 57.20 | 57.75 | 45.69 |
| 750 | 250 | 50.91 | 49.75 | 55.62 | **60.00** | 43.97 |

### Table V-B. Mechanism and continuation controls

| Arm | Status | Val385 | Test999 | PathVQA free | PathVQA A/B | Omni aligned | MMMU |
|---|---|---:|---:|---:|---:|---:|---:|
| LoRA-SFT4000 | complete | 52.99 | 56.56 | NA | 63.12 | 59.97 | 56.03 |
| 3000 LoRA-SFT + full rule-RL n4 | complete | 59.74 | 61.16 | NA | 62.61 | 69.70 | 53.45 |
| 3000 LoRA-SFT + full rule-RL n8 | complete | 64.16 | 63.46 | NA | 62.61 | 69.50 | 54.31 |
| Stage2 + continued rule-RL1000 | complete except MMMU | 59.74 | 61.86 | 59.31 | 56.19 | 54.29 | NA |
| 0-SFT + rule-RL4000 ckpt2500 | provisional | 63.38 | 63.76 | 65.20 | 62.85 | 67.59 | NA |

LoRA-SFT4000 对比 LoRA-SFT3000 的 Test999 基本没有提高，而后续 full rule-RL n4/n8 相对 LoRA-SFT4000 分别提高 +4.60/+6.91 pp；paired bootstrap CI 为 `[+2.06,+7.20]` 与 `[+4.15,+9.65]`，McNemar `p=0.000860` 与 `1.46e-6`。这支持“RL 增益不只是继续 SFT exposure”，但不能单独证明过程奖励优于所有规则奖励。

### Table S1. Five stochastic inference runs

| Model | Mean | SD | Student-t 95% CI |
|---|---:|---:|---:|
| Base | 49.31 | 0.48 | [48.71, 49.91] |
| L-r16 SFT80 | 52.65 | 1.09 | [51.30, 54.01] |
| Historical Stage2 | 58.72 | 0.89 | [57.61, 59.83] |
| Stage2 + continued rule-RL1000 | 60.92 | 0.72 | [60.02, 61.82] |
| Full rule-RL n8 | 61.16 | 1.05 | [59.85, 62.47] |
| Stage3-500 | 63.66 | 0.86 | [62.60, 64.73] |
| Stage3-1000 | 63.62 | 0.98 | [62.41, 64.84] |
| Stage3-1500 | 63.10 | 0.81 | [62.10, 64.11] |
| Rule-RL4000 ckpt2500 provisional | 60.26 | 1.53 | [58.37, 62.16] |

### Table S2. Quantitative interpretability

| Statistic | Result |
|---|---:|
| Predefined candidates | 160 |
| Dual-external-model parseable | 120/123 |
| Dual-model ROI consensus | 94 |
| Target clean-correct | 51/94 |
| Clean-correct and generation/score agreement | 47 |
| Positive reference deletion margin drop | 34 |
| Strict effective cases | 20 (A5/B7/C3/D5) |
| Reference deletion mean margin drop | 0.4542, 95% CI [0.1520, 0.8079] |
| Random matched deletion | 0.3452 [0.1550, 0.5744] |
| Neighbor matched deletion | 0.2456 [0.0661, 0.4520] |
| Reference − random extra drop | 0.1090 [-0.0648, 0.2989] |

20 例是满足严格筛选门槛的强个例集合；`reference-random` 群体差异 CI 跨 0，不能宣称已证实总体 ROI 显著优于随机。pathologist review 完成前统一使用 `predefined pseudo-reference evidence region`。

## 7. 人工复核回填计划

### 7.1 GPT-4o 奖励可信度（60例）

两名病理评审独立评分，A 评全部 60 例，B 至少重叠 30 例。保持 Judge、训练阶段、自动分数和抽样层完全盲化。分别报告代表性 36 例和 challenge 24 例，计算事件级准确率/F1、过程分与专家分 Spearman 相关、加权 kappa 或 ICC，以及 Bland–Altman/绝对误差。不能只报告合并相关系数。

### 7.2 Stage2/Stage3 生成质量（100例）

使用输出无关冻结 panel，随机交换 A/B 位置，隐藏模型身份。主指标为 A/B/Tie 偏好，次指标为医学事实、视觉依据和推理质量各 1–5 分。两人独立评分，报告一致率/Cohen's kappa；第三人只仲裁分歧。错转对案例包只作定性示例，不能用于总体偏好率。

### 7.3 可解释性 ROI（20例）

评审只看原图、问题、选项和候选区域，不看 attention、RISE、模型预测、删除效果或 activation patching。逐区域评价“是否包含诊断相关证据”和“是否覆盖主要证据”，并记录遗漏区域。专家确认后才能把 pseudo-reference 改称 `pathologist-validated diagnostically relevant region`；即便如此，也不能自动称为 causal ground truth，因图中可能有冗余证据。

### 7.4 外部参考辅助不替代独立人工评分

三项专家任务采用两阶段流程。第一阶段完全不显示外部模型意见并冻结人工评分；第二阶段才显示独立外部参考，并记录是否改判及参考帮助度。奖励复核使用 Claude Sonnet 4.6（不读取训练期 GPT-4o reward），ROI 使用 Gemini 3.1 Pro 与 Claude Opus 5，生成质量使用 Claude Sonnet 4.6 与 Gemini 3.1 Pro。主要人工结果、专家一致性和 GPT-4o 奖励一致性均以第一阶段为准；第二阶段只用于辅助、仲裁和敏感性报告，不能称为独立专家真值。

为降低专家逐题浏览成本，材料另提供 reference-assisted 入口：每题直接并列展示匿名病例内容与外部意见，并有上一题/下一题导航；奖励页固定显示与训练一致的六事件及 0.4 量表公式。该入口仍对候选模型身份盲化，但不对外部参考盲化，因此其评分只作为辅助/仲裁结果，不能取代上述独立盲评主统计。

该辅助入口现以中英文对照展示评分规则、题干、选项、参考答案、匿名待评回答、外部模型理由和 ROI 特征；英文原文继续保留为权威文本，翻译只用于辅助理解。冻结 60 例上，Claude Sonnet 4.6 与训练期 GPT-4o 的 360 个六事件判断 micro agreement 为 79.72%（病例聚类 bootstrap 95% CI 75.28%–83.89%），Cohen's kappa 为 0.566；换算后过程奖励精确相同率为 38.33%，85.00% 的病例差值不超过 0.2，Spearman 相关为 0.522（95% CI 0.308–0.698）。这属于中等机器—机器一致性，不能替代尚待完成的病理专家验证；challenge-enriched 面板也不能当作总体随机样本。

## 8. 结果与材料入口

- 最新审计：`protocol/final_revision_assets_audit_20260815.json`
- 结果追溯：`docs/result_catalog_20260815/result_lineage.json`
- 机器生成表：`docs/result_catalog_20260815/paper_tables.md`
- 病理基础模型：CONCH PathMMU `pathvlm_revision_eval_a100/runs/pathmmu_diagnostic/test999_matching_20260728_223308/conch/{run_config.json,metrics.json,predictions.jsonl}`；CONCH Omni `pathvlm_revision_eval_a100/runs/external_vqa_full_20260729_224503/conch/omnimedvqa/{run_config.json,metrics.json,predictions.jsonl}`；PLIP 位于同级 `plip` 目录；UNI/PathChat 的不可比性说明见本文件 Table II-B。
- PLIP/CONCH PathVQA 问题-答案陈述匹配：`pathvlm_revision_eval_a100/runs/pathvqa_statement_matching_20260815/{plip_full_yesno3362,conch_full_yesno3362}/{run_config.json,metrics.json,predictions.jsonl}`
- Stage3 自由 Yes/No：`pathvlm_revision_eval_a100/runs/missing_paper_evaluations_20260815/stage3_gpt4o_checkpoint*/.../metrics.json`
- Stage2 继续 RL Omni：`pathvlm_revision_eval_a100/runs/missing_paper_evaluations_20260815/stage2_continued_rule_rl1000_omnimedvqa_v4/full8518/metrics.json`

### 8.1 Stage2/Stage3 生成质量：原始回答与机器 Judge

- Stage2 原始 Test999 生成：`pathvlm_revision_eval_a100/runs/pathmmu_sft4000_stage2_20260729_221640/stage2_rl_test999/predictions.jsonl`
- 现有多 Judge 使用的 GPT-4o Stage3 原始生成：`pathvlm_revision_eval_a100/runs/stage3_selected_gpt4o_pathmmu_pathvqa_20260805_parallel_grok43/stage3_gpt4o/pathmmu_test999/predictions.jsonl`
- 当前 n=8 GPT-4o Stage3 step1500 原始生成：`pathvlm_revision_eval_a100/runs/stage3_gpt4o_n8_checkpoint1500_final_eval_20260813/pathmmu_test999/predictions.jsonl`
- 输出无关冻结 100 例 panel：`pathvlm_revision_eval_a100/runs/multijudge_reasoning_eval_20260810/frozen_panel.json`
- 三 Judge 最终汇总、逐指标均值、配对 bootstrap CI：`pathvlm_revision_eval_a100/runs/multijudge_reasoning_eval_20260810/final_six_models_stage3gpt.json`
- GPT-4o 逐条 Judge 输出：`pathvlm_revision_eval_a100/runs/multijudge_reasoning_eval_20260810/formal_gpt4o/judgments.jsonl`
- Claude Sonnet 4.6 逐条 Judge 输出：`pathvlm_revision_eval_a100/runs/multijudge_reasoning_eval_20260810/formal_claude/judgments.jsonl`
- Gemini 3.1 Pro 逐条 Judge 输出：`pathvlm_revision_eval_a100/runs/multijudge_reasoning_eval_20260810/formal_gemini/judgments.jsonl`
- 每位 Judge 的完整性摘要：对应 `formal_gpt4o/full_summary.json`、`formal_claude/full_summary.json`、`formal_gemini/full_summary.json`

### 8.2 人工复核材料

- 可直接交专家的盲化包（v2，内含详细说明）：`backup_archives/human_review_distribution_20260815_v2/pathvlm_human_review_reviewer_blinded_20260815.tar.gz`
- 课题负责人留存的完整包（v2，含映射/答案键，禁止发给评审）：`backup_archives/human_review_distribution_20260815_v2/pathvlm_human_review_owner_complete_20260815.tar.gz`
- 奖励 60 例与 ROI20 浏览源：`pathvlm_revision_eval_a100/human_review/expert_review_browse_packets_20260815`
- Stage2/Stage3 盲化 100 例、改善/退化案例：`pathvlm_revision_eval_a100/runs/stage2_stage3_human_review_packet_20260815`
- 逐步使用手册：`myr1/docs/20260815_human_review_packet_instructions.md`
- 两阶段本机入口：`pathvlm_revision_eval_a100/human_review/two_pass_expert_review_20260815/index.html`；第二阶段参考必须在第一阶段评分保存后查看。
- 中英文辅助入口：上述本机入口中的 `assisted_review_with_external_references/index.html`；辅助结果必须标为 `reference-assisted expert review`，不可混入独立盲评主统计。
- GPT-4o/Claude 奖励一致性：`docs/20260815_gpt4o_claude_reward_reference_agreement.md` 与 `pathvlm_revision_eval_a100/human_review/external_reference_reward_review_20260815/gpt4o_claude_agreement.json`。

现有盲化 100 例与已完成的三模型 Judge 使用同一个 `Stage3-GPT4o-selected`，保证人机逐题对齐；当前 n=8 step1500 目前只有按结果分层的改善/退化案例包，不能用来估计总体人工偏好。若正文最终以 n=8 step1500 作为唯一 Stage3 主模型并要求相应总体人工偏好，需要另建一份使用同一输出无关 panel 的 n=8 盲化 A/B 包。

## 9. 远端备份状态

论文修订增量已上传至 HF dataset `Freddie1946/PathVLM-R1-Migration-Archive-20260815` 的
`increments/20260815_final_revision_closure_v1`。该仓库为 public metadata + manual-gated
content；本次首轮远端校验 revision 为
`9527a11a19cf9024f02eba7d40bfe1eee4cd37c6`，共 40 个文件、417,056,604 bytes。内容包括：

- 本合并修订稿和人工复核说明；
- 85 项完整性审计；
- Stage3 三个 checkpoint 的自由 Yes/No 逐题 JSON；
- Stage2 后继续 rule-RL1000 的 OmniMedVQA 逐题 JSON；
- 结果 lineage/catalog；
- 270 MB 所有者完整人工包与 106 MB 盲化评审包。

当前 Codex 会话另以去密钥在线一致性快照保存到 private dataset
`Freddie1946/PathVLM-R1-Codex-Private-Snapshots`，tag 为 `20260815T062722Z`，远端 revision
为 `d53f7a7bc30ccc2a67cc08dbd60d34b02d02b350`。快照归档大小为 174,462,207 bytes，明确排除
`auth.json`、缓存和凭据；迁移后需要重新认证。

本次补齐的 UNI/CONCH/PathChat/PLIP 表格、Stage2/Stage3 生成质量入口和自包含人工复核 v2
包位于同一 gated dataset 的 `increments/20260815_results_material_index_v2`；上传校验记录为
`protocol/results_material_index_v2_hf_backup_20260815.json`。
