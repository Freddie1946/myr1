# Stage3 Judge、人类复核与视觉证据结论边界

## 多模型 Judge（冻结100例，六维宏平均）

每个 Judge 独立看到图像、问题、参考答案和匿名候选回答；分数范围均为 0–1。

| 模型 | Claude Sonnet 4.6 | Gemini 3.1 Pro | GPT-4o | 三 Judge 均值 |
|---|---:|---:|---:|---:|
| Base Qwen2.5-VL-7B | 0.5987 | 0.6335 | 0.7922 | 0.6748 |
| SFT3000 | 0.5939 | 0.6213 | 0.7717 | 0.6623 |
| SFT4000-control | 0.5873 | 0.6177 | 0.7634 | 0.6561 |
| Stage2 outcome-GRPO | 0.5952 | 0.6126 | 0.7804 | 0.6628 |
| Stage3 GPT-4o selected | 0.6123 | 0.6209 | 0.7937 | 0.6756 |
| Stage3 Grok-4.3 selected | 0.6268 | 0.6223 | 0.7993 | 0.6828 |

Stage3 GPT-4o 相对 Stage2 的六维宏平均配对差：

| Judge | 平均差（Stage3 − Stage2） | 95% bootstrap CI |
|---|---:|---:|
| Claude Sonnet 4.6 | +0.0171 | [-0.0227, +0.0555] |
| Gemini 3.1 Pro | +0.0082 | [-0.0105, +0.0276] |
| GPT-4o | +0.0133 | [-0.0254, +0.0517] |

三个点估计均为正，但区间均包含 0。因此当前100例支持“一致正向趋势”，不支持“Stage3 的回答质量已被证明显著高于 Stage2”。该 Judge 结果用于回答生成质量及减少单一 GPT-4o 评价的同源性疑虑，不用于证明视觉依赖。

## 人工比较设计

人工专家可以、也适合直接判断哪份回答更好。推荐使用与多 Judge 完全相同、在观察模型输出之前冻结的100例，随机交换 Stage2/Stage3 的 A/B 位置，隐藏模型身份。主指标是 A/B/Tie 偏好；次指标是医学事实、视觉依据和推理质量（各1–5分）。建议两名评审独立评分，并报告一致率/Cohen's kappa；分歧项由第三人仲裁。

另行提供的“Stage2错→Stage3对”材料只能用于定性分析改善案例，不得用于估计总体人工偏好率。

## 当前可辩护的视觉证据结论

现有结果足以支持较窄但强的结论：Stage3 模型的决策确实使用了图像特异、与问题和选项有关的视觉信息。

- 96例外观匹配错图替换：GPT-4o Stage3 的正确选项 margin 降低 0.451，95% CI [0.215, 0.710]，准确率降低 10.4 pp；Grok Stage3 分别为 0.440 [0.204, 0.684] 和 9.4 pp。
- 几何匹配错图替换仍有显著 margin 差，排除了图像尺寸和视觉 token 数量差异。
- LLM 第20/24层的 clean activation 回填能够恢复大部分因错图而损失的正确选项 margin；同范数置换方向无法恢复，支持方向特异的因果传播。
- option-conditioned RISE 所找区域在删除和保留两种相反干预下均显著优于同形状随机位置：GPT-4o deletion +0.2407、retention +0.2097，四项 Holm 校正后均 `p=0.00004`。

上述证据不依赖 Judge，也不要求证明 Stage3 显著优于 Stage2。它支持“最终 Stage3 模型在作答时有效利用视觉证据”，但不能单独表述为“Stage3 相比 Stage2 显著增强了视觉依赖”，也不能把尚未通过人工病理复核的 ROI 称为病理学 ground-truth region。

## 人工材料

生成目录：

`/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/stage2_stage3_human_review_packet_20260815`

- `blind_pairwise_multijudge_panel100/review_packet`：100例输出无关盲化A/B包。
- `improvements_multijudge_aligned_stage3`：与多 Judge 对齐模型的55例错转对案例。
- `improvements_current_n8_checkpoint1500`：当前 n=8 checkpoint1500 的183例错转对案例。
