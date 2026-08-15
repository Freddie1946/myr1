# PathVLM-R1 人工复核包使用说明

本说明对应三个相互独立的盲评任务。评审者只应收到 reviewer-blinded 包；包含答案映射、模型身份和自动分数的 owner 包必须由课题负责人保管。

## 任务 A：训练期 GPT-4o 过程奖励复核

入口：`reward_agreement_blinded/index.html`。

- Reviewer A 评全部 60 例；Reviewer B 至少评预设重叠 30 例。
- 不猜测模型阶段或 Judge 身份，只依据图像、问题和候选推理判断六个事件。
- 代表性 36 例与 challenge 24 例最终分层报告。
- 任何无法仅凭图像可靠判断的题应标为 uncertain，而不是强行给分。

## 任务 B：可解释性候选区域复核

入口：`interpretability_roi_blinded/index.html`。

- 共 20 例。候选框编号不代表算法排名。
- 逐框判断是否包含诊断相关视觉证据，并记录是否遗漏图中其他关键区域。
- 评审者不会看到模型预测、显著图、删除效果或 activation patching 结果。
- 未经专家确认，这些区域统一称 pseudo-reference evidence regions。

## 任务 C：Stage2 与 Stage3 回答质量盲评

入口：`stage2_stage3_blind_pairwise_100/README.md` 与 `ratings_template.csv`。

- 共 100 例，A/B 位置已按固定种子随机交换，模型身份隐藏。
- 主评分为 A better / B better / Tie。
- 次评分为医学事实、视觉依据和推理质量，各 1–5 分。
- 两名评审独立完成；第三名评审只用于仲裁分歧。

## 结果回传

请保留原始 CSV，不要覆盖另一位评审文件。建议命名为：

```text
reward_reviewer_A_completed.csv
reward_reviewer_B_completed.csv
roi_reviewer_A_completed.csv
roi_reviewer_B_completed.csv
pairwise_reviewer_A_completed.csv
pairwise_reviewer_B_completed.csv
```

改善案例包并非随机总体样本，只能用于定性展示，不能用来估计总体人工偏好率。
