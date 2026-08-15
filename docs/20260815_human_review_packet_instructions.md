# PathVLM-R1 人工复核包使用说明

本说明对应三个相互独立的盲评任务。评审者只应收到 reviewer-blinded 包；包含答案映射、模型身份和自动分数的 owner 包必须由课题负责人保管。

## 0. 负责人先做什么

只把下面这个文件发给病理专家：

`pathvlm_human_review_reviewer_blinded_20260815.tar.gz`

不要发送 `pathvlm_human_review_owner_complete_20260815.tar.gz`。后者含模型身份映射、自动评分和解盲键，一旦评审提前看到，Stage2/Stage3 A/B 比较和奖励一致性复核就不再盲化。

专家解压后，可直接双击各任务的 `index.html`/`review.html`。若浏览器阻止本地相对链接，可在解压目录运行：

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

再访问 `http://127.0.0.1:8000/`。HTML 只负责展示，评分必须填入 CSV；不要在 HTML 中期待“提交”按钮。

## 任务 A：训练期 GPT-4o 过程奖励复核

入口：`reward_agreement_blinded/index.html`。

- Reviewer A 按 `reviewer_A_all60.csv` 评全部 60 例；Reviewer B 按 `reviewer_B_suggested_overlap30.csv` 至少评预设重叠 30 例。
- 不猜测模型阶段或 Judge 身份，只依据图像、问题和候选推理判断六个事件。
- 代表性 36 例与 challenge 24 例最终分层报告。
- 任何无法仅凭图像可靠判断的题应标为 uncertain，而不是强行给分。

填写方式：复制 `ratings_template.csv` 为每位专家自己的 completed 文件，只保留该专家负责的 case_id。前三个事件判断回答是否实际包含图像特征分析、选项排除和医学知识支持；后三个事件判断是否存在组织学定义错误、逻辑矛盾或错误/过时诊断标准。布尔项填写 `true/false/uncertain`；质量、grounding、逻辑和信心使用 1（最差/最低）到 5（最好/最高）；无法判断时写 `uncertain` 并在 notes 解释，不要留给负责人猜测。

## 任务 B：可解释性候选区域复核

入口：`interpretability_roi_blinded/index.html`。

- 共 20 例。候选框编号不代表算法排名。
- 逐框判断是否包含诊断相关视觉证据，并记录是否遗漏图中其他关键区域。
- 评审者不会看到模型预测、显著图、删除效果或 activation patching 结果。
- 未经专家确认，这些区域统一称 pseudo-reference evidence regions。

需要填写两张表：

- `case_ratings_template.csv`：评价整题参考答案是否成立、全部候选区域总体相关性/覆盖度/特异性，是否漏掉诊断区域，是否需要修框；
- `region_ratings_template.csv`：逐个 `case_id + region_id` 判断该框是否包含诊断相关证据，并给边界质量 1–5 分。

若框只覆盖部分证据，使用 `partial`，不要勉强写 `yes`。若图中存在未框出的关键区域，必须将 `missing_diagnostic_region_yes_no` 标为 `yes` 并在 notes 中描述位置。该任务只验证区域的病理相关性，不要求专家判断 attention、删除效应或模型机制。

## 任务 C：Stage2 与 Stage3 回答质量盲评

入口：`stage2_stage3_blind_pairwise_100/README.md` 与 `ratings_template.csv`。

- 共 100 例，A/B 位置已按固定种子随机交换，模型身份隐藏。
- 这 100 例比较的是历史 Stage2 与完成三模型 Judge 时使用的 `Stage3-GPT4o-selected`，从而允许人工结果与既有 GPT-4o/Claude/Gemini 逐题对齐；它不是当前 n=8 step1500 的总体盲评。
- 主评分为 A better / B better / Tie。
- 次评分为医学事实、视觉依据和推理质量，各 1–5 分。
- 两名评审独立完成；第三名评审只用于仲裁分歧。

逐题打开 `stage2_stage3_blind_pairwise_100/case_XXX/review.html`，同时查看同一张图、问题、选项与匿名回答 A/B。填写：

- `preferred_response_A_B_Tie`：只能是 `A`、`B` 或 `Tie`；
- `confidence_1_to_5`：对偏好判断的信心；
- A/B 的医学事实正确性、视觉依据充分性、推理质量：各自独立给 1–5；
- `notes`：记录关键事实错误、图像证据遗漏或选择 Tie 的原因。

不要因为回答更长就判为更好，也不要只看最终选项；要同时检查医学事实、是否真正引用图像可见证据以及推理是否支持最终答案。若两者都错或都同样好，应选 `Tie`，而不是强行拉开差距。

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

owner 完整包另含当前 n=8 step1500 相对 Stage2 的 183 个错转对和 154 个对转错案例索引/材料；由于它们按模型输出结果筛选，只能解释“哪些病例变好/变坏”，不能替代当前 n=8 的输出无关总体盲评。若论文最终要报告“专家更偏好当前 n=8 step1500”，必须另外从同一冻结 100 例 panel 构建一次 n=8 盲化 A/B 包，不能复用这些改善案例计算比例。

## 负责人收到 CSV 后怎么处理

1. 先做字段和值域检查，不改专家原始记录；错误项退回原专家确认。
2. 两位专家结果在仍然盲化时合并，计算任务 A 的事件级一致率/F1、质量分相关与一致性，任务 B 的区域相关率/覆盖率及专家间一致性，任务 C 的 A/B/Tie 比例和 Cohen's kappa。
3. 仅在统计冻结后使用 owner 包中的映射解盲。任务 C 将匿名 A/B 还原为 Stage2/Stage3；第三位专家只复核两位专家有分歧的题。
4. 代表性/challenge、改善/退化案例必须分层或定性报告，不能与输出无关的 100 例总体 panel 混合计算偏好率。
5. 保留每位专家的原始 CSV、清洗后的只读副本、合并脚本输出和最终统计表，形成可审计链。

人工任务尚未完成前，论文中只能写“材料已冻结并等待专家复核”，不能把外部模型的区域或 Judge 分数称为病理专家结论。

## 外部模型参考意见：严格两阶段使用

三项任务都提供独立外部模型意见，但它们只能用于第二阶段辅助、仲裁和敏感性分析：

1. 专家先在完全盲化的页面独立完成评分并保存第一阶段 CSV；
2. 之后才打开 `second_pass_external_references` 中同一 `case_id` 的参考页；
3. 专家可维持或修改判断，但必须填写 `post_reference_change_log.csv`；
4. 论文主要人工统计只使用第一阶段评分，第二阶段改判率和参考帮助度另行报告。

奖励复核参考来自 Claude Sonnet 4.6，且不读取训练期 GPT-4o 奖励；ROI 参考分别展示 Gemini 3.1 Pro 和 Claude Opus 5 的独立区域意见；生成质量参考分别展示 Claude Sonnet 4.6 和 Gemini 3.1 Pro 对匿名回答 A/B 的分项意见。它们都不是病理专家真值。

当前机器可直接运行：

```bash
cd /home/dataset-assist-0/czy/wjy/myr1
bash scripts/serve_two_pass_expert_review.sh 8765
```

然后访问 `http://127.0.0.1:8765/`；若使用远程 IDE，应先转发/预览 8765 端口。服务目录不包含 API key 或 owner 解盲键。

根页面现在提供两种入口：

- `独立盲评入口`：不显示外部参考，用于论文主要人工统计；
- `外部参考辅助评审入口`：每题左右并列显示病例/待评内容与外部模型意见，并提供上一题、题目目录、下一题导航。

辅助入口继续隐藏候选回答的模型与训练阶段身份，但不再对外部参考盲化，因此结果必须标为 `reference-assisted expert review`。奖励任务页面直接显示与训练一致的六事件规则：三个正向事件缺失数为 `m`，三个错误事件出现数为 `e`，`S_integrity=max(0,1-0.4m)`、`S_knowledge=max(0,1-0.4e)`、`R_process=(S_integrity+S_knowledge)/2`。幻觉与参考答案歧义仍作为补充审计字段，不进入六事件公式。
