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

辅助入口继续隐藏候选回答的模型与训练阶段身份，但不再对外部参考盲化，因此结果必须标为 `reference-assisted expert review`。奖励任务页面现已严格收缩为训练 Judge 的六个布尔事件，不再要求 1–5 分、幻觉、参考答案噪声或总体质量分。每个事件只有 `True/False` 两个可点击选项，其下直接显示 Claude Sonnet 4.6 的对应 `True/False` 参考；三个正向事件缺失数为 `m`，三个错误事件出现数为 `e`，`S_integrity=max(0,1-0.4m)`、`S_knowledge=max(0,1-0.4e)`、`R_process=(S_integrity+S_knowledge)/2`。

奖励页必须通过 `scripts/serve_two_pass_expert_review.sh` 启动，不能再用普通 `python -m http.server`，因为只有应用服务器才能接收点击结果。专家先填写页面顶部的 reviewer ID，再逐项点击并按“保存本题评分”；服务端只接受精确的六个布尔字段，以原子替换方式写入：

`/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/human_review/expert_submissions_20260815/<reviewer_id>/`

每题保存为一个 JSON，同时刷新 `reward_six_event_ratings.csv`。刷新页面或更换病例后可按 reviewer ID 恢复既有评分。网页即时显示人工过程分、Claude 参考过程分以及逐题 `人工−Claude` 分差。

`Reviewer ID` 应由负责人在评审开始前分配为固定匿名编号，例如 `pathologist_A`、`pathologist_B`，不要填写真实姓名或邮箱。它不是为了在同一台设备区分多人：正式分发时，每位专家各自收到一份自包含包，在自己的电脑上运行并使用自己的编号。专家第一次在第一题录入；后续浏览器只用 localStorage 自动带出该编号。真正的恢复来源是本机解压目录 `expert_submissions/<Reviewer ID>/` 中的 JSON/CSV；不查询负责人机器、HF 或任何远程服务。换浏览器时输入同一编号并点击“载入本题已保存评分”即可；换机器时必须一并迁移该本地目录。编号拼错会被视为另一位新评审者并创建新目录，因此正式开始前应将编号列表冻结。

每位专家完成后点击“导出我的评分包”，浏览器会下载 `pathvlm_reward_review_<Reviewer ID>_complete60.zip`。ZIP 内含 60 个逐题 JSON、汇总 CSV 和带 SHA-256 的 `EXPORT_MANIFEST.json`，直接交还负责人即可。未完成时也允许导出断点备份，但文件名会明确标成 `partialN`，不能当作完整人工结果。负责人收到不同专家 ZIP 后先核对 reviewer ID、`complete=true`、`completed_cases=60` 和 manifest 校验，再进行汇总分析。

辅助入口已将评分规则、题干、选项、参考答案、待评分回答、外部模型理由和区域特征全部改为中英文对照。中文用于降低阅读负担，英文原文同时保留并作为语义冲突时的权威版本；`<think>`、`<answer>` 和 A/B/C/D 标签经过自动完整性检查，不能在翻译中增删。评分 CSV 的机器字段名仍保持英文，以确保汇总脚本兼容；页面内提供逐字段中文释义。

## 外部参考中的结构化指标是什么意思

奖励复核的六个布尔事件中，前三项检查回答是否实际完成了图像特征分析、选项排除和医学知识支持；后三项检查是否存在组织学定义错误、逻辑矛盾和错误/过时的病理标准。`true` 表示该正向成分确实存在，或该负向错误确实发生；`false` 表示不存在。当前直接点击入口为了与训练 schema 完全一致，不提供 `uncertain` 或任何额外质量分；确实无法判断时可暂不保存并在病例讨论后再完成。

ROI 参考中的 `box=[x1,y1,x2,y2]` 是将宽高归一化到 0–1000 后的坐标，不是原图像素；`feature` 是框内病理征象；`role` 表示该征象支持参考答案、反驳干扰项或只提供背景；`importance∈[0,1]` 是外部模型估计的本题相关重要度；`confidence∈[0,1]` 是模型对整份标注的自评信心。后二者都不是校准概率，也不是专家一致率。

生成质量参考中的 `r_acc`、`k_acc`、`rigor`、`professionalism`、`clarity`、`conciseness` 均在 `[0,1]`：分别衡量推理对最终答案的支持度、医学知识准确性、严谨性、专业性、清晰度和简洁性，越高越好。这些也是外部 Judge 的连续建议分，不是概率，专家仍应依据病例独立判断。

## 外部参考与训练期 GPT-4o 奖励的一致性

在冻结的 60 例面板上，以 Claude Sonnet 4.6 的结构化意见直接代替专家评分，与训练期 GPT-4o 六事件记录比较：360 个事件的 micro agreement 为 79.72%（病例聚类 bootstrap 95% CI 75.28%–83.89%），Cohen's kappa 为 0.566；由六事件换算的过程奖励精确相同率为 38.33%，85.00% 的病例差值不超过 0.2，奖励 Spearman 相关为 0.522（95% CI 0.308–0.698）。Claude 的平均过程奖励为 0.7067，GPT-4o 为 0.7750，平均差为 -0.0683。

因此当前只能表述为“总体中等一致，足以作为辅助参考，但不能互换”。差异主要集中在逻辑矛盾（Claude 更常判为存在）和错误/过时病理标准（GPT-4o 更常判为存在）。该面板还人为加入了 24 个 challenge 病例，不是总体随机样本；这些数值是机器—机器一致性，不是训练奖励已经通过专家验证的证据。

后续人工汇总以最终过程分差为主终点，而不是六事件总体一致率：逐病例用同一 0.4 公式分别换算人工、训练期 GPT-4o 和外部 Claude 的过程分，报告平均有符号差、MAE、RMSE、精确同分率、差值不超过 0.2 的比例和病例聚类 bootstrap 95% CI；六事件 agreement 只保留为辅助定位指标。汇总命令为：

```bash
cd /home/dataset-assist-0/czy/wjy/myr1
python3 scripts/analyze_human_reward_score_differences.py \
  --ratings-dir /home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/human_review/expert_submissions_20260815 \
  --gpt-key /home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/human_review/reward_agreement_60case_20260814/internal_selection_key.jsonl \
  --claude-references /home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/human_review/external_reference_reward_review_20260815/references.jsonl \
  --output-json /home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/human_review/expert_submissions_20260815/final_score_differences.json \
  --output-md /home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/human_review/expert_submissions_20260815/final_score_differences.md
```

由于当前点击入口从评分前就显示 Claude 参考，其结果只能称为参考辅助人工复核，不能再用于声称“独立病理专家验证”。如论文需要独立专家主统计，仍应使用不显示外部参考的第一阶段入口。
