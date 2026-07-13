# 代码版本审计记录

更新时间：2026-07-05

本文档记录当前本机 `/home/wjy` 与远程服务器 `root@region-42.seetacloud.com:53985` 上已发现的核心代码、数据版本差异、论文描述与代码实现之间的潜在分歧，以及后续统一版本时需要重点确认的缺失隐患。

## 1. 背景

当前工作区存在多套训练、评测、打分和数据处理脚本，文件版本管理较混乱。论文 `PathVLM_R1__A_Reinforcement_Learning_Driven_Reasoning_Model_for_Pathology_Visual_Language_Tasks.pdf` 描述的是三阶段训练流程：

1. **Stage 1: SFT / Knowledge Injection**
   - 使用 3000 条 PathMMU 数据对 Qwen2.5-VL-7B-Instruct 进行监督微调，得到 Alpha 模型。

2. **Stage 2: GRPO / Outcome Rewards**
   - 使用 1000 条 RL 数据，通过 Accuracy Reward 和 Format Reward 训练，得到 Beta 模型。

3. **Stage 3: GRPO / Dual Rewards**
   - 继续使用同一批 1000 条 RL 数据，在 Accuracy Reward 和 Format Reward 基础上加入 GPT-4o 生成的 Process Reward，得到最终 PathVLM-R1。

论文中的 Process Reward 逻辑为：将图像、问题、标准答案、模型推理链和模型答案输入 GPT-4o，由 GPT-4o 输出 `integrity_score` 和 `knowledge_score`，最终过程奖励约等于二者平均值。

## 2. 本机核心代码

### 2.1 训练相关

- `/home/wjy/papers/VLM-R1-main/src/open-r1-multimodal/src/open_r1/grpo_rec.py`
  - 本机与远程 `/root/VLM-R1/src/open-r1-multimodal/src/open_r1/grpo_rec.py` 哈希一致。
  - 当前注册的 reward 只有 `accuracy_reward` 和 `format_reward`。
  - 对应论文中的 **Stage 2: GRPO with Accuracy + Format Reward**。
  - 目前未发现该文件接入 GPT-4o Process Reward。

- `/home/wjy/papers/VLM-R1-main/src/open-r1-multimodal/src/open_r1/grpo_rec_old.py`
  - 存在旧版 REC/BBox 训练逻辑。
  - 本机与远程 `grpo_rec_old.py` 哈希不同。
  - 旧版中出现过 IoU reward 相关实现或注册痕迹，需要单独确认是否曾用于 REC 训练或只是历史残留。

- `/home/wjy/papers/VLM-R1-main/src/open-r1-multimodal/src/open_r1/grpo_jsonl.py`
  - 本机版本包含 `OpenAI` 调用，但主要用于答案相似度/LLM 判断，如 `llm_reward`。
  - 没有发现论文 Fig.3 描述的 `integrity_score + knowledge_score` 过程奖励训练逻辑。
  - 远程版本更简单，仅包含 `accuracy_reward` 和 `format_reward`。

### 2.2 评测与打分相关

- `/home/wjy/evaluate_with_gpt4o_v2.py`
  - 最接近论文 Fig.3 中 GPT-4o process scorer 的离线实现。
  - 输出字段包括 `integrity_score`、`knowledge_score`、`wrong_reasoning_but_correct_answer`。
  - 该脚本是离线评测/打分脚本，目前没有发现它被接入 GRPO 训练循环。

- `/home/wjy/scorev3.py`
  - 离线六维推理质量评测脚本。
  - 输出维度包括 `integrity_score`、`knowledge_score`、`logic_score`、`professionalism_score`、`clarity_score`、`conciseness_score`、`wrong_reasoning_but_correct_answer`。
  - 更接近论文中后续评测 reasoning quality 的扩展版本。

- `/home/wjy/score4.py`
  - 改进版离线评测脚本。
  - 将旧的 `integrity_score` 思路调整为更重视有效推理的 `reasoning_effectiveness`。
  - 主要为了解决旧评分标准对冗长回答的偏置，即 verbosity bias。
  - 该脚本更适合作为修订后评测逻辑，但不是训练 reward 实现。

- `/home/wjy/calculate_scores.py`、`/home/wjy/calculate_scores_v3.py`、`/home/wjy/calculate_scores_v4.py`
  - 分别用于汇总不同版本评分结果。

- `/home/wjy/recompute.py`
  - 用于重新判定 `correct` 字段。
  - 逻辑是从 `ground_truth` 的 `<answer>` 中提取 A/B/C/D，再检查 `model_output` 的 `<answer>` 中是否包含正确选项。

### 2.3 数据处理相关

- `/home/wjy/add_image_field.py`
  - 根据 question/problem 匹配，将源数据中的 `image` 字段补到模型输出 JSON 中。

- `/home/wjy/extract_matching_questions.py`
  - 根据参考文件中的 question 集合，从其他结果文件中过滤出匹配样本。

- `/home/wjy/download.py`
  - 用于下载/转换 OmniMedVQA 数据。

## 3. 远程服务器核心代码

远程服务器已确认的关键目录：

- `/root/VLM-R1`
- `/root/data`
- `/root/paper/VLM-R1-main`
- `/root/autodl-tmp`

远程 `/root/VLM-R1` 是唯一确认的 Git 仓库。当前远程 Git 状态曾显示以下文件存在改动或未跟踪：

- 修改：`src/open-r1-multimodal/data_config/rec.yaml`
- 修改：`src/open-r1-multimodal/run_grpo_rec.sh`
- 修改：`src/open-r1-multimodal/src/open_r1/grpo_rec.py`
- 未跟踪：`src/open-r1-multimodal/README.md`
- 未跟踪：`src/open-r1-multimodal/data_config/processed_rec.yaml`
- 未跟踪：`src/open-r1-multimodal/run_grpo_rec_old.sh`
- 未跟踪：`src/open-r1-multimodal/src/open_r1/grpo_rec_old.py`

需要注意：虽然远程 Git 状态显示 `grpo_rec.py` 被修改，但实际与本机 `papers/VLM-R1-main/.../grpo_rec.py` 计算出的 SHA256 哈希一致，说明本机当前版本已经包含远程该文件内容。

## 4. 本机与远程的已知差异

### 4.1 `grpo_rec.py`

- 本机：`/home/wjy/papers/VLM-R1-main/src/open-r1-multimodal/src/open_r1/grpo_rec.py`
- 远程：`/root/VLM-R1/src/open-r1-multimodal/src/open_r1/grpo_rec.py`
- 状态：哈希一致。
- 结论：当前主训练 reward 逻辑两边一致，均为 Accuracy + Format，没有发现 GPT-4o Process Reward。

### 4.2 `grpo_rec_old.py`

- 本机与远程哈希不同。
- 远程文件大小约 14156 bytes，本机约 14147 bytes。
- 该文件可能是历史 REC/BBox reward 版本，需要后续 diff 细看。
- 风险：旧版 IoU reward 与当前论文 PathMMU 多选问答 reward 可能混在同一实验目录中，容易误判实际使用逻辑。

### 4.3 `grpo_jsonl.py`

- 本机版本包含 `OpenAI` 客户端和 `llm_reward` 等逻辑。
- 远程版本未发现 OpenAI 客户端，仅有 Accuracy + Format reward。
- 本机版本的 OpenAI 逻辑用于答案相似度判定，不等价于论文中基于推理链完整性和医学知识合理性的 Process Reward。
- 风险：本机 `grpo_jsonl.py` 可能是另一次实验或后续修改版本，不应直接认定为论文最终 Stage 3 训练代码。

### 4.4 `rec.yaml`

本机：

```yaml
datasets:
    - json_path: /home/wjy/rec_jsons_processed/refcoco_train.json
    - json_path: /home/wjy/rec_jsons_processed/refcocop_train.json
    - json_path: /home/wjy/rec_jsons_processed/refcocog_train.json
```

远程：

```yaml
datasets:
    - json_path: /root/autodl-tmp/coco2014/annotations/refcoco_train.json
    - json_path: /root/autodl-tmp/coco2014/annotations/refcocop_train.json
    - json_path: /root/autodl-tmp/coco2014/annotations/refcocog_train.json
```

差异主要是绝对路径不同。需要继续确认两边 JSON 内容哈希是否一致。

### 4.5 `run_grpo_rec.sh`

远程 `/root/VLM-R1/src/open-r1-multimodal/run_grpo_rec.sh` 的当前内容存在路径不一致风险：

- 脚本位于远程服务器，但 `--dataset_name` 指向 `/home/wjy/papers/VLM-R1-main/.../processed_rec.yaml`
- `--image_root` 指向 `/home/wjy/processed_data/images`

这些路径看起来更像本机路径，而不是远程路径。旧版远程脚本 `run_grpo_rec_old.sh` 使用的是：

- `--model_name_or_path /root/autodl-tmp/pretrained/Qwen2.5-VL-3B-Instruct`
- `--dataset_name data_config/rec.yaml`
- `--image_root /root/autodl-tmp/coco2014/images`

风险：远程新脚本可能是从本机复制后未完成路径迁移，直接运行可能找不到数据。

### 4.6 评测脚本 `test_rec_baseline.py`

本机与远程主要差异：

- 本机 `MODEL_PATH` 指向 `/home/wjy/LLaMA-Factory/saves/qwen2_5_vl-3b-3.20.1648/full/sft/checkpoint-1200`
- 远程 `MODEL_PATH` 仍是占位符 `path/to/Qwen2.5-VL-3B-Instruct`
- 本机输出文件名带有 `baseline_3.20_1200`
- 远程输出文件名为通用 `baseline.json`

结论：本机版本更像实际运行过的 baseline 评测脚本，远程版本更像模板或未配置版本。

### 4.7 评测脚本 `test_rec_r1.py`

本机与远程主要差异：

- 本机 `steps = 200`
- 远程 `steps = 100`
- 本机 `re.search(answer_tag_pattern, content, re.DOTALL)`
- 远程 `re.search(answer_tag_pattern, content)`，缺少 `re.DOTALL`

风险：如果模型输出 `<answer>` 跨行，远程版本可能提取失败。本机版本更稳妥。

## 5. 数据版本情况

### 5.1 `/root/data` 与本机 `processed_data`

已确认：

- 远程 `/root/data/picked.json`
  - 与本机 `/home/wjy/processed_data/picked.json` 哈希一致。
  - 1000 条。
  - 字段：`image`, `problem`, `solution`。

- 远程 `/root/data/output.json`
  - 与本机 `/home/wjy/processed_data/output.json` 哈希一致。
  - 6719 条。
  - 字段：`image`, `problem`, `solution`。

- 远程 `/root/data/dataset.json`
  - 与本机 `/home/wjy/processed_data/dataset.json` 结构一致，但哈希不同。
  - 差异主要来自 `image_path` 的绝对路径不同：远程为 `/root/data/images/...`，本机为 `/home/wjy/processed_data/images/...`。

结论：`picked.json` 和 `output.json` 两边可视为同源；`dataset.json` 内容大概率同源，但路径环境不同。

### 5.2 PathMMU 数据

本机 PathMMU 相关目录较完整：

- `/home/wjy/PathMMU`
- `/home/wjy/train_data`
- `/home/wjy/valid_data`
- `/home/wjy/processed_data`

论文中使用的数据划分为：

- 3000 条 SFT 训练
- 1000 条 RL 训练
- 1385 条测试

当前已经看到与 1000 条 RL 数据对应的 `picked.json`，但仍需确认 3000 条 SFT 数据和 1385 条测试数据的最终 canonical 文件是哪一个。

## 6. 论文逻辑与代码实现的潜在分歧

### 6.1 最关键分歧：Stage 3 训练代码可能缺失

论文最终模型 PathVLM-R1 声称使用：

```text
Accuracy Reward + Format Reward + GPT-4o Process Reward
```

但当前已检查的本机和远程训练代码中，只完整发现：

```text
Accuracy Reward + Format Reward
```

未找到完整的 `process_reward` 函数，也未找到 `reward_funcs_registry` 注册 `process`、`gpt4o`、`integrity` 或 `knowledge` reward 的训练脚本。

因此，目前可见代码只能严谨支撑论文中的 Stage 2/Beta 模型训练逻辑；最终 Stage 3/PathVLM-R1 训练实现可能缺失、未同步、在其他目录，或当时以临时脚本方式运行后没有保存。

### 6.2 离线评测代码与训练 reward 容易混淆

`evaluate_with_gpt4o_v2.py` 与论文 Fig.3 非常接近，但它是离线评分脚本，不是 GRPO 训练 reward。

`scorev3.py` 和 `score4.py` 是更复杂的离线评测脚本，也不应直接描述为训练阶段 reward，除非后续明确将其接入训练循环。

### 6.3 旧 REC/BBox 代码与 PathMMU 多选问答代码混杂

`grpo_rec.py`、`grpo_rec_old.py`、`test_rec_r1.py`、`test_rec_baseline.py` 中仍保留 REC/BBox、IoU、RefCOCO 等逻辑。

论文主任务是 PathMMU 病理多选视觉问答。需要避免把 REC grounding 实验代码误写成 PathMMU 主实验代码。

### 6.4 模型基座存在 3B/7B 表述风险

论文描述基座为 Qwen2.5-VL-7B-Instruct；部分代码路径和脚本名中出现 Qwen2.5-VL-3B 或 `Qwen2.5-VL-3B-Instruct`。

需要确认最终论文实验到底使用的是 7B 还是 3B，以及代码目录中 3B 脚本是否只是 VLM-R1 原始 REC 代码残留。

## 7. 当前建议的统一版本策略

暂不修改代码的前提下，建议先建立如下版本登记：

```text
canonical_record/
  stage1_sft/
    status: 待确认
    expected_data: 3000 PathMMU samples
    expected_model: Qwen2.5-VL-7B-Instruct -> Alpha

  stage2_grpo_acc_format/
    status: 已找到主要实现
    code: papers/VLM-R1-main/src/open-r1-multimodal/src/open_r1/grpo_rec.py
    remote_match: yes
    rewards: accuracy, format

  stage3_grpo_process_reward/
    status: 未找到完整训练实现
    expected_rewards: accuracy, format, process
    process_reward_source: GPT-4o integrity_score + knowledge_score

  offline_eval_paper_style/
    status: 已找到近似实现
    code: evaluate_with_gpt4o_v2.py

  offline_eval_revision/
    status: 已找到
    code: scorev3.py, score4.py, calculate_scores_v3.py, calculate_scores_v4.py
```

## 8. 后续待办

1. 继续搜索是否存在真正的 Stage 3 训练脚本，关键词包括：`process_reward`、`integrity_score`、`knowledge_score`、`gpt4o_reward`、`dual_reward`、`OpenAI` + `GRPOTrainer`。
2. 对 `grpo_rec_old.py` 做本机/远程 diff，确认旧版 IoU reward 是否有实际使用价值。
3. 对本机和远程的 REC JSON 标注文件计算哈希，确认 `rec.yaml` 只是路径不同还是数据内容也不同。
4. 明确论文实验使用的模型尺寸：3B 还是 7B。
5. 明确 3000 条 SFT、1000 条 RL、1385 条测试分别对应当前文件系统中的哪些 JSON 文件。
6. 后续如果需要补全论文 Stage 3 训练代码，应基于 `grpo_rec.py` 增加独立的 `process_reward`，而不是直接修改现有 Stage 2 代码，以免破坏已有可复现版本。

## 9. 当前结论

当前最可靠的判断是：

- 两台机器上的主 GRPO 训练文件 `grpo_rec.py` 已同步。
- 当前可见训练代码对应论文 Stage 2，即 Accuracy + Format Reward。
- 论文最终 Stage 3 所需的 GPT-4o Process Reward 训练实现尚未在已检查路径中发现。
- GPT-4o 过程评分逻辑以离线脚本形式存在，主要体现在 `evaluate_with_gpt4o_v2.py`、`scorev3.py` 和 `score4.py` 中。
- 数据 `picked.json` 和 `output.json` 在本机与远程之间一致，`dataset.json` 主要差异是绝对路径。
- 目前不建议直接合并或覆盖代码，应先完成版本登记和缺失项确认。
