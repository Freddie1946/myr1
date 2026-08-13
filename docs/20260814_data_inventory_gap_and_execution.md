# 数据盘点、缺口与执行状态（2026-08-14）

## 已有数据

### 核心训练主线

- Base、历史 full-SFT3000、历史 Stage2 outcome-RL1000：PathMMU、PathVQA、MMMU已有；OmniMedVQA统一合同正在补齐。
- 当前 L-r16-SFT3000 selected parent：PathMMU Val/Test、PathVQA固定A/B、OmniMedVQA、MMMU、视觉依赖均已有。
- full rule-RL n4/n8 step1000：域内、PathVQA固定A/B、OmniMedVQA、MMMU均已有；n8是Stage3 parent。
- GPT-4o Stage3 step500/1000/1500：PathMMU、PathVQA固定A/B、OmniMedVQA、MMMU已齐。
- LoRA-SFT4000控制：训练完成；PathMMU Val/Test、PathVQA Val两种接口、MMMU、Normal/Shuffle/Blank已齐；PathVQA Test3362与OmniMedVQA运行中。

### 数据规模/配比消融

- 250 SFT + 750 rule-RL、500+500、750+250：训练和完整权重均存在，统一评测已于2026-08-14启动。
- 3000 L-LoRA SFT + 1000 full rule-RL n4/n8：完整。
- 3000 L-LoRA SFT + 相同1000额外LoRA-SFT：训练完成，统一评测运行中。
- 0 SFT + 4000 rule-RL：只有checkpoint-2500；覆盖4000条至少一次，但没有达到原6000-step计划，必须标为中间诊断。

### 训练机制与可解释性

- SFT结构：C0、L-r16、L-r32、A、B2、B4已有训练与核心评测。
- RL机制：R0/R1、G20-LR1/LR3、G2-LR1、LoRA/full、n4/n8已有。
- 数据划分审计：Base上SFT3000与RL1000没有显著静态难度偏差；parent-conditioned审计已完成。
- 可解释性：58例行为干预、29例严格主集、30例dual-stream internal causal tracing已有；外部模型ROI尚非病理专家真值。

### PathVQA基线覆盖

已经跑过PathVQA的外部/病理基线包括：Huatuo、InternVL、Lingshu、MedGemma、MedVLM-R1、Qwen3、Qwen2.5-VL-7B、DeepSeek-VL2、Llama-3.2-90B-Vision、ScaleReasoner-R1、Claude Haiku 4.5、Qwen-VL-Plus。Llama-3.2-11B只有smoke，未完成全量。

其中只有当前核心主线使用了最新固定 `A=Yes/B=No` 生成合同。其他基线多数使用旧的short-answer/自由Yes-No合同；逐题预测仍可保留和重评分，但不能把它们伪装成已按新prompt重跑。已知旧合同yes/no结果示例：ScaleReasoner-R1 57.53%、DeepSeek-VL2 paper-aligned yes/no 59.52%、Llama-3.2-90B 62.49%、Claude Haiku 4.5为1355/3359=40.34%、Qwen-VL-Plus为2272/3356=67.70%。后两者存在少量API失败，分母不是3362。

## 尚缺数据

1. 三组小规模配比的统一协议 PathMMU Val/Test、PathVQA Test A/B、MMMU、OmniMedVQA（已启动）。
2. 最终三模型的5-seed随机推理重复：L-SFT3000、rule-RL n8、GPT-4o Stage3；主greedy结果保持不变（已排队）。
3. LoRA-SFT4000的PathVQA Test与OmniMedVQA全量终点（运行中）。
4. 历史SFT/Stage2的OmniMedVQA统一合同终点（运行中）。
5. 外部基线若要与当前A/B主表完全同prompt公平比较，需要逐模型重跑；现有旧合同结果只能作为分层表。
6. 病理专家对解释性ROI以及生成质量的盲评（由用户后续处理）。
7. 最终生成质量多Judge统一抽样表尚未收口。

## 重复推理与统计口径

- 主性能仍报告deterministic greedy。
- 重复推理：5个固定seed（42–46），temperature 0.7、top-p 0.9、top-k disabled、max_new_tokens 1024；报告run-level mean、sample SD和Student-t 95% CI。
- case/bootstrap CI描述测试样本不确定性；5-seed CI描述解码随机性；两者不能互换。
- 不再扩展全模型paired flip/badcase表。由于审稿人要求统计显著性，最终仅为2–3个核心对照保留paired bootstrap/McNemar。

## 备份

- GitHub：代码、报告、运行合同、manifest和小型汇总。
- Hugging Face dataset：逐题预测、metrics、run_config、关键日志和hash manifest；不上传原始图片、密钥、优化器shard或超大的重复RISE中间张量。
- 当前HF已有两个评测仓库；`PathVLM-R1-Revision-Evaluation-Results`约659MiB，`PathVLM-R1-Evaluation-Archive-20260812`约1.3MiB。HF认证正常。

2026-08-14当前快照已上传到私有dataset仓库：

- Repository：`Freddie1946/PathVLM-R1-Evaluation-Archive-20260814`
- Revision：`57f836833cf3417692ffa4cb40002349fc8dd89b`
- Archive：`evaluation_results_20260814.tar.gz`
- Archive size：118,407,533 bytes
- Archive SHA256：`ff82191de645f277e0db012b9aa97dc9e9259dfac88d915772a452d7169c70d8`
- Manifest SHA256：`8429efdbd97a7d0d9412370a66b2385eac39e694a462fc84faf03e2d1e8a6980`
- 归档文件数：1,337
- 未压缩结果数据：1,009,698,691 bytes

这是启动新补充评测之后、其完成之前的时间点快照；新运行完成后必须再做增量/最终快照。
