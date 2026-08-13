# 最终实验报表与数据完备度（2026-08-14）

## 状态口径

- `READY`：统一协议的逐题预测和指标已经存在，可以直接汇总。
- `RUNNING`：训练或评测已启动，完成后可以入表。
- `REEVAL`：权重存在，但旧结果不能与当前协议直接混用，需要统一重评。
- `MISSING`：实验、标注或重复推理尚未完成。
- `PARTIAL`：仅有中间 checkpoint 或部分数据，不能宣称完整实验。

## 表 1：核心模型与基线总体性能

建议列：模型、训练阶段/可训练参数、PathMMU Val385、PathMMU Test999、PathVQA Test3362（固定 `A=Yes/B=No`）、OmniMedVQA 8518（contract-aligned 与 official 两列）、MMMU non-medical 116、解析率、截断率。

| 模型族 | 状态 | 说明 |
|---|---|---|
| Base Qwen2.5-VL-7B | `RUNNING` | PathMMU/PathVQA 已有；OmniMedVQA 统一重评已完成 Base，历史队列继续运行。 |
| 历史 full-SFT3000 | `RUNNING` | PathMMU/PathVQA/MMMU 已有；OmniMedVQA 统一重评运行中。 |
| 历史 Stage2 full-RL1000 | `RUNNING` | PathMMU/PathVQA/MMMU 已有；OmniMedVQA 将自动接续。 |
| 当前 L-r16 SFT3000 step80 | `READY` | 域内、PathVQA、OmniMedVQA、MMMU 和视觉依赖均已有。 |
| 当前 full rule-RL n=4 step1000 | `READY` | 核心域内/域外结果已有。 |
| 当前 full rule-RL n=8 step1000 | `READY` | 核心域内/域外结果已有；为 Stage3 parent。 |
| GPT-4o Stage3 step500/1000/1500 | `READY` | 三个 checkpoint 的 PathMMU/PathVQA/MMMU/OmniMedVQA 已有；step500/1000/1500 可做早中晚曲线。 |
| 开源/闭源外部基线 | `PARTIAL` | 历史结果较多，但需按评测合同版本分层；不能把旧的开放问答解析结果与当前固定选择接口混合。 |

## 表 2：数据规模与 SFT/RL 配比消融

建议列：SFT 样本数、RL 样本数、总样本池、训练范式、epoch/prompt exposure、PathMMU Val/Test、PathVQA、OmniMedVQA、MMMU。

| 配比 | 状态 | 可用边界 |
|---|---|---|
| 250 SFT + 750 rule-RL | `REEVAL` | 权重和训练审计完整；全参数语言训练、总池1000、SFT 10 epochs。需统一评测。 |
| 500 SFT + 500 rule-RL | `REEVAL` | 同上。 |
| 750 SFT + 250 rule-RL | `REEVAL` | 同上。 |
| 3000 L-LoRA SFT + 1000 full rule-RL（n=4） | `READY` | 当前主路线完整端点。 |
| 3000 L-LoRA SFT + 1000 full rule-RL（n=8） | `READY` | 当前主路线完整端点和 Stage3 parent。 |
| 3000 L-LoRA SFT + 额外1000 LoRA-SFT | `RUNNING` | 训练已完成；核心统一评测运行中。 |
| 历史 full-SFT3000 + 额外1000 full-SFT | `REEVAL` | 权重存在，旧评测合同不宜直接混入当前表。 |
| 0 SFT + 4000 rule-RL | `PARTIAL` | checkpoint-2500 覆盖全部4000条至少一次，但原6000-step/3-epoch计划在2601停止；不能写成完整4000-RL。 |

小规模三组只能表述为“固定1000条总池上的稀疏配比筛查”，不能冒充完整4000条规模曲线；它们内部训练范式一致，因此三组之间可以公平比较。

## 表 3：训练机制消融

建议拆为两个子表，避免一次改变多个变量。

### 3A：SFT 架构与容量

列：C0/full-LLM、L-r16、L-r32、A(Projector train)、B2/B4(Vision LoRA)，以及 PathMMU、PathVQA、MMMU、Normal-Shuffle、Normal-Blank、train-val gap。

状态：`READY`。原始预测和训练曲线大部分齐全，仍需统一汇总和置信区间。

### 3B：RL 优化机制

列：R0(acc+format)、R1(acc only)、G20-LR1、G20-LR3、G2-LR1、LoRA-RL、full-RL、n=4、n=8；配套 train probe、Val/Test、KL、effective ΔW、paired flips、active/zero-advantage group。

状态：`READY`。机制筛查数据已经存在；还需生成一张统一方向/幅度表，避免只比较最终 accuracy。

## 表 4：泛化、接口与视觉依赖

建议包含：

1. PathVQA 自由 Yes/No 与固定 `A=Yes/B=No` 两种接口；
2. 解析率、严格格式率、generation-cap hit；
3. OmniMedVQA 总体以及 Chest CT、DR、ISIC、OCT 分源结果；
4. MMMU non-medical retention；
5. Normal、Shuffle、Blank 与 `Delta_vision`；
6. Base-correct→model-wrong、Base-wrong→model-correct 的 paired flips 和代表性 bad/good cases。

状态：主 lineage `READY`；历史 full-SFT/Stage2 OmniMedVQA `RUNNING`；新增 LoRA-4000 `RUNNING`。

## 表 5：不确定性与统计显著性

建议报告：

- 每个 accuracy 的 case-bootstrap 95% CI；
- 同一测试集模型之间的 paired bootstrap 差值 CI；
- McNemar exact test（错→对、对→错）；
- OmniMedVQA 按图像/source cluster bootstrap，避免把同图多题当完全独立；
- 多 checkpoint 曲线仅作 post-hoc 诊断，不以 Test999 反复挑点后宣称独立最终测试。

状态：`PARTIAL`。部分 PathMMU 比较已有 bootstrap/McNemar；最终所有模型的统一统计表尚未生成。当前绝大多数推理是 deterministic greedy，因此“同一个模型随机推理多次”的重复性实验仍为 `MISSING`。若审稿意见明确要求 inference repeats，需要为最终少数模型补固定 seed 的 stochastic repeats；case-bootstrap 不能冒充随机推理重复。

## 表 6：推理/对话质量与 Judge 稳健性

建议列：模型、回答正确性、病理依据正确性、图像依据充分性、逻辑一致性、幻觉、格式；分别给 GPT-4o、至少一个不同模型家族 Judge，以及人工盲评小样本。

状态：`PARTIAL`。历史多-Judge结果存在，但当前 n=8/最终 GPT-4o Stage3 的统一抽样尚未完整纳入；病理专家人工评分为 `MISSING`。必须明确 GPT-4o 既用于部分训练奖励又参与评价时的潜在同源偏倚，并把异构 Judge/人工评分作为主要稳健性证据。

## 表 7：可解释性与因果视觉证据

建议分三层：

1. 行为干预：Original、reference deletion、neighbor/random deletion 的 margin drop、flip rate、bootstrap CI；
2. 内部传播：Vision token 与 query-position 的逐层 divergence/restoration；
3. Activation patching：正确 patch、permuted patch、opposite-direction control 的 raw recovery 与 normalized recovery。

状态：计算部分 `READY`。扩展 panel 已有58例，严格主集29例，dual-stream主分析30例；参考区域由外部 GPT-4o 预先标注，尚未经过病理专家盲校，因此“病理学 ground truth ROI”仍为 `MISSING`。当前可形成有统计支持的模型内部因果结果，但专家定位正确性需要后补。

## 表 8：资源、成本与复现性

建议列：GPU型号/数量、训练时长、峰值显存、trainable parameters、optimizer steps、prompt/trajectory exposure、API模型与请求数、API费用、checkpoint策略、代码/数据/权重 hash。

状态：`PARTIAL`。Stage3 GPT-4o费用、训练日志和关键 hash 已有；所有历史实验的GPU时与API成本尚未统一汇总。

## 当前最小闭环顺序

1. 完成 LoRA-4000 核心评测并加入表2、表4；
2. 统一评测三组小规模比例权重；
3. 决定是否把纯4000-RL从 checkpoint-2500续到原6000-step终点；若不续，只报告为中间诊断；
4. 汇总核心模型和基线的统一协议主表；
5. 从逐题预测统一生成 bootstrap CI、McNemar和paired-flip表；
6. 对最终少数模型补生成质量的异构Judge与小规模人工盲评；
7. 获取病理专家对reference evidence region的事后盲校。

