# PathVQA 高覆盖解析、域外退化诊断与视觉适配门禁

日期：2026-08-11

## 结论摘要

1. PathVQA 的旧生成式评分确有明显格式损失。更长的生成上限只能把 GPT-4o/Grok Stage3
   的确定性解析率从约 91% 提高到约 92.6%；真正接近 Base 的解析率需要“严格规则解析 →
   盲 LLM 意图提取 → 独立矛盾核验”的保守阶梯。
2. 解析率提高不等于性能提高。OmniMedVQA 在 99.98% 严格解析覆盖下，GPT-4o Stage3
   仍从 Base 的 60.06% 降至 46.30%，配对差值 -13.76 个百分点，95% bootstrap CI
   [-15.02, -12.51]，McNemar `p=5.01e-100`。这已排除“主要由截断/解析失败造成”的解释。
3. 当前正式训练不是“只训练文本编码器”：它更新完整 causal LM（约 7.616B 参数），冻结
   vision tower（约 676.6M）和 multimodal merger/projector（约 44.6M）。
4. 当前证据更符合窄任务继续训练造成的语言决策边界/接口遗忘和域、任务契约迁移，而不是
   已经证明视觉编码器失效。因此暂不直接解冻视觉塔；若需训练消融，应依次做 LM LoRA、
   LM LoRA+merger、最后才是 LM LoRA+merger+受限 vision LoRA/末层视觉块。

## 1. 高覆盖解析协议

### 1.1 冻结校准

- 数据：PathVQA 官方 validation 中冻结 128 个 yes/no 病例，Yes/No 各 64 个、图像不重复；
  不依据任何模型输出选样。
- 校准仅允许用解析率、生成封顶率选择提示；禁止用准确率选提示。
- 三种合同中，最简单的 `short_v1_192` 最稳健。更强 XML 约束反而降低解析率，要求只输出
  A/B 以及 32-token 上限几乎全部触发封顶，因此均被拒绝。

| 模型 | short 解析率 | XML 解析率 | A/B-only, 32 tokens | A/B, 192 tokens |
|---|---:|---:|---:|---:|
| GPT-4o Stage3 step 1000 | 90.63% | 85.94% | 3.13% | 88.28% |
| Grok Stage3 step 1500 | 93.75% | 90.63% | 1.56% | 未继续（GPT 臂已劣于 short） |

补做 192-token A/B 方案把截断因素排除后仍只有 88.28%，所以 A/B 提示本身也没有优于
简单短问法。该补充是看到 32-token 封顶后的自适应 validation 诊断，未用于正式 test 选型。

### 1.2 保守解析阶梯

1. 冻结确定性解析器；
2. 只把确定性未解析项发送给 `gpt-4.1-mini-2025-04-14`，输入仅含问题和模型输出；
3. 禁止发送图像、参考答案和被评模型身份；
4. LLM 必须输出 yes/no/unresolved，并引用模型原输出中的精确子串；
5. 所有恢复出的 yes/no 再由独立的 `gpt-4.1-2025-04-14` 检查是否存在实质矛盾；
6. 任一模式、引用、矛盾核验或重试失败均保持 unresolved，并按错误计分。

该流程只恢复已经写在 completion 中的回答意图，不能替模型判断医学正确性。

### 1.3 结果

- Base：3353/3362 可可靠解析，覆盖率 99.73%，准确率（未解析按错）66.54%。
- GPT-4o Stage3：3323/3362 可靠解析，覆盖率 98.84%，准确率（未解析按错）57.23%；若不做
  独立矛盾核验，名义意图覆盖可到 99.40%，但其中 18 项存在实质矛盾，故不采用。
- Grok Stage3：3331/3362 可靠解析，覆盖率 99.08%，准确率（未解析按错）57.61%；216 项
  由 LLM 恢复并通过独立核验，31 项仍因无判断、歧义或矛盾保持未解析。

重新生成旧 64-token 封顶且未解析的目标盲子集，只把 GPT/Grok 的确定性解析率提高到
92.56%/92.65%。由于旧运行 batch size=8、新运行 batch size=1，bf16 贪心解码并非逐 token
完全可复现，故这些结果明确标为 post-hoc fresh regeneration sensitivity，而不是原输出续写。

## 2. 格式中立的 PathVQA 能力诊断

为避免把生成格式当成知识能力，另对每题直接比较单 token `Yes` 和 `No` 的 next-token logits。
该指标有 100% 覆盖，不需要解析自由文本，也不是视觉归因指标。

128 例冻结 validation 面板结果：Base 71.09%，SFT3000 67.19%，Stage2 67.19%，SFT4000
57.81%，GPT-4o Stage3 67.97%，Grok Stage3 67.97%。Stage3 相对 Base 的 -3.13 个百分点
置信区间跨 0；SFT4000 的 -13.28 个百分点 95% CI 为 [-24.22, -2.34]，并在平衡面板上
产生 81.25% 的 Yes 预测率。

图像干预同一面板上，原图到均值空白图的准确率下降为 Base 22.66、GPT-4o 21.88、Grok
22.66 个百分点；循环错配图也使三者明显下降。Stage3 没有表现出明显弱于 Base 的视觉依赖。

全量 3362 题结果如下，所有模型覆盖率均为 100%。

| 模型 | 准确率 | 相对 Base | 95% paired bootstrap CI | McNemar p |
|---|---:|---:|---:|---:|
| Base | 66.39% | 0 | — | — |
| SFT3000 | 58.45% | -7.94 pp | [-9.67, -6.19] pp | 1.52e-18 |
| SFT4000 | 59.34% | -7.05 pp | [-9.28, -4.91] pp | 1.83e-10 |
| Stage2 Outcome-RL | 58.69% | -7.70 pp | [-9.43, -6.01] pp | 9.74e-18 |
| GPT-4o Stage3 | 59.04% | -7.35 pp | [-9.13, -5.59] pp | 7.29e-16 |
| Grok Stage3 | 58.86% | -7.53 pp | [-9.28, -5.80] pp | 8.20e-17 |

Stage2 到 GPT-4o/Grok Stage3 仅分别增加 0.36/0.18 个百分点，95% CI 均跨 0，McNemar
`p=0.182/0.561`；因此不能声称 Stage3 恢复了域外能力，但也没有证据显示它进一步造成了
主要损害。约 7–8 个百分点的格式中立下降在 SFT3000 后已经出现。

GPT-4o Stage3 的高覆盖自由生成准确率为 57.23%，强制二元分数为 59.04%；约 1.81 个百分点
仍与自由生成/回答接口有关，但即使完全绕开解析，距 Base 仍有 7.35 个百分点，故格式只解释
较小部分，不能解释主体退化。
Grok Stage3 同样从高覆盖自由生成的 57.61% 上升到强制二元的 58.86%，但仍比 Base 低
7.53 个百分点，结论一致。

## 3. 为什么同属“病理/医学”仍会域外下降

### 3.1 数据与任务合同并不相同

本地数据审计发现：

| 特征 | PathMMU 训练 4000 | PathVQA test yes/no 3362 |
|---|---:|---:|
| 唯一图像 | 2829 | 839 |
| 每图平均问题数 | 1.41 | 4.01 |
| 问题平均词数 | 39.41 | 7.46 |
| 输出 | 四选一 + 长 `<think>/<answer>` | 短二元开放问答 |
| 精确图像内容重叠 | \multicolumn{2}{c}{0} |

PathVQA 来自病理教材和数字图书馆图像及 caption-derived questions；PathMMU 则汇集论文、
社交媒体、教育视频、图谱和分类数据集，并包含长四选一问答。因此“都叫 pathology”只说明
大的医学主题接近，不保证图像来源、组织层级、问法、标签空间和回答接口同分布。

数据集构造依据：PathVQA 原论文 <https://aclanthology.org/2021.acl-short.90/>；PathMMU 原论文
<https://arxiv.org/abs/2401.16355>。

OmniMedVQA 的四个来源实际是胸部 CT、眼底、皮肤镜和视网膜 OCT，主要不是组织病理切片。
它测到的是更宽的医学视觉域外泛化。

### 3.2 高覆盖后仍存在真实负迁移

OmniMedVQA Base/Stage3 的严格覆盖率分别为 99.81%/99.98%，但准确率为 60.06%/46.30%。
Base 正确而 Stage3 错有 2145 题，反向只有 973 题。分任务下降为：疾病诊断 -5.67、病变分级
-19.03、模态识别 -23.18、其他生物属性 -17.76 个百分点；四个来源全部下降。

目标位置 A/B/C/D 的下降分别为 -21.36/-13.63/-13.47/-4.33 个百分点；Stage3 的预测从
Base 的 A=2633、D=1446 改为 A=2018、D=2070。训练集 A/B/C/D 基本严格均衡，因此这不是
训练标签频率的直接结果，而是决策边界/校准发生变化。模态识别 bad cases 中，Stage3 会把普通
眼底图过度解释为 angiography/ultrasonography，并使用与图像不符的 stroma、cross-section 等
病理化叙述，说明窄域的长推理模板覆盖了 Base 原有的简单视觉类别识别能力。

综合训练轨迹也支持“遗忘首先发生在 SFT”而非“Stage3 单独破坏视觉”：冻结 validation 上
SFT3000 已从 Base 下降，Stage2 没有继续下降，两个 Stage3 反而恢复约 0.8 个百分点；SFT4000
最差且产生强 Yes 偏置。

## 4. 视觉侧训练门禁

当前不启动视觉解冻，原因是：真实 OOD 退化成立，但 PathVQA 的格式中立视觉干预没有显示
Stage3 的视觉依赖明显消失。优先检验减少语言侧遗忘，避免一次性引入多重变量。
而且 vision tower 与 merger 从 Base 到所有正式三阶段 checkpoint 权重均被冻结，因此退化不可能
由视觉编码器自身的参数漂移直接造成；它更可能来自语言解码器对同一批视觉 token 的重新解释。
“源域训练时视觉侧未适配”仍可作为待检验假说，但不能解释为什么未适配的 Base 在域外反而更好。

建议顺序：

1. **B：LM LoRA，vision 和 merger 冻结。** 与当前 full-LM 训练直接比较，检验全参数语言
   更新是否造成遗忘。
2. **C：LM LoRA + 可训练 merger/projector，vision 冻结。** 只有 B 仍明显退化时再做，检验
   跨模态对齐是否需要适配。
3. **D：LM LoRA + merger + vision LoRA 或最后若干视觉块。** 只有 C 仍不足且视觉干预证据
   指向视觉表征问题时再做。
4. 不以 4000 个窄样本直接全量解冻视觉塔；这既有灾难性视觉遗忘风险，也无法区分 projector
   和 vision tower 的贡献。

现有正式 runner 明确禁止 PEFT，并且当前 PEFT 名称过滤还会排除 merger；上述消融需要新建
实验 runner，不能静默修改正式 Stage3 配置。

## 5. 可解释性实验现状

用户已决定暂时搁置继续深挖。当前可用于论文的窄结论是：在冻结的 96-case PathMMU 面板上，
两个 Stage3 模型使用了 image-specific、option-relevant 的空间证据。

- 外观匹配错图使 GPT-4o/Grok 正确选项 margin 分别下降 0.451/0.440，95% CI 均排除 0；
- geometry-matched activation patching 在 decoder layer 20/24 出现方向特异的因果恢复，正确
  激活方向优于等范数置换方向，四组 95% CI 均排除 0；
- option-conditioned black-box RISE 的删除/保留相对面积形状匹配随机对照均显著：GPT-4o
  +0.2407/+0.2097，Grok +0.2205/+0.2241，Holm `p=4e-5`；双方向同一病例同时为正的比例
  为 87.5%/82.3%。

不能声称的内容：raw attention 或 gradient×attention 在逐层严格检验中均未通过；prompt-only
bbox 输出不稳定；外部模型框仍是伪标注且未经病理专家复核。因此不再把单张 attention heatmap
作为真实病灶定位或完整决策逻辑的证明。

## 6. 结果边界

- PathVQA test999/完整 test 已被用于事后诊断，这些均标为 post-hoc sensitivity，不是 untouched
  final test。
- LLM 解析器永远不看参考答案；未解析项按错误计分。
- 强制 Yes/No logits 是格式中立诊断，不替代原论文的自由生成正式指标。
- 当前证据足以否定“域外下降主要是解析问题”，但不足以唯一确定某一个网络模块是原因；视觉
  适配必须通过分阶段消融而不是事后归因。

## 7. 主要机器可读结果

- PathVQA forced binary：
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/reports/pathvqa_forced_binary_test_summary_v2_20260811.json`
- OmniMedVQA paired comparison：
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/reports/omnimed_high_coverage_base_vs_stage3_gpt4o_v2_20260811.json`
- Pathology OOD contract audit：
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/reports/pathology_ood_contract_shift_20260811.json`
- PathVQA Base/GPT/Grok 高覆盖逐题结果：
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/pathvqa_high_coverage_20260811/`
