# PathVQA Yes/No 主合同与 A/B 补充诊断合同（2026-08-14）

## 最终报告决定

PathVQA 二元题保留原始 `Yes/No` 答案接口作为主评测合同。原因是该接口最接近
数据集原始任务定义，也保留了与已经完成的外部基线结果相同的答案语义。精确 prompt
仍分为历史 short-answer 和当前结构化 reasoning + Yes/No 两个子合同；只有相同子合同
才能声称严格的 prompt-matched 可比性，跨子合同结果必须显式标注。

固定 `A = Yes, B = No` 的 PathMMU 式结构化生成合同保留为补充接口诊断，不替代、
覆盖或与原始 Yes/No 主分数混合。该合同用于检验模型在 PathMMU 风格训练后出现的
PathVQA 性能变化中，有多少可由任务接口、输出模板或解析适配解释。

本决定取代 `20260812_pathvqa_fixed_ab_evaluation_contract.md` 中“A/B 为主指标”的报告
优先级，但不使任何已经完成的 A/B 原始预测或指标失效。

## 固定合同

### 主合同族：原始 Yes/No 生成

- 输入保持原始二元问题，不把答案改写为多选题。
- 模型自由生成简短回答；评分使用冻结的、target-blind 的 Yes/No 语义解析。
- 报告语义准确率、解析率、生成上限命中率以及未解析数量。
- 历史外部基线保留 `legacy_v1_64` short-answer 子合同；当前主线保留冻结的
  `domain_think_answer_v2_2048` 子合同。主表保留 Yes/No 答案语义，但必须显示子合同，
  不能把两者伪装成完全相同的 prompt。

### 补充合同：固定 A/B 结构化生成

```text
<question>
Options:
A) Yes
B) No
First give image-grounded reasoning in <think>...</think>, then return
<answer>A) Yes</answer> or <answer>B) No</answer>.
```

- 映射永远固定为 `A = Yes, B = No`，不做选项反转。
- 这是完整接口变换，而不是对生成文本事后替换字符串。
- 报告两套互不混淆的结果：
  1. strict A/B：字母、语义映射和结构化标签均满足合同；
  2. semantic A/B：忽略标签瑕疵后，生成内容在语义上是否表达正确的 Yes/No。
- A/B 结果只进入补充分析，不用于替代外部基线 Yes/No 分数，也不用于回选训练
  checkpoint。

## 配对解释规则

两种合同必须对同一批病例逐题配对，并报告四格转换：

| 原始 Yes/No | A/B | 解释边界 |
|---|---|---|
| 对 | 对 | 对接口稳健 |
| 错 | 对 | 支持接口/模板不匹配解释，称为 A/B recovery |
| 对 | 错 | A/B 接口造成损害，不能用于解释原始下降 |
| 错 | 错 | 接口变换未恢复；需进一步区分视觉语义、知识、推理或共同解析失败 |

主诊断量为配对净恢复：

```text
net interface recovery = count(Yes/No wrong -> A/B correct)
                       - count(Yes/No correct -> A/B wrong)
```

同时报告 McNemar 的不一致对计数和 case-bootstrap 区间。若只有 strict A/B 改善而
semantic A/B 不改善，证据仅支持格式遵循改善；只有语义正确率也恢复，才支持二元
决策接口漂移解释。

## 可声称与不可声称的结论

可以声称：

- 相同病例在 PathMMU 风格接口下恢复了多少答案；
- 性能下降中存在多大比例的输出/任务接口敏感性；
- 结构化格式遵循和医学语义正确性分别发生了什么变化。

不能仅凭 A/B 改善声称：

- 模型的病理视觉能力没有下降；
- 所有 Yes/No 错误都只是解析错误；
- A/B 分数可与旧 Yes/No 基线直接横向比较。

两种合同均错误的病例仍需图像、理由和答案级 bad-case 分析。视觉语义是否真正退化，
应由开放式描述、诊断多选和图像干预等独立 probe 判断。

## 已有 512 例配对证据

冻结的 PathVQA validation 512 例面板已经在相同病例上比较过
`domain_think_answer_v2_2048` 与 `pathmmu_ab_v1_2048`。以下 bootstrap 使用固定 seed 42、
20,000 次 case resampling；面板每例对应不同图像。

| 模型 | Yes/No | A/B | 错→对 | 对→错 | 净恢复 | 差值 95% CI | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| L-r16 SFT80 parent | 59.18% | 64.65% | 82 | 54 | +28 / +5.47 pp | [+0.98, +9.96] pp | 0.0203 |
| Full-RL n4 step500 | 58.98% | 62.50% | 69 | 51 | +18 / +3.52 pp | [-0.78, +7.62] pp | 0.1203 |
| Full-RL n8 step500 | 57.23% | 63.28% | 86 | 55 | +31 / +6.05 pp | [+1.56, +10.55] pp | 0.0113 |

这已经支持“任务/输出接口敏感性是 PathVQA 下降的一部分解释”：三组点估计方向一致，
其中 parent 和 n8 的配对差异达到常用显著性阈值。但它不是完整解释，因为仍有
127、141、133 例分别在两种合同下都错误；这些共同错误不能归咎于 A/B/Yes-No 格式，
仍可能包含真实视觉语义、知识或推理退化。

源汇总：
`pathvlm_revision_eval_a100/runs/pathvqa_choice_parent_n4_n8_20260812/complete_comparison.json`
（SHA-256 `3d8d8e8558047d57379c7432b808282470226c177e0714d0d7c3670c1279129a`）。
