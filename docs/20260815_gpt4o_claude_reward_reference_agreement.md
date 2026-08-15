# GPT-4o 训练奖励与 Claude 外部参考一致性

> 这是机器—机器一致性，不是专家验证。60 例包含 36 例 representative 与 24 例 challenge，不能当作总体随机样本。

| 六事件 | Agreement | Cohen κ | GPT positive | Claude positive | Claude-vs-GPT F1 |
|---|---:|---:|---:|---:|---:|
| 图像特征分析 / Image-feature analysis | 93.3% | 0.464 | 56 | 56 | 0.964 |
| 选项排除 / Option elimination | 83.3% | 0.140 | 59 | 49 | 0.907 |
| 医学知识支持 / Medical-knowledge support | 86.7% | 0.262 | 53 | 55 | 0.926 |
| 组织学定义错误 / Histological-definition error | 70.0% | 0.397 | 31 | 35 | 0.727 |
| 逻辑矛盾 / Logical contradiction | 65.0% | 0.335 | 14 | 33 | 0.553 |
| 错误或过时病理标准 / Incorrect/outdated criterion | 80.0% | -0.032 | 11 | 1 | NA |

## 汇总

- 360 个事件判断的 micro agreement：79.7%，case-cluster bootstrap 95% CI [75.3%, 83.9%]；micro κ=0.566。
- 同一 0.4 公式下：GPT-4o 平均奖励 0.775，Claude 0.707，Claude 平均低 0.068。
- reward MAE=0.162；完全相同 38.3%；差值不超过 0.2 的病例 85.0%。
- reward Spearman ρ=0.522，bootstrap 95% CI [0.308, 0.698]。
- representative：agreement 83.8%、κ=0.661；challenge：agreement 73.6%、κ=0.408。

## 解释边界

整体属于中等一致性，而不是高度可互换。图像分析事件一致性高；逻辑矛盾与错误/过时标准的阳性率差异很大，显示 rubric 边界仍依赖 Judge。真正回应审稿人仍需使用第一阶段病理专家评分。
