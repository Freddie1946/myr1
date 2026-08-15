# GPT-4o 训练奖励与 Claude 外部参考的最终过程分差

> 主终点是用同一 0.4 公式换算后的逐病例过程分差；六事件一致率只作辅助定位。这是机器—机器比较，不是专家验证。60 例包含 36 例 representative 与 24 例 challenge，不能当作总体随机样本。

## 最终过程分差（主结果）

- GPT-4o 平均过程分：0.775；Claude：0.707。
- Claude−GPT-4o 平均有符号分差：-0.068，病例 bootstrap 95% CI [-0.127, -0.013]。
- MAE：0.162，95% CI [0.120, 0.207]；RMSE：0.233。
- 最终过程分完全相同：38.3%；绝对分差不超过 0.2：85.0%。

## 六事件分歧定位（辅助结果）

| 六事件 | Agreement | Cohen κ | GPT positive | Claude positive | Claude-vs-GPT F1 |
|---|---:|---:|---:|---:|---:|
| 图像特征分析 / Image-feature analysis | 93.3% | 0.464 | 56 | 56 | 0.964 |
| 选项排除 / Option elimination | 83.3% | 0.140 | 59 | 49 | 0.907 |
| 医学知识支持 / Medical-knowledge support | 86.7% | 0.262 | 53 | 55 | 0.926 |
| 组织学定义错误 / Histological-definition error | 70.0% | 0.397 | 31 | 35 | 0.727 |
| 逻辑矛盾 / Logical contradiction | 65.0% | 0.335 | 14 | 33 | 0.553 |
| 错误或过时病理标准 / Incorrect/outdated criterion | 80.0% | -0.032 | 11 | 1 | NA |

- 360 个事件判断的 micro agreement：79.7%，case-cluster bootstrap 95% CI [75.3%, 83.9%]；micro κ=0.566。
- reward Spearman ρ=0.522，bootstrap 95% CI [0.308, 0.698]。
- representative：agreement 83.8%、κ=0.661；challenge：agreement 73.6%、κ=0.408。

## 解释边界

最终分差显示 Claude 整体给分略低，但病例级差异不可忽略；事件表只用于定位差异来自哪些 rubric 边界。该结果仍是机器—机器比较，不是病理专家验证。
