# P1 关键模型备份清单与 HF 重叠审计（2026-08-15）

P1 使用 `public + manual gated` 仓库存放只含可加载权重的关键快照；不含 optimizer、scheduler 和 RNG state。原始清单约 183.32 GB；经 HF SHA256 重叠审计和用户确认后，活动队列已收缩为尚未远端备份的 50.51 GB。

| P1 项目 | 快照 | 大小（GB） | 目标仓库 | 与现有 HF 权重的关系 |
|---|---|---:|---|---|
| Historical full-SFT3000 | epoch3/step1125 | 16.60 | `PathVLM-R1-SFT-n3000-seed42-epoch3-GatedArchive` | 已完成；原 private 权重已不在当前远端列表中，因此当前无实体重复 |
| Full rule-RL n8 | step1000 | 16.60 | `PathVLM-R1-FullRuleRL-n8-step1000-seed42-GatedArchive` | 已完成；同名 private 仓仅约 2 KB、没有权重，无实体重复 |
| Full rule-RL n4 | step1000 | 16.60 | `PathVLM-R1-FullRuleRL-n4-step1000-seed42-GatedArchive` | 与现有 private n4 四个权重分片 SHA256 全部相同；已按用户要求终止重复上传并移出活动队列 |
| GPT-4o Stage3（n8 parent） | step500/1000/1500 | 49.80 | `PathVLM-R1-Stage3-GPT4o-n8-parent-seed42-GatedArchive` | 现有 private GPT-4o 是另一条旧 run；与这三个 checkpoint 均 0/4 分片匹配，不是重复 |
| Selected LoRA-SFT3000 | step80 | 0.177 | `PathVLM-R1-LoRA-SFT3000-step80-seed42-GatedArchive` | 当前 HF 无对应 adapter 备份 |
| LoRA-SFT4000 control | final + step11/22 adapters | 0.532 | `PathVLM-R1-LoRA-SFT4000-Control-seed42-GatedArchive` | 当前 HF 无对应 adapter 备份 |
| Grok-4.3 Stage3 | step500/1000/1500 | 49.80 | `PathVLM-R1-Stage3-Grok43-seed42-GatedArchive` | 用户决定不迁移 Grok；已移出活动队列 |
| Historical outcome-GRPO | epoch2/step1000 | 16.60 | `PathVLM-R1-Outcome-GRPO-n1000-seed42-epoch2-GatedArchive` | 与现有 private Outcome-GRPO 四个分片全部相同；已移出活动队列 |
| Historical full-SFT4000 control | checkpoint250 | 16.60 | `PathVLM-R1-Full-SFT4000-Control-seed42-GatedArchive` | 与现有 private SFT4000 四个分片全部相同；已移出活动队列 |

## 当前活动队列

1. GPT-4o Stage3 step500/1000/1500：49.80 GB；
2. selected LoRA-SFT3000 step80：0.177 GB；
3. LoRA-SFT4000 control：0.532 GB。

活动队列合计约 50.51 GB；Historical full-SFT3000 与 full-rule-RL n8 已在本轮开始时完成，不会再次上传。

## 处置边界

- 这些重复是有意进行的 `private -> public/manual-gated` 迁移，不是 P1 误选了无关模型。
- 只有 gated 目标完成、远端文件数/总字节数/SHA256 校验通过后，private 副本才成为可删除候选。
- 本清单不授权删除任何本地或远端数据；删除前必须再次取得用户确认。

## OmniMedVQA 边界

旧 OmniMedVQA 结果不再进入论文修订表或完成度判断。正式口径仅接受新统一合同 `omnimed_domain_think_answer_v4_1024`；尚未完成的模型显示为 `RUNNING`，不得用旧 64-token 或旧 parser 结果补位。
