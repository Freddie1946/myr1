# PathVLM-R1 终极工作区备份审计

审计时间：2026-08-15 21:42（Asia/Shanghai）

## 结论

工作区中的关键科学资产现已具备远端副本：Git 代码与协议、10组关键模型、49,245个非-smoke
实验/评测记录、全部训练 rollout/reward/Judge/费用与配置、数据 split/冻结 panel、人工复核和可解释性
材料、原论文与审稿意见原件，以及最新可恢复的 Codex 会话。最终实验记录归档以后，训练和评测记录
目录没有新增相关文件。

这仍不是 2.4 TB 工作区的逐字节镜像。大量空间属于 optimizer/ZeRO 状态、探索 checkpoint、环境、缓存
和可重新下载的第三方模型；这些均按用户已批准的备份边界有意排除。

## 全工作区分类

| 类别 | 本地规模 | 文件数 | 迁移处理 |
|---|---:|---:|---|
| `pathvlm_r1_v1_a100` | 2,170,378,063,294 bytes | 190,166 | 关键模型单独备份；正式记录进入最终记录归档 |
| `pathvlm_revision_eval_a100` | 243,396,636,247 bytes | 384,768 | 逐题结果、报告、人工与解释性材料进入最终记录归档 |
| `backup_archives` | 10,926,685,131 bytes | 5,787 | 远端归档的本地副本，恢复不要求整体复制 |
| `.codex-wjy` | 885,217,594 bytes | 5,519 | 使用新的脱敏 private 快照恢复 |
| Git 工作树 | 68,841,756 bytes | 3,880 | GitHub 分支恢复 |
| `rewrite_docs` | 1,453,032 payload bytes | 3 | 新增 private migration supplement |

训练工作区中共有96个 `checkpoint-*` 目录。289个权重文件合计965,709,901,832 bytes；549个
optimizer/scheduler/RNG/ZeRO 状态文件合计1,062,959,671,620 bytes。后者没有长期恢复价值且已被用户
明确排除；探索权重也不因“本地存在”自动升级为关键模型。

## 已验证远端资产

### 关键模型

`protocol/migration_restore_inventory_20260815.json` 中10个模型仓库共183,317,331,972 bytes。
本次重新查询每个固定 revision，10/10 revision 与远端总字节数均匹配。

### 最终实验记录

```text
dataset: Freddie1946/PathVLM-R1-Migration-Archive-20260815
revision: 5d2485312266ec0670a14494b1fe999d10fe94e1
prefix: increments/20260815_final_experiment_records_v1
payload: 49,245 files / 10,226,535,291 bytes
archive: 3,737,087,357 bytes
sha256: 0c00a19b827ec8d3ec0c539ca6741507355397569d4e851e5eb60c66ae6cb625
```

该归档已经逐成员双重校验，覆盖训练 rollout、奖励事件、Judge 响应、费用、trainer 配置和日志，
评测 predictions/metrics/合同，数据配比与冻结 panel，以及最新人工复核/可解释性材料。

### 原论文和审稿原件

```text
dataset: Freddie1946/PathVLM-R1-Private-Migration-Supplement-20260815
visibility: private
revision: 9f2377c03827d60e38a1fa6d5e03fdf250cfafdd
prefix: snapshots/20260815_final_workspace_v1
archive sha256: 55a3e1e5a19dadc8470eee15df72927050fdc23701bb56408787cd59bef3a4d8
```

包含原论文 PDF、编辑/审稿 PDF 和审稿 Markdown；旧 Stage3 工作草案按用户决定未纳入。

### 最新 Codex 会话

```text
dataset: Freddie1946/PathVLM-R1-Codex-Private-Snapshots-Clean-20260815
visibility: private
revision: 4c9e0778895d8a06fda45f391bbe64e7699fd634
prefix: snapshots/20260815T134000Z
archive: 152,363,544 bytes
sha256: 6e87ab9182a705e31e35fc11e30606f2d5d7f88e3ccc16d5021219b7b345c4ab
```

新快照排除了 `auth.json`、日志数据库和 shell 快照，并对复制的文本/session执行 token 脱敏。
67个文件、284,237,643 payload bytes通过秘密扫描，58,175行 JSONL 全部解析成功。

## 重要安全发现

旧 private 仓库 `Freddie1946/PathVLM-R1-Codex-Private-Snapshots` 中的历史快照虽然排除了
`auth.json`，但 session/shell/log 状态仍含当前 OpenRouter 与 AIGCBest key 的精确副本。因此：

- 旧仓库不得用于恢复；
- 新的 clean 仓库已经提供安全替代；
- 旧仓库当前 HEAD `e8afdb8f07220ea99bbbbd498a97d09b1870673a` 已增加
  `SECURITY_DEPRECATED_DO_NOT_RESTORE.md` 警告；
- 应轮换这两个 API key；
- 删除旧 HF 仓库及本地冗余污染快照属于破坏性安全处置，等待用户明确批准。

## 有意不备份

- optimizer、scheduler、RNG、DeepSpeed ZeRO；
- smoke、失败、小规模数据配比、视觉适配和机制探索权重；
- 用户已排除的 Kimi/Grok checkpoint；
- 缓存、临时文件、重复模型树和环境二进制；
- 可按固定 revision 重建的第三方数据和基础模型；
- 无权重新分发的第三方资产；
- 已被用户忽略的旧 Stage3 工作草案。

环境依赖由 Git 中 bootstrap、overlay requirements、可复现环境命令，以及最终记录归档内的运行环境
快照和基线 `pip freeze` 恢复，不复制14 GB训练环境和47 GB异构评测环境。

## 唯一后续增量

论文范围内只剩真实病理专家评分及收到评分后的统计。专家返回文件一旦产生，应立即建立新的 private
增量备份；当前浏览器功能测试提交不是正式专家数据。
