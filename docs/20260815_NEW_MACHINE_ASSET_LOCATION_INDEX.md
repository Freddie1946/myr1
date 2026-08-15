# 新机器 Codex：PathVLM-R1 数据、文档与实验记录位置总索引

> 更新日期：2026-08-15。本文回答“什么资产在哪里”。恢复步骤见
> `docs/QUICK_NEW_MACHINE_CODEX_RESUME_20260815.md` 和
> `docs/NEW_MACHINE_CODEX_RESTORE_20260815.md`。任何同名结果都必须结合固定 HF revision、
> `run_config.json`、模型 checkpoint 和评测合同判断，不能只看目录名或最终 accuracy。

## 1. 最先读取的权威入口

新机器 Codex 克隆 Git 后，按以下顺序读取：

1. `docs/LATEST.md`：当前文档入口和实验状态；
2. `docs/20260815_FINAL_WORKSPACE_BACKUP_AUDIT.md`：终极工作区盘点、缺漏补传与安全边界；
3. `docs/20260815_NEW_MACHINE_ASSET_LOCATION_INDEX.md`：本资产位置索引；
4. `docs/20260815_reviewer_response_manuscript_revision_and_complete_tables.md`：审稿回复、正文替换段落和完整实验表的统一底稿；
5. `docs/result_catalog_20260815/paper_tables.md`：单独抽出的论文数据表；
6. `docs/result_catalog_20260815/result_lineage.json`：模型—checkpoint—数据—prompt—parser—结果的机器可读谱系；
7. `protocol/migration_restore_inventory_20260815.json`：远端资产和固定 revision 总清单；
8. `docs/20260815_FINAL_MIGRATION_AND_HUMAN_REVIEW_CLOSURE.md`：非人工实验闭合状态和剩余人工工作；
9. `protocol/new_machine_codex_restore_prompt_20260815.txt`：恢复后首次审计提示。

Git 权威仓库：

```text
https://github.com/Freddie1946/myr1.git
branch: codex/a100-stage3-eval
```

Git 保存代码、split 定义、运行脚本、protocol/manifest、结果索引和科学决策记录；大模型权重、逐题大体积
JSONL 和 Codex 会话快照不依赖 Git 保存。

## 2. 原机器本地目录分别保存什么

统一根目录为 `/home/dataset-assist-0/czy/wjy`。

| 原路径 | 内容 | 迁移后的处理 |
|---|---|---|
| `myr1/` | Git 代码、脚本、数据划分、协议、文档和结果索引 | 必须从 Git 克隆 |
| `pathvlm_r1_v1_a100/runs/` | SFT、rule-RL、Stage3 训练目录、checkpoint、trainer state 和日志 | 关键 model-only checkpoint 从独立 HF 模型仓库恢复；不恢复 optimizer/ZeRO 分片 |
| `pathvlm_revision_eval_a100/runs/` | PathMMU、PathVQA、OmniMedVQA、MMMU、基线、Judge 和可解释性运行结果 | 从 Migration Archive 的评测快照/增量恢复 |
| `pathvlm_revision_eval_a100/reports/` | bad-case、重评分、费用、传输和阶段性审计报告 | 主要内容在评测归档及 Git 文档中 |
| `pathvlm_revision_eval_a100/human_review/` | 奖励复核、ROI、生成质量材料、外部参考和专家回收记录 | 从 gated 人工复核增量恢复；正式专家新结果另行增量备份 |
| `pathvlm_r1_v1_a100/data/` | PathMMU image-disjoint v2 split、SFT/RL 配比与训练数据派生物 | 核心 split/manifest 已进 Git；图像按上游固定 revision 重建 |
| `pathvlm_revision_eval_a100/datasets/` | PathVQA/Omni/MMMU 评测集及冻结 panel | 数据本体按上游 revision 重建；冻结 panel/manifest 在 Git/结果归档 |
| `pathvlm_r1_v1_a100/models/` | 基座和部分本地模型副本 | 不整体迁移；按模型固定 revision 下载 |
| `pathvlm_revision_eval_a100/models/` | 外部本地基线模型 | 不整体迁移；按上游 repo/revision 和许可重建 |
| 两个工作区的 `envs/` | SFT/GRPO 与异构基线环境 | 不复制环境二进制；按 Git 中环境脚本重建 |
| `backup_archives/` | 人工复核分发包、负责人包和本机迁移打包产物 | 关键包已上传 HF；旧的重复归档不必全部恢复 |
| `migration_backups/` | 迁移过程中的本地快照和审计产物 | 以 HF verification/protocol 为准按需恢复 |
| `rewrite_docs/` | 原论文 PDF 和审稿意见原件 | 已进入 private migration supplement；最新版修订内容以 Git 统一 Markdown 为准 |
| `.codex-wjy/` | 当前机器 Codex session/history/SQLite 和本地配置 | 由 private secret-free Codex 快照恢复；认证信息需重新登录 |
| `.secrets/` | API/HF 等本机凭据 | **从未上传；禁止迁移或提交** |

## 3. 文档、论文修订与实验表在哪里

本地 Git 内的权威文件：

```text
docs/20260815_reviewer_response_manuscript_revision_and_complete_tables.md
docs/result_catalog_20260815/paper_tables.md
docs/result_catalog_20260815/result_lineage.json
docs/20260815_paper_experiment_final_audit_and_tables.md
docs/20260815_human_review_packet_instructions.md
docs/20260815_EXPERT_REVIEWER_QUICK_START.md
```

统一审稿回复/正文修订/完整表的精确 HF 副本：

```text
dataset: Freddie1946/PathVLM-R1-Migration-Archive-20260815
revision: ef660a6a82a931df3f0c07b7aa05c4c4041bd607
path: increments/20260815_bilingual_expert_review_v1/documents/
      reviewer_response_manuscript_revision_and_complete_tables.md
sha256: aee4f48a0e9f652edd79ee86343177c170b197b42a20acca95f51065f4165923
```

独立实验表的精确 HF 副本：

```text
revision: fdc819c9c84fbc72dff2e40a14e74b4bf37d6940
path: increments/20260815_external_review_and_pathvqa_matching_v1/documents/paper_tables.md
sha256: 42eeff4d551b970485619b202e3bce7b62a0b4a71ad8a7bf5397cf99c8f9455b
```

## 4. 逐题评测 JSON、metrics、日志和测试口径在哪里

主仓库均为 `Freddie1946/PathVLM-R1-Migration-Archive-20260815`，类型为 dataset，权限为
public + manual gated。新机器应优先恢复下面的“最终实验记录快照”；它是自包含记录层，不需要先拼接
旧增量。旧快照继续保留，用于核对当时的备份历史和不可变 revision。

```text
revision: 5d2485312266ec0670a14494b1fe999d10fe94e1
prefix: increments/20260815_final_experiment_records_v1
archive: pathvlm_final_experiment_records_20260815T123626Z.tar.gz
archive sha256: 0c00a19b827ec8d3ec0c539ca6741507355397569d4e851e5eb60c66ae6cb625
payload: 49,245 files / 10,226,535,291 bytes before compression
restore: scripts/restore_final_experiment_records_increment_20260815.sh
audit: docs/20260815_FINAL_EXPERIMENT_RECORDS_BACKUP_AUDIT.md
```

历史增量如下：

| 固定 revision | 前缀 | 内容 |
|---|---|---|
| `4c1999d2164b4cab4f0f633a65efe01b482c61f9` | `evaluation_snapshots/20260815T020000Z` | 首轮 1,561 个评测/实验记录文件，约1.28 GB未压缩 |
| `3576750e12541133c9432a8a17362d5d35913e94` | `increments/20260815_gpu_evaluations_v1` | 后续 GPU 评测增量 |
| `82323d5cff454a97c4e137733c4521911c8d5e47` | `evaluation_snapshots/omnimed_final_20260815T022447Z` | 最终统一 OmniMedVQA 逐题结果、日志、manifest |
| `9527a11a19cf9024f02eba7d40bfe1eee4cd37c6` | `increments/20260815_final_revision_closure_v1` | 最后补齐的 PathVQA/Omni/论文评测与结果目录 |
| `b182e8de003412c0d68ea9d31fa53a81d99bc8ba` | `increments/20260815_results_material_index_v2` | 结果谱系、材料目录及负责人复核包 |
| `fdc819c9c84fbc72dff2e40a14e74b4bf37d6940` | `increments/20260815_external_review_and_pathvqa_matching_v1` | 外部复核、PLIP/CONCH PathVQA statement matching 与更新表 |
| `ef660a6a82a931df3f0c07b7aa05c4c4041bd607` | `increments/20260815_bilingual_expert_review_v1` | 双语专家材料、外部参考、统计脚本和最新版统一修订 Markdown |
| `06ccabf7292d6014eef60c9adf2498611a3d87c1` | `increments/20260815_expert_reviewer_quick_start_v1` | 含简明指南的最新专家分发包 |
| `5d2485312266ec0670a14494b1fe999d10fe94e1` | `increments/20260815_final_experiment_records_v1` | 排除 smoke 的最终自包含实验记录、逐文件 SHA、审计和双重校验 |

每次评测至少同时读取：

```text
run_config.json       # 模型、checkpoint、数据、prompt、生成和 parser 合同
predictions.jsonl     # 逐题输入摘要、原始输出、解析结果和正确性
metrics.json          # 汇总指标及样本数
*.log                 # 运行/重试/异常记录
manifest/verification # 文件大小、SHA-256 与归档状态
```

不要把 PathVQA 自由 Yes/No、A=Yes/B=No、forced logits 或旧 parser 结果当作同一口径；不要把 OmniMedVQA
旧 legacy/aligned/official 目录仅凭名称合并。最终合同见统一修订 Markdown 和 `result_lineage.json`。

## 5. 关键模型 checkpoint 在哪里

| 模型/用途 | HF 模型仓库 | 固定 revision |
|---|---|---|
| Stage3 GPT-4o step500/1000/1500 | `Freddie1946/PathVLM-R1-Stage3-GPT4o-n8-parent-seed42-GatedArchive` | `49dea265dbb7607b431d3c74293e4ba2cf80a348` |
| LoRA SFT3000 step80 parent | `Freddie1946/PathVLM-R1-LoRA-SFT3000-step80-seed42-GatedArchive` | `4341c46a5e3764bbe6063ee112e83299e344f60e` |
| LoRA SFT4000 control | `Freddie1946/PathVLM-R1-LoRA-SFT4000-Control-seed42-GatedArchive` | `586973a525e912fb1d4b5b2896d7fb3c5221e46b` |
| Full rule-RL n4 step1000 | `Freddie1946/PathVLM-R1-FullRuleRL-n4-step1000-seed42` | `38cee47d749410be8e30f3bc1a65372a9d648bc8` |
| Full rule-RL n8 step1000（Stage3 parent） | `Freddie1946/PathVLM-R1-FullRuleRL-n8-step1000-seed42-GatedArchive` | `b8ba2e1b9be6fe31d44c8415fd5eb0ee60f3a5a1` |
| 历史 Full SFT3000 | `Freddie1946/PathVLM-R1-SFT-n3000-seed42-epoch3-GatedArchive` | `86151d3369d412e59a776dd1da7ebe16cb095413` |
| Full SFT4000 control | `Freddie1946/PathVLM-R1-SFT-n4000-control-seed42-epoch2` | `31ecd18b9dc9de5c4118efde586bb625e4a6da06` |
| 历史 Stage2 Outcome-GRPO | `Freddie1946/PathVLM-R1-Outcome-GRPO-n1000-seed42-epoch2` | `6496331a597246bda84e8945a3f554c92a3ccc84` |
| 旧 Stage3 GPT-4o selected | `Freddie1946/PathVLM-R1-Process-GRPO-GPT4o-n1000-seed42-epoch2` | `3ade3cffd46b64abc864ed9f271b47632810ec9c` |
| 4000 rule-RL step2500 + Stage2 后继续1000 rule-RL step1500 | `Freddie1946/PathVLM-R1-P2-Model-Snapshot-Archive-20260815` | `45e55ae634de41fd2adbe9d27907e3083b3f6e55` |

P2 映射：`snapshots/snapshot_0001` 是 4000 rule-RL checkpoint2500；`snapshot_0002` 是 Stage2 后继续
1000 rule-RL checkpoint1500。权重以 model-only/adapter 快照为主；没有承诺恢复 optimizer、scheduler、
RNG 或 DeepSpeed ZeRO 分片。下载后先核对各仓库 manifest 并做 load smoke。

## 6. 数据集、split 与基础模型在哪里

核心 PathMMU image-disjoint v2 split 和数据配比定义在 Git：

```text
data/pathmmu_image_disjoint_v2/
protocol/evaluation_assets_preparation_completion_manifest_20260725_004340.json
protocol/base_model_manifest.json
```

第三方数据/基础模型没有重复上传到我们的归档，应按固定上游 revision 重建：

| 资产 | 上游 repo | revision |
|---|---|---|
| PathMMU | `jamessyx/PathMMU` | `054e64e56e599e9636024f1471d49ecae4a2784f` |
| PathVQA | `flaviagiammarino/path-vqa` | `1685832883334b5bb5beaf4e4b333fdeecaa4ad9` |
| OmniMedVQA | `foreverbeliever/OmniMedVQA` | `1ba51c28fc0773bdf7efb8396e5bcfd4227e22da` |
| MMMU | `MMMU/MMMU` | `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68` |
| Qwen2.5-VL-7B-Instruct 基座 | `Qwen/Qwen2.5-VL-7B-Instruct` | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |

若某第三方 gated 资产无法重新获取，先核对许可证和原账号权限，再做受控点对点迁移；不要擅自公开上传。

## 7. 人工复核与可解释性材料在哪里

本地入口：

```text
/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/human_review/
/home/dataset-assist-0/czy/wjy/backup_archives/human_review_distribution_20260815_v4/
/home/dataset-assist-0/czy/wjy/human_review_returns/reward/
```

最新专家分发包：

```text
revision: 06ccabf7292d6014eef60c9adf2498611a3d87c1
path: increments/20260815_expert_reviewer_quick_start_v1/human_review/
      pathvlm_human_review_bilingual_assisted_with_guide_20260815.tar.gz
```

负责人解盲/统计包：

```text
revision: b182e8de003412c0d68ea9d31fa53a81d99bc8ba
path: increments/20260815_results_material_index_v2/human_review/
      pathvlm_human_review_owner_complete_20260815.tar.gz
```

详细入口：

```text
docs/20260815_HUMAN_REVIEW_DISTRIBUTION_AND_ANALYSIS_PATHS.md
docs/20260815_human_review_packet_instructions.md
docs/20260815_EXPERT_REVIEWER_QUICK_START.md
scripts/import_expert_reward_exports.py
scripts/analyze_human_reward_score_differences.py
```

可解释性候选、行为删除、RISE、activation patching 和外部 ROI 复核的本地原始运行位于
`pathvlm_revision_eval_a100/runs/` 对应目录；小型人工材料另在 private dataset：

```text
Freddie1946/PathVLM-R1-Revision-Evaluation-Results
revision: b2e61cbec20a679c99bba640975f772f033f121a
prefix: snapshots/20260814_interpretability_human_case_selection
```

## 8. 环境、代码和异构基线如何定位

不要复制整个 conda/micromamba 环境。重建入口：

```text
formal_machine/bootstrap_formal_machine.sh
formal_machine/a100_stage3_eval_workspace.env.example
env/sft_overlay_requirements.txt
env/grpo_overlay_requirements.txt
FORMAL_MACHINE_CODEX_GUIDE.md
manuscript/environment/baseline_repos_and_envs.md
manuscript/environment/reproducible_env_commands.md
```

原机器的 `a100_stage3_eval_workspace.env` 含机器路径，应从 example 重新生成。API key、HF token、Codex
`auth.json` 和 `.secrets/` 均未备份，新机器必须重新认证。

## 9. Codex 会话和长期实验记忆在哪里

private、经过 token 脱敏和 JSONL 解析校验的会话快照：

```text
dataset: Freddie1946/PathVLM-R1-Codex-Private-Snapshots-Clean-20260815
revision: 4c9e0778895d8a06fda45f391bbe64e7699fd634
prefix: snapshots/20260815T134000Z
archive: codex_online_snapshot_20260815T134000Z.tar.gz
sha256: 6e87ab9182a705e31e35fc11e30606f2d5d7f88e3ccc16d5021219b7b345c4ab
```

它包含该时间点之前的 session/history 及必要 SQLite 状态，不含 `auth.json`、日志数据库、shell 快照
或凭据；58,175 行 JSONL 全部可解析。旧仓库 `PathVLM-R1-Codex-Private-Snapshots` 因历史快照可能
含 token 副本，禁止用于恢复。若 `codex resume --all` 不能显示旧 session，使用
`protocol/new_machine_codex_restore_prompt_20260815.txt` 启动新会话恢复语境。

原论文与审稿意见原件的 private 补充包：

```text
dataset: Freddie1946/PathVLM-R1-Private-Migration-Supplement-20260815
revision: 9f2377c03827d60e38a1fa6d5e03fdf250cfafdd
prefix: snapshots/20260815_final_workspace_v1
archive sha256: 55a3e1e5a19dadc8470eee15df72927050fdc23701bb56408787cd59bef3a4d8
```

## 10. 新机器建议的恢复落点

```text
$WJY_WORK_ROOT/myr1/                              # Git
$WJY_WORK_ROOT/restored_results/pathvlm_20260815/ # HF 评测归档，不覆盖 Git
$WJY_WORK_ROOT/pathvlm_r1_v1_a100/models/         # 按需恢复关键模型
$WJY_WORK_ROOT/pathvlm_revision_eval_a100/        # 重建评测数据、环境和新运行
$WJY_WORK_ROOT/restore_stage/                     # 下载/校验临时区
$WJY_WORK_ROOT/.codex-wjy-restored-*/             # 隔离的恢复后 CODEX_HOME
```

先恢复 Git、会话和结果索引，再按任务下载模型/数据。不要一次下载全部模型；先计算磁盘预算并核对所需
revision。解压归档到独立目录，不覆盖新机器已有结果。

## 11. 完整性与科学边界

截至本索引，论文范围内非人工资产已通过 85/85 审计；最终实验记录快照另通过 49,245/49,245
逐成员 SHA 校验。尚待完成的是病理专家评分和收到评分后的统计。
备份是关键科学资产迁移，不是 6.3 TiB 工作区的逐字节镜像。以下内容有意不归档：凭据、可重新下载的
缓存、重复模型树、全部失败/冒烟 checkpoint、optimizer/scheduler/RNG/ZeRO 状态，以及无权重新分发的
第三方资产。

任何恢复验收先做只读核对：Git HEAD、HF revision、manifest/SHA、模型 load、数据样本数和评测合同；
未经用户确认，不立即训练、调用付费 API、删除旧资产或覆盖已有目录。
