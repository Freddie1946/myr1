# 在新机器快速恢复 PathVLM-R1 工作区与 Codex 会话

本指南只做两件事：先恢复最新 Git 工作区，再恢复 private HF 中的 secret-free Codex 会话快照。
完成后可用 `codex resume --all` 打开原会话。模型、数据集和大型评测归档按需下载，不应在首次恢复时
一次性拉取。

截至本指南最后一次审计，论文冻结范围内 85/85 项非人工资产检查通过，状态为
`complete_except_human_ratings`。当前没有训练、推理或评测任务在运行；唯一未完成的论文结果单元是
病理专家人工复核及其统计汇总。权威说明见
`docs/20260815_FINAL_MIGRATION_AND_HUMAN_REVIEW_CLOSURE.md`。

## 0. 边界

- 最新会话快照：`Freddie1946/PathVLM-R1-Codex-Private-Snapshots`
- 固定 revision：`7d9a6032c02321c636dc0b07f0bb273223c9e8ec`
- prefix：`snapshots/20260815T103534Z`
- archive SHA-256：`1f5079557828c1f8e489793c49da31873c423ebd75ba6da73d7db79489387c9a`
- Git：`https://github.com/Freddie1946/myr1.git`，branch `codex/a100-stage3-eval`

快照不含 HF token、API key、Codex `auth.json`、缓存或模型权重。新机器必须重新完成 HF 和 Codex
认证。快照是在线一致性时间点，能恢复本地 session/history/SQLite 状态，但不能恢复该时间点之后
仍在旧机器产生的未来消息；这部分以 Git 的 `docs/LATEST.md` 为准。

## 1. 准备相同工作路径并认证 HF

优先保持原绝对路径，避免历史配置中的工作目录失效：

```bash
export WJY_WORK_ROOT=/home/dataset-assist-0/czy/wjy
export HF_HOME="$WJY_WORK_ROOT/cache/huggingface"
mkdir -p "$WJY_WORK_ROOT" "$HF_HOME"
chmod 700 "$WJY_WORK_ROOT" "$HF_HOME"

command -v hf >/dev/null || python3 -m pip install --user -U huggingface_hub
hf auth login
hf auth whoami
```

`hf auth whoami` 应显示有权访问 `Freddie1946` private/gated 仓库的账号。

## 2. 克隆最新代码和实验长期记忆

```bash
git clone --branch codex/a100-stage3-eval --single-branch \
  https://github.com/Freddie1946/myr1.git \
  "$WJY_WORK_ROOT/myr1"

cd "$WJY_WORK_ROOT/myr1"
git status --short --branch
git log -1 --oneline
test -f docs/QUICK_NEW_MACHINE_CODEX_RESUME_20260815.md
```

如果目录已经存在，不要覆盖；先让新机器 Codex检查现有目录和远端关系。

## 3. 下载并核验最新 Codex 会话快照

```bash
export RESTORE_STAGE="$WJY_WORK_ROOT/restore_stage/codex_20260815T103534Z"
mkdir -p "$RESTORE_STAGE/download" "$RESTORE_STAGE/unpacked"

hf download Freddie1946/PathVLM-R1-Codex-Private-Snapshots \
  --repo-type dataset \
  --revision 7d9a6032c02321c636dc0b07f0bb273223c9e8ec \
  --include 'snapshots/20260815T103534Z/*' \
  --local-dir "$RESTORE_STAGE/download"

echo '1f5079557828c1f8e489793c49da31873c423ebd75ba6da73d7db79489387c9a  snapshots/20260815T103534Z/codex_online_snapshot_20260815T103534Z.tar.gz' \
  > "$RESTORE_STAGE/download/CHECKSUMS.sha256"

cd "$RESTORE_STAGE/download"
sha256sum -c CHECKSUMS.sha256

tar -xzf snapshots/20260815T103534Z/codex_online_snapshot_20260815T103534Z.tar.gz \
  -C "$RESTORE_STAGE/unpacked"
```

只有出现 `OK` 才继续。

## 4. 恢复到隔离的 CODEX_HOME

不要覆盖新机器已有的 Codex home。这里使用独立目录：

```bash
export RESTORED_CODEX_HOME="$WJY_WORK_ROOT/.codex-wjy-restored-20260815T103534Z"
test ! -e "$RESTORED_CODEX_HOME"
mkdir -p "$RESTORED_CODEX_HOME"

cp -a "$RESTORE_STAGE/unpacked/codex_online_snapshot_20260815T103534Z/codex-home-snapshot/." \
  "$RESTORED_CODEX_HOME/"
chmod 700 "$RESTORED_CODEX_HOME"
export CODEX_HOME="$RESTORED_CODEX_HOME"

test -d "$CODEX_HOME/sessions"
test -f "$CODEX_HOME/history.jsonl"
find "$CODEX_HOME/sessions" -type f -name '*.jsonl' | wc -l
```

该 `CODEX_HOME` 必须在以后每次启动 Codex 前重新 `export`，或者放入你自己的安全 shell 启动配置。

## 5. 重新登录并恢复会话

```bash
codex login
codex login status

cd "$WJY_WORK_ROOT/myr1"
codex resume --all
```

在 picker 中选择时间最新、原工作目录为 `/home/dataset-assist-0/czy/wjy` 或其 `myr1` 子目录的会话。
恢复后第一条消息建议输入：

```text
请先读取 docs/LATEST.md、docs/QUICK_NEW_MACHINE_CODEX_RESUME_20260815.md、
protocol/migration_restore_inventory_20260815.json 和
protocol/new_machine_codex_restore_prompt_20260815.txt；只做只读恢复验收，汇报当前会话、Git、HF备份、
关键模型与结果资产状态，不要立即训练、评测、删除或调用付费API。
```

## 6. 如果 `codex resume --all` 看不到原会话

先核对当前进程确实使用恢复后的 home：

```bash
echo "$CODEX_HOME"
find "$CODEX_HOME/sessions" -type f -name '*.jsonl' -printf '%TY-%Tm-%Td %TH:%TM %p\n' \
  | sort | tail -n 10
codex --version
```

若 session 文件存在但当前 CLI 版本无法显示，先不要改写这些文件。直接启动新会话并让新 Codex读取
迁移提示和 Git 长期记忆：

```bash
cd "$WJY_WORK_ROOT/myr1"
codex "$(cat protocol/new_machine_codex_restore_prompt_20260815.txt)"
```

这不能在 UI 上续接同一个 session ID，但可以依靠完整实验文档、协议、结果目录和迁移清单恢复工作语境。

## 7. 按需恢复结果与模型

首次恢复会话不需要下载约 183 GB 的全部关键模型。先阅读：

```text
docs/20260815_NEW_MACHINE_ASSET_LOCATION_INDEX.md
docs/NEW_MACHINE_CODEX_RESTORE_20260815.md
docs/LATEST.md
protocol/migration_restore_inventory_20260815.json
```

当前 gated 结果/人工复核增量：

```text
repo: Freddie1946/PathVLM-R1-Migration-Archive-20260815
revision: ef660a6a82a931df3f0c07b7aa05c4c4041bd607
prefix: increments/20260815_bilingual_expert_review_v1
```

评测 JSON/JSONL、metrics、日志及测试口径不是只存在于论文表格中。恢复时按下列顺序核对：

```text
protocol/evaluation_results_priority_backup_20260815.json
protocol/final_revision_hf_backup_20260815.json
protocol/final_revision_assets_audit_20260815.json
protocol/results_material_index_v2_hf_backup_20260815.json
protocol/migration_restore_inventory_20260815.json
```

其中 `evaluation_results_priority_backup_20260815.json` 记录了 1,561 个文件、约 1.28 GB
未压缩结果的首轮远端快照；后续 GPU 评测增量和最终修订增量分别补齐其后的 JSON/JSONL。
新机器 Codex必须依据 manifest 中的 SHA-256 和固定 revision 下载，不能仅凭相似目录名猜测模型或测试口径。

### 下载完整实验记录、评测 JSON/JSONL 与测试口径

下面不是只下载汇总表，而是依次恢复：首轮 1,561 文件快照、后续 GPU 评测增量、最终 OmniMedVQA
归档、最后补齐的论文评测，以及人工复核/解释性材料。可直接执行：

```bash
export RESULTS_RESTORE="$WJY_WORK_ROOT/restored_results/pathvlm_20260815"
mkdir -p "$RESULTS_RESTORE"

hf download Freddie1946/PathVLM-R1-Migration-Archive-20260815 \
  --repo-type dataset --revision 4c1999d2164b4cab4f0f633a65efe01b482c61f9 \
  --include 'evaluation_snapshots/20260815T020000Z/*' \
  --local-dir "$RESULTS_RESTORE/migration_archive"

hf download Freddie1946/PathVLM-R1-Migration-Archive-20260815 \
  --repo-type dataset --revision 3576750e12541133c9432a8a17362d5d35913e94 \
  --include 'increments/20260815_gpu_evaluations_v1/*' \
  --local-dir "$RESULTS_RESTORE/migration_archive"

hf download Freddie1946/PathVLM-R1-Migration-Archive-20260815 \
  --repo-type dataset --revision 82323d5cff454a97c4e137733c4521911c8d5e47 \
  --include 'evaluation_snapshots/omnimed_final_20260815T022447Z/*' \
  --local-dir "$RESULTS_RESTORE/migration_archive"

hf download Freddie1946/PathVLM-R1-Migration-Archive-20260815 \
  --repo-type dataset --revision 9527a11a19cf9024f02eba7d40bfe1eee4cd37c6 \
  --include 'increments/20260815_final_revision_closure_v1/*' \
  --local-dir "$RESULTS_RESTORE/migration_archive"

hf download Freddie1946/PathVLM-R1-Migration-Archive-20260815 \
  --repo-type dataset --revision ef660a6a82a931df3f0c07b7aa05c4c4041bd607 \
  --include 'increments/20260815_bilingual_expert_review_v1/*' \
  --local-dir "$RESULTS_RESTORE/migration_archive"
```

首轮与 Omni 最终归档各含 `ARCHIVE_MANIFEST.json`、`ARCHIVE_VERIFICATION.json` 和一个 tar.gz。
先核对 verification 与 tar.gz 的 SHA-256，再解压到独立目录；不要覆盖 Git 仓库或新机器已有结果。
GPU/最终增量已经按模型与评测合同分层保存 `predictions.jsonl`、`metrics.json`、`run_config.json` 和日志。
下载后以 `docs/result_catalog_20260815/result_lineage.json` 建立模型—checkpoint—prompt—parser—dataset
对应关系；同名指标但合同不同的结果不得合并。

专家评分历史不从 HF 在线读取。每位专家在自己的分发包内使用
`expert_submissions/<Reviewer ID>/` 保存和恢复，并将导出的 `complete60.zip` 交给负责人。负责人使用：

```bash
python3 scripts/import_expert_reward_exports.py /path/to/returned/*.zip \
  --output-dir "$WJY_WORK_ROOT/pathvlm_revision_eval_a100/human_review/expert_submissions_20260815"
```

专家结果与训练期 GPT-4o、Claude、ROI 删除对照及 Stage2/Stage3 盲评的具体比较关系见
`docs/20260815_HUMAN_REVIEW_DISTRIBUTION_AND_ANALYSIS_PATHS.md` 和
`docs/20260815_human_review_packet_instructions.md`。向专家分发材料时，同时提供
`docs/20260815_EXPERT_REVIEWER_QUICK_START.md`。

完整模型仓库、固定 revision、大小、第三方数据来源和许可证边界见权威恢复指南。新 Codex应先列出
本次任务真正需要的资产及磁盘预算，经确认后再逐项下载。

本指南本身及最终迁移闭环文件另有一个小型 manual-gated 镜像：

```text
repo: Freddie1946/PathVLM-R1-Migration-Archive-20260815
revision: 27c792f4da3524f18d870c90f749aa8aece011eb
prefix: increments/20260815_final_migration_guide_v1
verification: protocol/final_migration_guide_hf_backup_20260815.json
```
