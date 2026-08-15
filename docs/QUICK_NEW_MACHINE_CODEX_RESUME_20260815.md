# 在新机器快速恢复 PathVLM-R1 工作区与 Codex 会话

本指南只做两件事：先恢复最新 Git 工作区，再恢复 private HF 中的 secret-free Codex 会话快照。
完成后可用 `codex resume --all` 打开原会话。模型、数据集和大型评测归档按需下载，不应在首次恢复时
一次性拉取。

## 0. 边界

- 最新会话快照：`Freddie1946/PathVLM-R1-Codex-Private-Snapshots`
- 固定 revision：`a7b576e8450bc676ec93cc51db6ae592b9493d47`
- prefix：`snapshots/20260815T100317Z`
- archive SHA-256：`d10f7b3e72c2f7a75770b5d283efc2c3cc9209c002faa1c9917cb305f2198ee9`
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
export RESTORE_STAGE="$WJY_WORK_ROOT/restore_stage/codex_20260815T100317Z"
mkdir -p "$RESTORE_STAGE/download" "$RESTORE_STAGE/unpacked"

hf download Freddie1946/PathVLM-R1-Codex-Private-Snapshots \
  --repo-type dataset \
  --revision a7b576e8450bc676ec93cc51db6ae592b9493d47 \
  --include 'snapshots/20260815T100317Z/*' \
  --local-dir "$RESTORE_STAGE/download"

echo 'd10f7b3e72c2f7a75770b5d283efc2c3cc9209c002faa1c9917cb305f2198ee9  snapshots/20260815T100317Z/codex_online_snapshot_20260815T100317Z.tar.gz' \
  > "$RESTORE_STAGE/download/CHECKSUMS.sha256"

cd "$RESTORE_STAGE/download"
sha256sum -c CHECKSUMS.sha256

tar -xzf snapshots/20260815T100317Z/codex_online_snapshot_20260815T100317Z.tar.gz \
  -C "$RESTORE_STAGE/unpacked"
```

只有出现 `OK` 才继续。

## 4. 恢复到隔离的 CODEX_HOME

不要覆盖新机器已有的 Codex home。这里使用独立目录：

```bash
export RESTORED_CODEX_HOME="$WJY_WORK_ROOT/.codex-wjy-restored-20260815T100317Z"
test ! -e "$RESTORED_CODEX_HOME"
mkdir -p "$RESTORED_CODEX_HOME"

cp -a "$RESTORE_STAGE/unpacked/codex_online_snapshot_20260815T100317Z/codex-home-snapshot/." \
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
docs/NEW_MACHINE_CODEX_RESTORE_20260815.md
docs/LATEST.md
protocol/migration_restore_inventory_20260815.json
```

当前 gated 结果/人工复核增量：

```text
repo: Freddie1946/PathVLM-R1-Migration-Archive-20260815
revision: 2c90a34185c383054acebb80336201e99fbc96fd
prefix: increments/20260815_bilingual_expert_review_v1
```

完整模型仓库、固定 revision、大小、第三方数据来源和许可证边界见权威恢复指南。新 Codex应先列出
本次任务真正需要的资产及磁盘预算，经确认后再逐项下载。
