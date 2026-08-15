# 新机器 Codex 工作区恢复指南（权威入口，2026-08-15）

## 结论与恢复边界

截至本指南生成时，迁移所需的**关键科学资产**已经形成远端副本：Git 代码/协议、关键模型、
逐题评测输出与汇总、可解释性及人工复核候选包、以及不含凭据的 Codex 本地会话快照。

这不是 `/home/dataset-assist-0/czy/wjy` 的逐字节 6.3 TiB 镜像。以下内容有意不上传：

- API key、HF token、Codex `auth.json`；
- optimizer、scheduler、RNG 和 DeepSpeed ZeRO 分片；
- 可重新下载的模型/数据缓存、重复权重和完整虚拟环境二进制；
- 许可证不允许重新分发的第三方基础模型。

因此可以恢复论文实验、继续基于关键 model-only checkpoint 训练并复核全部结果，但不能逐比特
恢复旧 Adam 动量。第三方数据与基础模型按下文固定 repo/revision 重建。

机器可读清单见 `protocol/migration_restore_inventory_20260815.json`，交给新机器 Codex 的启动提示见
`protocol/new_machine_codex_restore_prompt_20260815.txt`。各种数据、文档、模型和记录的本地/远端位置总索引见
`docs/20260815_NEW_MACHINE_ASSET_LOCATION_INDEX.md`。

## 1. 新机器最小准备

建议仍使用相同根路径，能够避免历史 `run_config.json` 中绝对路径的歧义：

```bash
export WJY_WORK_ROOT=/home/dataset-assist-0/czy/wjy
mkdir -p "$WJY_WORK_ROOT"
chmod 700 "$WJY_WORK_ROOT"

export HF_HOME="$WJY_WORK_ROOT/cache/huggingface"
mkdir -p "$HF_HOME"
chmod 700 "$HF_HOME"
```

机器至少需要 `git`、`python3`、可用的 NVIDIA 驱动/CUDA，以及 Hugging Face CLI。分别重新认证；
不要从旧机器复制 token：

```bash
hf auth login
hf auth whoami
```

## 2. 恢复代码、协议、split 与长期实验记忆

```bash
git clone --branch codex/a100-stage3-eval --single-branch \
  https://github.com/Freddie1946/myr1.git \
  "$WJY_WORK_ROOT/myr1"

cd "$WJY_WORK_ROOT/myr1"
git status --short --branch
test -f docs/NEW_MACHINE_CODEX_RESTORE_20260815.md
test -f protocol/migration_restore_inventory_20260815.json
```

Git 中已经包含 PathMMU 的精确 image-disjoint v2 split、数据配比、评测协议、runner、结果目录和
实验决策记录。新 Codex 首先阅读：

```text
docs/NEW_MACHINE_CODEX_RESTORE_20260815.md
docs/LATEST.md
docs/result_catalog_20260815/README.md
docs/result_catalog_20260815/paper_tables.md
docs/20260814_final_nonhuman_results_and_migration_handoff.md
protocol/migration_restore_inventory_20260815.json
```

## 3. 恢复最新 Codex 会话状态

最新 secret-free 在线快照位于 private dataset；最短恢复步骤另见
`docs/QUICK_NEW_MACHINE_CODEX_RESUME_20260815.md`：

```text
Freddie1946/PathVLM-R1-Codex-Private-Snapshots
revision: 7d9a6032c02321c636dc0b07f0bb273223c9e8ec
prefix: snapshots/20260815T103534Z
archive SHA-256: 1f5079557828c1f8e489793c49da31873c423ebd75ba6da73d7db79489387c9a
```

下载与恢复到隔离的 `CODEX_HOME`：

```bash
export RESTORE_STAGE="$WJY_WORK_ROOT/restore_stage"
export NEW_CODEX_HOME="$WJY_WORK_ROOT/.codex-wjy-restored-20260815T103534Z"
mkdir -p "$RESTORE_STAGE/codex"

hf download Freddie1946/PathVLM-R1-Codex-Private-Snapshots \
  --repo-type dataset \
  --revision 7d9a6032c02321c636dc0b07f0bb273223c9e8ec \
  --include 'snapshots/20260815T103534Z/*' \
  --local-dir "$RESTORE_STAGE/codex"

sha256sum "$RESTORE_STAGE/codex/snapshots/20260815T103534Z/"\
"codex_online_snapshot_20260815T103534Z.tar.gz"

mkdir -p "$RESTORE_STAGE/codex-unpacked"
tar -xzf "$RESTORE_STAGE/codex/snapshots/20260815T103534Z/"\
"codex_online_snapshot_20260815T103534Z.tar.gz" \
  -C "$RESTORE_STAGE/codex-unpacked"

mkdir -p "$NEW_CODEX_HOME"
cp -a "$RESTORE_STAGE/codex-unpacked/codex_online_snapshot_20260815T103534Z/"\
"codex-home-snapshot/." "$NEW_CODEX_HOME/"
chmod 700 "$NEW_CODEX_HOME"
export CODEX_HOME="$NEW_CODEX_HOME"
```

快照故意不含认证信息，因此需要执行：

```bash
codex login
codex login status
```

官方 Codex CLI 将会话保存在 `CODEX_HOME`，`codex resume` 可继续已保存的本地会话。进入恢复后的
仓库后运行：

```bash
cd "$WJY_WORK_ROOT/myr1"
codex resume --all
```

从 picker 选择时间最新、原工作目录为 `/home/dataset-assist-0/czy/wjy` 或其 `myr1` 子目录的会话。
如果客户端版本差异导致旧 session 无法直接 resume，则启动一个新会话，并把下面的提示作为首条任务：

```bash
cd "$WJY_WORK_ROOT/myr1"
codex "$(cat protocol/new_machine_codex_restore_prompt_20260815.txt)"
```

边界：此方法恢复 CLI 本地 session/history 和 SQLite 状态；它不等同于把某个网页端对话服务器对象
迁移到另一个账户。快照之后产生的少量对话，应以 Git 中的本指南、`docs/LATEST.md` 和结果清单为准。

## 4. 恢复评测 JSON、rollout 和结果表

主归档为 public + manual-gated dataset：

```text
Freddie1946/PathVLM-R1-Migration-Archive-20260815
final records revision: 5d2485312266ec0670a14494b1fe999d10fe94e1
```

优先使用最终自包含记录快照，它将正式评测、训练过程记录、数据合同和最新人工/可解释性材料统一到
一个经过逐成员 SHA 校验的前缀：

```text
increments/20260815_final_experiment_records_v1
```

直接用 Git 内脚本下载、校验并解压到隔离目录：

```bash
cd "$WJY_WORK_ROOT/myr1"
bash scripts/restore_final_experiment_records_increment_20260815.sh \
  "$WJY_WORK_ROOT/restored_results/final_experiment_records_20260815" --extract
```

历史 Omni 和 GPU 增量仍保留用于时点审计：

```text
revision 82323d5cff454a97c4e137733c4521911c8d5e47:
evaluation_snapshots/omnimed_final_20260815T022447Z
increments/20260815_gpu_evaluations_v1
```

可解释性 20 例与盲法人工生成质量评分 60 例的 private 包：

```text
dataset: Freddie1946/PathVLM-R1-Revision-Evaluation-Results
revision: b2e61cbec20a679c99bba640975f772f033f121a
prefix: snapshots/20260814_interpretability_human_case_selection
```

## 5. 恢复模型

只下载当前工作真正需要的模型，不要一次拉取全部约 183 GB：

| 资产 | HF repo | 固定 revision | 远端大小 |
|---|---|---|---:|
| GPT-4o Stage3 step500/1000/1500 | `Freddie1946/PathVLM-R1-Stage3-GPT4o-n8-parent-seed42-GatedArchive` | `49dea265dbb7607b431d3c74293e4ba2cf80a348` | 49.80 GB |
| LoRA SFT3000 step80 parent | `Freddie1946/PathVLM-R1-LoRA-SFT3000-step80-seed42-GatedArchive` | `4341c46a5e3764bbe6063ee112e83299e344f60e` | 177 MB |
| LoRA SFT4000 control | `Freddie1946/PathVLM-R1-LoRA-SFT4000-Control-seed42-GatedArchive` | `586973a525e912fb1d4b5b2896d7fb3c5221e46b` | 532 MB |
| Full rule-RL n4 step1000 | `Freddie1946/PathVLM-R1-FullRuleRL-n4-step1000-seed42` | `38cee47d749410be8e30f3bc1a65372a9d648bc8` | 16.60 GB |
| Full rule-RL n8 step1000 | `Freddie1946/PathVLM-R1-FullRuleRL-n8-step1000-seed42-GatedArchive` | `b8ba2e1b9be6fe31d44c8415fd5eb0ee60f3a5a1` | 16.60 GB |
| 原始 Full SFT3000 | `Freddie1946/PathVLM-R1-SFT-n3000-seed42-epoch3-GatedArchive` | `86151d3369d412e59a776dd1da7ebe16cb095413` | 16.60 GB |
| 原始 Full SFT4000 control | `Freddie1946/PathVLM-R1-SFT-n4000-control-seed42-epoch2` | `31ecd18b9dc9de5c4118efde586bb625e4a6da06` | 16.60 GB |
| 原始 Stage2 Outcome-GRPO | `Freddie1946/PathVLM-R1-Outcome-GRPO-n1000-seed42-epoch2` | `6496331a597246bda84e8945a3f554c92a3ccc84` | 16.60 GB |
| 旧版 GPT-4o Stage3 selected | `Freddie1946/PathVLM-R1-Process-GRPO-GPT4o-n1000-seed42-epoch2` | `3ade3cffd46b64abc864ed9f271b47632810ec9c` | 16.60 GB |
| 4000 rule-RL checkpoint2500 + Stage2 后继续1000 rule-RL checkpoint1500 | `Freddie1946/PathVLM-R1-P2-Model-Snapshot-Archive-20260815` | `45e55ae634de41fd2adbe9d27907e3083b3f6e55` | 33.20 GB |

通用下载形式：

```bash
hf download REPO_ID --revision REVISION --local-dir LOCAL_DIR
```

P2 仓库中的映射为：`snapshots/snapshot_0001` = 4000 rule-RL checkpoint2500，
`snapshots/snapshot_0002` = Stage2 后继续1000 rule-RL checkpoint1500。

这些均是可加载的 model-only/adapter 快照，不保证原 optimizer state。每次下载后先读取仓库中的
`snapshot_manifest.json` 或 `archive_manifest.json`，再做最小 load smoke。

## 6. 数据集与基础模型重建

精确 PathMMU split 已进入 Git，图像和第三方评测数据按固定上游 revision 下载：

| 资产 | repo/revision | 校验 |
|---|---|---|
| PathMMU | `jamessyx/PathMMU@054e64e56e599e9636024f1471d49ecae4a2784f` | `images.zip` 及 v2 image-content manifest |
| PathVQA | `flaviagiammarino/path-vqa@1685832883334b5bb5beaf4e4b333fdeecaa4ad9` | 三个 test parquet SHA-256 见 completion manifest |
| OmniMedVQA | `foreverbeliever/OmniMedVQA@1ba51c28fc0773bdf7efb8396e5bcfd4227e22da` | `OmniMedVQA.zip` SHA-256 `12245e0f...76adee0` |
| MMMU | `MMMU/MMMU@98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68` | panel/source shard SHA-256 见 retention manifest |
| 基座 | `Qwen/Qwen2.5-VL-7B-Instruct@cc594898137f460bfe9f0759e9844b3ce807cfb5` | `protocol/base_model_manifest.json` |

完整 SHA 和来源在：

```text
protocol/evaluation_assets_preparation_completion_manifest_20260725_004340.json
protocol/base_model_manifest.json
data/pathmmu_image_disjoint_v2/manifest.json
data/pathmmu_image_disjoint_v2/image_content_sha256.json
```

第三方数据/基础模型没有重复上传到我们的 HF 归档；这是许可证与去重边界，不是缺失。若上游访问被撤回，需
从本机做受控点对点迁移，而不能默认公开再分发。

## 7. 环境重建与验收

复制示例环境文件并只改根路径：

```bash
cp "$WJY_WORK_ROOT/myr1/formal_machine/a100_stage3_eval_workspace.env.example" \
   "$WJY_WORK_ROOT/a100_stage3_eval_workspace.env"
source "$WJY_WORK_ROOT/a100_stage3_eval_workspace.env"
```

主 SFT/GRPO 环境使用：

```text
formal_machine/bootstrap_formal_machine.sh
env/sft_overlay_requirements.txt
env/grpo_overlay_requirements.txt
```

各异构基线的隔离环境和固定源码/模型 revision 见：

```text
FORMAL_MACHINE_CODEX_GUIDE.md
manuscript/environment/baseline_repos_and_envs.md
manuscript/environment/reproducible_env_commands.md
```

完成后，先运行硬件/数据/model load smoke，不要立刻续训：

```bash
nvidia-smi
df -h "$WJY_WORK_ROOT"
python3 -m compileall -q "$WJY_WORK_ROOT/myr1/scripts"
```

新 Codex 必须先核对：Git 分支、HF 账号、所有目标 revision、数据 SHA、模型 load、GPU 拓扑与磁盘余量，
再依据 `docs/LATEST.md` 决定下一步。

## 8. 验收状态

已完成并远端核验：

- P1 三项最新关键资产，以及历史 n8/full-SFT3000；
- P2 两个 rule-RL 对照端点；
- 原始 Stage2、Full SFT4000、n4/n8、旧 Stage3 等既有关键仓库；
- 最终 Omni 闭环后的逐题评测归档；
- 可解释性/人工复核候选包；
- 最新 private Codex 在线快照；
- Git 源码、协议、split、结果目录。

未做且不应声称已做：

- 6.3 TiB 工作区逐字节镜像；
- 所有失败/冒烟/小型机制探索 checkpoint；
- optimizer/ZeRO 状态；
- 第三方模型与数据的无条件再分发；
- 当前快照之后未来新增对话与实验的自动增量备份。
