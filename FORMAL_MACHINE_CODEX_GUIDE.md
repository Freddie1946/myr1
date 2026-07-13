# 正式机器 Codex 快速指南

本指南用于让一台没有当前聊天记忆的正式训练机器，通过 Git 恢复项目上下文、盘点共享算力、完成配置，并按门禁继续实验。

## 1. 拉取或更新仓库

首次拉取：

```bash
mkdir -p ~/work && cd ~/work
git clone --branch codex/formal-machine-handoff --single-branch \
  https://github.com/Freddie1946/myr1.git
cd myr1
git status --short --branch
git log -1 --oneline
```

已经拉取过：

```bash
cd /path/to/myr1
git status --short
```

如果工作树不干净，停止并让 Codex报告，不要自动 `reset`、覆盖或删除。工作树干净时再执行：

```bash
git fetch origin
git switch codex/formal-machine-handoff
git pull --ff-only origin codex/formal-machine-handoff
```

## 2. 在仓库根目录启动 Codex

让 Codex 收到下面这段首条指令：

```text
你正在正式训练机器的 myr1 仓库根目录。不要依赖其他机器的聊天记忆。
先完整阅读 AGENTS.md，并严格按其 Mandatory reading order 阅读，包括
FORMAL_MACHINE_CODEX_GUIDE.md、CODEX_START_HERE.md、docs/LATEST.md、
manuscript/README.md、论文、编辑决定、全部审稿意见和 protocol 文档。

阅读后先不要修改文件、下载数据或启动训练。请报告：
1. 当前 commit、branch 和 worktree；
2. GPU/进程占用、驱动、CPU、RAM、swap、磁盘和 CUDA；
3. Conda/Python 与网络状态；
4. 数据、模型、环境和实验各阶段的当前真实状态；
5. 审稿意见对应的实验任务；
6. 发现的冲突、缺口和下一步建议。
不得终止其他用户进程，不得访问 test 做调参。
```

## 3. 算力和环境盘点

Codex 应执行并保存结果，但不得打印任何 token：

```bash
hostname
date -Is
git status --short --branch
git log -1 --oneline
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv
nvidia-smi pmon -c 1
lscpu
free -h
swapon --show
df -hT
nvcc --version || true
command -v conda || true
conda info || true
conda env list || true
python3 --version
```

这是共享服务器：只选择明确空闲且经用户允许的 GPU；不得停止、迁移或干扰其他用户进程。CUDA toolkit 缺失不一定阻止使用 PyTorch wheel，但必须如实记录。

## 4. 配置数据、模型和环境

PathMMU 已获授权时，在当前 shell 安全输入 Hugging Face 只读 token，避免写入 shell 历史和 Git：

```bash
read -rsp 'HF read token: ' HF_TOKEN && export HF_TOKEN && echo
```

复制配置：

```bash
cp formal_machine/formal_machine.env.example formal_machine.env
realpath .
${EDITOR:-nano} formal_machine.env
git check-ignore formal_machine.env
```

至少确认：

- `INSTALL_ROOT` 是空间充足的持久目录。
- `BUNDLE_ROOT` 是当前 `myr1` 的绝对路径。
- `SPLIT_ROOT=$BUNDLE_ROOT/data/pathmmu_image_disjoint_v1`。
- `CONDA_EXE` 是可执行的 Conda 绝对路径或命令。
- `ONLINE=1`、`PATHMMU_AUTO_DOWNLOAD=1`。
- `CUDA_VISIBLE_DEVICES` 和 `NPROC_PER_NODE` 与获准使用的 GPU 一致。

运行配置流程：

```bash
set -o pipefail
mkdir -p ~/pathvlm_setup_logs
bash setup_formal_machine.sh formal_machine.env \
  2>&1 | tee ~/pathvlm_setup_logs/bootstrap_$(date +%Y%m%d_%H%M%S).log
```

该流程负责创建 SFT/GRPO 环境、下载并筛选 PathMMU 图片、下载固定 revision 的 Qwen2.5-VL-7B、生成数据适配器和配置，并写出 `preflight_report.json`。

如果 bootstrap 失败，保留日志并诊断；不要跳过失败步骤或直接训练。成功后：

```bash
source formal_machine.env
python3 -c "import json; p=json.load(open('$INSTALL_ROOT/reports/preflight_report.json')); print('passed=',p['passed']); print(p['gates'])"
```

只有 `passed=True` 才能进入训练门禁。

## 5. 执行任务的顺序

### A. 先补齐并执行正式 SFT smoke

当前仓库会生成 SFT YAML，但正式 SFT 的一键启动、运行清单、保存/重载/续训门禁仍需补齐。Codex 必须先完成这些工程项，不得直接运行长训练：

1. 生成可审计的 SFT launcher、`command.txt`、`run_manifest.yaml` 和日志目录。
2. 用 8 个样本、seed 42、一个 optimizer step 运行 7B 全参数语言模型 SFT。
3. 保存 checkpoint，验证可重新加载，再从该 checkpoint 续训一步。
4. 确认 language model 可训练，vision tower 和 projector 冻结。
5. 记录 loss、gradient norm、样本数、GPU、环境版本、模型/数据/代码哈希。
6. 标记 smoke 为 `formal_result: false`，向用户报告并等待确认。

### B. SFT 正式实验

门禁通过后才依次运行：

1. seed 42：SFT 500/1000/2000/3000 数据规模实验。
2. 只在 validation 上冻结协议。
3. SFT 3000：seed 43 和 44；seed 42 复用规模实验的 3000 运行。
4. 保存每个 seed 的 checkpoint、原始 validation 预测和均值/标准差；不要挑选有利 seed。

### C. 后续阶段

1. Outcome GRPO 必须以对应 seed 的新正式 SFT checkpoint 为 parent。
2. 先通过 parser v2、reward 方差、gradient 和语言/视觉参数 delta 门禁。
3. 再运行 RL 250/500/1000 规模实验和 seeds 42/43/44。
4. Stage 3 Process Reward 的科学定义未敲定，未经用户确认不得正式训练。
5. 最后冻结 prompt、checkpoint、解码和评分，再运行 test；test 永不用于选择。

## 6. Codex 每次汇报格式

每个阶段至少报告：状态、执行命令、输入/输出路径、数据量、seed、GPU、耗时、关键指标、门禁结果、异常和下一步。每个重要计划、失败、修复和完成都新增时间戳文档并更新 `docs/LATEST.md`；不得改写失败历史。

模型权重、图片、环境、缓存、token 和大规模生成输出留在 `INSTALL_ROOT`，不得提交 Git。Git 只提交代码、配置、清单、小型结果和文档。
