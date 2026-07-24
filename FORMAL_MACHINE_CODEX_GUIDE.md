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

也可以使用持久的用户级登录（推荐用于需要重跑的 bootstrap）：

```bash
/home/wjy/.conda/envs/wjy/bin/hf auth login
/home/wjy/.conda/envs/wjy/bin/hf auth whoami
```

使用与 PathMMU 网页审批相同的账号和只读 token；若询问是否保存为 Git credential，选择
`n`。凭据只能保存在用户级 Hugging Face 缓存或当前 shell，禁止写入仓库和
`formal_machine.env`。

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
- `SPLIT_ROOT=$BUNDLE_ROOT/data/pathmmu_image_disjoint_v2`。
- `CONDA_EXE` 是可执行的 Conda 绝对路径或命令。
- `ONLINE=1`、`PATHMMU_AUTO_DOWNLOAD=1`。
- `CUDA_VISIBLE_DEVICES` 和 `NPROC_PER_NODE` 与获准使用的 GPU 一致。
- 当前 `YiwuServer` 的正式环境选择 PyTorch 2.6.0 cu124 wheel；驱动 580.142 支持该
  runtime。系统 `nvcc 11.5` 不作为 PyTorch runtime，训练使用 Torch AdamW，避免依赖
  DeepSpeedCPUAdam 的本机 JIT 编译。

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

## 5. 迁移外部评测数据与病理基线环境

这一节是可选的修订实验迁移流程，与第 4 节的正式 SFT/GRPO bootstrap 完全隔离。只有在
用户明确批准外部评测资产准备后才执行。当前冻结范围和源版本见：

- `protocol/evaluation_assets_preparation_manifest_20260725_000657.json`
- `protocol/pathology_baseline_source_manifest_20260725_000657.json`
- `protocol/pathology_clip_requirements_20260725.txt`

先设置机器相关路径。不要把 token 写入命令、脚本或 `.env`：

```bash
cd /path/to/myr1
export EVAL_ROOT="${EVAL_ROOT:-$HOME/pathvlm_revision_eval}"
export HF_BIN="${HF_BIN:-$HOME/.conda/envs/wjy/bin/hf}"
export CONDA_EXE="${CONDA_EXE:-/opt/miniconda3/bin/conda}"
mkdir -p "$EVAL_ROOT"/{downloads,datasets,models,envs,reports,sources}
"$HF_BIN" auth whoami
df -h "$EVAL_ROOT"
```

外部资产不进入 Git。迁移前后至少保留 450 GiB 可用空间；空间不足时停止并报告，不自动
清理。迁移只准备数据/环境，不运行推理、训练或 Stage 3，也不占用 GPU。

### 5.1 下载固定版本的数据

下载 PathMMU 的小型元数据，不重复下载已经由正式 bootstrap 管理的图片归档：

```bash
"$HF_BIN" download jamessyx/PathMMU \
  data.json instructions.md socialpath_mapping.json README.md construct_pathcls.py \
  --repo-type dataset \
  --revision 054e64e56e599e9636024f1471d49ecae4a2784f \
  --local-dir "$EVAL_ROOT/downloads/pathmmu_metadata"
```

下载固定提交的 OmniMedVQA 官方归档：

```bash
"$HF_BIN" download foreverbeliever/OmniMedVQA OmniMedVQA.zip README.md \
  --repo-type dataset \
  --revision 1ba51c28fc0773bdf7efb8396e5bcfd4227e22da \
  --local-dir "$EVAL_ROOT/downloads/omnimedvqa" \
  --max-workers 2
```

只抽取已批准的 Chest CT、ISIC2020、Retinal OCT-C8 和 Diabetic Retinopathy：

```bash
unzip -q -n "$EVAL_ROOT/downloads/omnimedvqa/OmniMedVQA.zip" \
  'OmniMedVQA/Images/Chest CT Scan/*' \
  'OmniMedVQA/Images/ISIC2020/*' \
  'OmniMedVQA/Images/Retinal OCT-C8/*' \
  'OmniMedVQA/Images/Diabetic Retinopathy/*' \
  'OmniMedVQA/QA_information/Open-access/Chest CT Scan.json' \
  'OmniMedVQA/QA_information/Open-access/ISIC2020.json' \
  'OmniMedVQA/QA_information/Open-access/Retinal OCT-C8.json' \
  'OmniMedVQA/QA_information/Open-access/Diabetic Retinopathy.json' \
  'OmniMedVQA/README.md' \
  -d "$EVAL_ROOT/datasets/omnimedvqa_ood_v1"
```

当前已验证的归档 SHA-256 和真实结构是：

- `OmniMedVQA.zip`:
  `12245e0f99afbc7d6f70e4ca3c2e5a7979a01816cd8159ae34539f4aa76adee0`
- Chest CT: 382 个唯一引用图像、871 个 QA。
- ISIC2020: 1,499 个唯一引用图像、1,580 个 QA。
- Retinal OCT-C8: 3,224 个唯一引用图像、4,016 个 QA。
- Diabetic Retinopathy: 1,966 个唯一引用图像、2,051 个 QA。

PathVQA 使用数据卡明确说明来自作者 2023-02-15 更新分发版本的规范化 Hugging Face
test split，只下载三个 test parquet，不下载 train/validation：

```bash
"$HF_BIN" download flaviagiammarino/path-vqa \
  README.md \
  data/test-00000-of-00003-e9adadb4799f44d3.parquet \
  data/test-00001-of-00003-7ea98873fc919813.parquet \
  data/test-00002-of-00003-1628308435019820.parquet \
  --repo-type dataset \
  --revision 1685832883334b5bb5beaf4e4b333fdeecaa4ad9 \
  --local-dir "$EVAL_ROOT/datasets/pathvqa_test_v1" \
  --max-workers 2
```

三个 shard 应分别为 2,240/2,240/2,239 行，共 6,719 QA、858 个唯一图像内容。对应
SHA-256 依次是：

- `533175be08e87b9ccf2fc06e2f82e58bb2e9bf2299a4a138cf5311f1d26ac157`
- `84bde926245c4b136de6ef48b197a0f3f177b3fbcb03b7f3cbfcfcb212f20c50`
- `ae676f41a7ddfc85be8503a731c719e1608da37ddc981de42a10dd2d55a8ee87`

### 5.2 固定基线源码

下面四个提交是准备阶段实际审计的版本。迁移时 checkout 提交而不是浮动 `main`：

```bash
git clone --filter=blob:none --no-checkout \
  https://github.com/mahmoodlab/CONCH.git "$EVAL_ROOT/sources/CONCH"
git -C "$EVAL_ROOT/sources/CONCH" fetch --depth 1 origin \
  141cc09c7d4ff33d8eda562bd75169b457f71a62
git -C "$EVAL_ROOT/sources/CONCH" checkout --detach \
  141cc09c7d4ff33d8eda562bd75169b457f71a62

git clone --filter=blob:none --no-checkout \
  https://github.com/mahmoodlab/UNI.git "$EVAL_ROOT/sources/UNI"
git -C "$EVAL_ROOT/sources/UNI" fetch --depth 1 origin \
  42715efc11722a496e0a67f3369505a8f277206c
git -C "$EVAL_ROOT/sources/UNI" checkout --detach \
  42715efc11722a496e0a67f3369505a8f277206c

git clone --filter=blob:none --no-checkout \
  https://github.com/PathologyFoundation/plip.git "$EVAL_ROOT/sources/plip"
git -C "$EVAL_ROOT/sources/plip" fetch --depth 1 origin \
  f010f3d0bef20f4e8cc64cc26c301cbd26305fa1
git -C "$EVAL_ROOT/sources/plip" checkout --detach \
  f010f3d0bef20f4e8cc64cc26c301cbd26305fa1

git clone --filter=blob:none --no-checkout \
  https://github.com/microsoft/LLaVA-Med.git "$EVAL_ROOT/sources/LLaVA-Med"
git -C "$EVAL_ROOT/sources/LLaVA-Med" fetch --depth 1 origin \
  30697ca50b5c29a8e955c99330b259776aef27b9
git -C "$EVAL_ROOT/sources/LLaVA-Med" checkout --detach \
  30697ca50b5c29a8e955c99330b259776aef27b9
```

### 5.3 创建 PLIP/CONCH/UNI 隔离环境

不要在正式 `sft`、`grpo` 或用户的通用 `wjy` 环境里安装这些依赖：

```bash
"$CONDA_EXE" create -p "$EVAL_ROOT/envs/pathology_clip" \
  python=3.10 pip=25.1 setuptools=78.1 wheel=0.45 -y

"$EVAL_ROOT/envs/pathology_clip/bin/pip" install \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1

"$EVAL_ROOT/envs/pathology_clip/bin/pip" install \
  -r protocol/pathology_clip_requirements_20260725.txt

"$EVAL_ROOT/envs/pathology_clip/bin/pip" install --no-deps \
  -e "$EVAL_ROOT/sources/CONCH" \
  -e "$EVAL_ROOT/sources/UNI"
```

不要对固定提交 `f010f3d...` 的 PLIP 仓库执行 `pip install -e`：其上游 `setup.py`
声明不存在的 `plip/` 包，而源码实际是顶层 `plip.py`，会在 `egg_info` 阶段失败。
评测适配器优先使用 PLIP README 同样给出的
`transformers.CLIPModel/CLIPProcessor.from_pretrained("vinid/plip")`；若仅需审计上游
wrapper，则临时设置 `PYTHONPATH="$EVAL_ROOT/sources/plip"` 后 `import plip`，不要修改上游
源码伪装修复。

CONCH 和 UNI 权重需要各自在 Hugging Face 页面接受许可；不要把 token 作为 Python 参数、
命令行参数或源码常量。PLIP/CONCH 仅作为图文匹配基线；公开 CONCH 不含生成解码器。
UNI 是视觉编码器，只做定位/表征对比，不伪装成生成式 VQA 基线。LLaVA-Med 的独立环境和
约 15 GB 权重仍需后续单独门禁。

`Freddie1946` 账号最初只能读模型卡，权重鉴权返回 403；用户完成申请后，2026-07-25
对固定版本的 CONCH/UNI `pytorch_model.bin` 鉴权 HEAD 均已通过。迁移到新机器后仍需由
账号本人分别在以下页面接受许可，并重新做 HEAD 检查；文件总大小约 0.80 GB 和 1.21 GB：

- `https://huggingface.co/MahmoodLab/CONCH`
- `https://huggingface.co/MahmoodLab/UNI`

迁移完成后只做 CPU 导入和一致性检查：

```bash
for d in CONCH UNI plip LLaVA-Med; do
  git -C "$EVAL_ROOT/sources/$d" rev-parse HEAD
done
sha256sum "$EVAL_ROOT/downloads/omnimedvqa/OmniMedVQA.zip"
"$EVAL_ROOT/envs/pathology_clip/bin/pip" check
CUDA_VISIBLE_DEVICES="" "$EVAL_ROOT/envs/pathology_clip/bin/python" -c \
  "import torch, torchvision, transformers, timm, datasets; import conch, uni; print('core imports PASS')"
PYTHONPATH="$EVAL_ROOT/sources/plip" CUDA_VISIBLE_DEVICES="" \
  "$EVAL_ROOT/envs/pathology_clip/bin/python" -c \
  "import plip; from transformers import CLIPModel, CLIPProcessor; print('PLIP imports PASS')"

"$EVAL_ROOT/envs/pathology_clip/bin/python" scripts/audit_external_eval_assets.py \
  --omnimed-root "$EVAL_ROOT/datasets/omnimedvqa_ood_v1/OmniMedVQA" \
  --pathvqa-root "$EVAL_ROOT/datasets/pathvqa_test_v1" \
  --formal-content-manifest data/pathmmu_image_disjoint_v2/image_content_sha256.json \
  --output "$EVAL_ROOT/reports/external_eval_asset_audit_20260725.json"
```

将 `pip freeze`、源提交、数据哈希、文件/QA 数量、缺失路径和精确内容重叠报告写入
`$EVAL_ROOT/reports/`，并把小型摘要/清单写回仓库。任何外部评测开始前，必须先冻结
prompt、解码、评分、统计和污染审计协议；test 仍不得用于选择。

### 5.4 迁移论文原有的全部基线

论文 Table II 的十四个基线和 Table IV 额外的 HuatuoGPT-Vision-7B 均为必跑项，完整
集合及当前候选模型 ID 见
`protocol/all_manuscript_baselines_manifest_20260725_011528.json`。PLIP、CONCH、UNI
属于额外审稿回应组，不能替代这十五个原有基线。

不要直接为所有模型执行浮动版本的 `pip install` 或批量下载 `main`。迁移顺序固定为：

1. 先为每一行解析准确模型/API ID、固定 revision、许可和精确下载字节数；
2. 把五个托管模型的 endpoint、版本/日期、SDK 和服务商写进独立 manifest；凭据只能
   通过仓库外的权限为 `600` 的环境文件或服务商凭据存储注入；
3. 分别创建 Qwen、Meta Llama Vision、DeepSeek-VL2、MedGemma、InternVL3、
   Huatuo/LLaVA 和 API client 环境，不能污染正式 `sft`/`grpo` 环境；
4. 每个环境先做 CPU 导入、`pip check` 和最小处理器加载，再在共享服务器 GPU 门禁后做
   单样本 smoke；
5. 每个模型保存 `pip freeze`、模型/代码哈希、prompt、图像预处理、解码参数、原始返回、
   parser 版本、错误和重试事件。

建议的迁移目录为：

```text
$EVAL_ROOT/
  envs/{qwen_vl,llama32_vision,deepseek_vl2,medgemma,internvl3,huatuo_llava,api_clients}/
  models/<provider>--<model>--<revision>/
  configs/manuscript_baselines/
  outputs/manuscript_baselines/
  reports/manuscript_baselines/
```

原论文 Table II 的 500 个历史样本 ID 尚未恢复，因此不能把新跑结果写成精确历史复现。
统一 rerun 数据集、prompt 和 parser 必须先在 validation 上冻结；正式 PathMMU test 在
所有十五个基线完成配置与 smoke 前保持密封，且每个模型只做预声明的最终评测。

存储是硬门禁：90B BF16 权重本身约 180 GB，完整本地集合可能需要数百 GB。先汇总官方
文件列表和字节数，再决定存储位置；未获批准不得批量下载。不能为了节省空间静默改用
量化权重、不同参数规模或第三方转换版本。若历史 API 模型已下线/重定向，仍需运行经
批准的当前替代版本，但必须明确标记为 contemporary rerun，不能冒充原模型。

## 6. 执行任务的顺序

### A. 执行正式 SFT smoke

仓库已经提供 `scripts/launch_formal_sft_smoke.sh`。它会生成可审计的
`command.txt`、`run_manifest.yaml`、环境快照和日志，运行一步、保存并重新加载
checkpoint，再续训一步并比较语言/视觉张量。bootstrap/preflight 通过且 GPU 获得明确授权后：

```bash
source formal_machine.env
source "$INSTALL_ROOT/FORMAL_PATHS.env"
export PATHVLM_SMOKE_CUDA_VISIBLE_DEVICES="1,2,3,4"  # 改为实际获准的卡
export PATHVLM_SMOKE_NPROC_PER_NODE=4
bash scripts/launch_formal_sft_smoke.sh
```

成功标准：8 个 SFT 样本、seed 42、7B 全参数语言模型；vision/projector 冻结；第一步
checkpoint 可加载；可从第一步续训到第二步；两个更新均有 language tensor delta 且
visual tensor 严格相同。该 smoke 必须标记 `formal_result: false`，完成后向用户报告并等待确认。

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

## 7. Codex 每次汇报格式

每个阶段至少报告：状态、执行命令、输入/输出路径、数据量、seed、GPU、耗时、关键指标、门禁结果、异常和下一步。每个重要计划、失败、修复和完成都新增时间戳文档并更新 `docs/LATEST.md`；不得改写失败历史。

模型权重、图片、环境、缓存、token 和大规模生成输出留在 `INSTALL_ROOT`，不得提交 Git。Git 只提交代码、配置、清单、小型结果和文档。
