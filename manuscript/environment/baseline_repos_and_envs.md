# 大修基线代码与环境记录

更新时间：2026-07-05

本文件记录为回应审稿人关于“病理专用模型/医学 VLM 基线不足”的意见，已拉取或已评估的候选基线代码、环境依赖、是否适合直接用于 PathMMU 多选 VQA，以及下一步执行建议。

## 1. 存放位置

代码统一存放在：

```text
/home/wjy/revision_runs/04_baselines/repos/
```

文档统一存放在：

```text
/home/wjy/Majorrevision/env_doc/
```

配置和输出建议使用：

```text
/home/wjy/revision_runs/04_baselines/configs/
/home/wjy/revision_runs/04_baselines/outputs/
```

## 2. 已拉取仓库

| 名称 | 仓库 | 当前 commit | 本地目录 | 大小 | 适合任务 | 当前判断 |
|---|---|---|---|---:|---|---|
| CONCH | `https://github.com/mahmoodlab/CONCH.git` | `141cc09c7d4ff33d8eda562bd75169b457f71a62` | `/home/wjy/revision_runs/04_baselines/repos/CONCH` | 3.8M | 病理图文基础模型/检索/零样本分类 | 重要相关工作，可作为 pathology VLM/CLIP-like 基线候选；不一定能直接做多选 VQA |
| UNI | `https://github.com/mahmoodlab/UNI.git` | `42715efc11722a496e0a67f3369505a8f277206c` | `/home/wjy/revision_runs/04_baselines/repos/UNI` | 12M | 病理视觉表征模型 | 重要病理 foundation model；通常不是对话 VQA 模型，适合写作定位或特征/检索类对比 |
| PLIP | `https://github.com/PathologyFoundation/plip.git` | `f010f3d0bef20f4e8cc64cc26c301cbd26305fa1` | `/home/wjy/revision_runs/04_baselines/repos/PLIP` | 1.5M | pathology CLIP 图文相似度 | 可作为 CLIP-like 零样本/选项匹配基线候选 |
| LLaVA-Med | `https://github.com/microsoft/LLaVA-Med.git` | `30697ca50b5c29a8e955c99330b259776aef27b9` | `/home/wjy/revision_runs/04_baselines/repos/LLaVA-Med` | 1.6G | 医学对话/VQA | 最接近可直接做医学 VQA 的新增基线，但环境较重，模型权重约 7B，需要单独环境 |

未成功访问：

| 名称 | 尝试仓库 | 结果 | 备注 |
|---|---|---|---|
| PathChat | `https://github.com/mahmoodlab/PathChat.git` | `Repository not found` | PathChat 可能没有公开 GitHub 代码，需通过论文、官网、HuggingFace 或作者页面确认；仍应在 related work 中讨论 |

## 3. 依赖摘要

### 3.1 CONCH

配置文件：`CONCH/pyproject.toml`

核心依赖：

```text
python >= 3.9
torch >= 2.0.1
torchvision
transformers
tokenizers
numpy
scikit-learn
timm >= 0.9.8
regex
ftfy
h5py
pandas
```

权重：README 指向 HuggingFace `MahmoodLab/conch`。

许可：CC BY-NC 4.0，注意非商业限制。

建议用途：

- 作为 pathology vision-language foundation model 的相关工作重点讨论。
- 可尝试将多选项文本作为候选 caption，用 image-text similarity 做零样本多选。
- 若用于正式表格，必须说明其不是生成式 VQA 模型，比较方式与 VLM 对话模型不同。

### 3.2 UNI

配置文件：`UNI/setup.py`

核心依赖：

```text
torch >= 2.0.1
torchvision
timm == 0.9.8
numpy < 2
pandas
scikit-learn
tqdm
transformers
```

权重：README 指向 MahmoodLab/UNI 系列模型，需要查看 HuggingFace 或申请条款。

许可：CC BY-NC 4.0，注意非商业限制。

建议用途：

- 作为 pathology foundation model 的相关工作与定位对比。
- 不适合直接作为多选 VQA 生成模型。
- 可考虑：图像 encoder + 文本选项 embedding/linear probe/retrieval，但这需要设计公平协议。

### 3.3 PLIP

配置文件：`PLIP/setup.py`、`PLIP/requirements.txt`

注意：当前 `requirements.txt` 为空，需要根据 README 的 HuggingFace 用法安装：

```text
transformers
Pillow
torch
torchvision
numpy
```

模型：`vinid/plip`

建议用途：

- 可作为 pathology CLIP-like 选项匹配基线。
- 使用方式：将每个选项转为文本 prompt，与图像算 similarity，选择最高分。
- 优点是轻量、容易跑；缺点是无法生成推理链，也不是专门的多选医学推理模型。

### 3.4 LLaVA-Med

配置文件：`LLaVA-Med/pyproject.toml`

核心依赖：

```text
python >= 3.8
transformers == 4.36.2
tokenizers >= 0.15.0
sentencepiece == 0.1.99
accelerate == 0.21.0
peft == 0.4.0
bitsandbytes == 0.41.0
pydantic < 2, >= 1
gradio == 3.35.2
gradio_client == 0.2.9
httpx == 0.24.0
einops == 0.6.1
timm == 0.9.12
openai == 1.12.0
```

可选依赖：

```text
train: deepspeed==0.9.5, ninja, wandb
eval: azure-ai-ml, datasets, fire, opencv-python, openpyxl==3.1.2, pillow==9.4.0, python-Levenshtein, rich, streamlit==1.29.0, typer[all], word2number
```

模型：`microsoft/llava-med-v1.5-mistral-7b`

显存：README 提到 RTX 4090/3090 等 24GB 显存可尝试 `--num-gpus 2` 运行。

建议用途：

- 作为医学 VLM 直接多选 VQA 基线，优先级较高。
- 需要单独 conda 环境，避免污染当前 LLaMA-Factory/Huatuo 环境。
- 运行 PathMMU 需要写一个 adapter，把 PathMMU 的 image/problem/options 转成 LLaVA-Med prompt，并解析 A/B/C/D。

## 4. 推荐环境规划

不要把所有基线装进同一个环境。建议：

```text
conda env: pathvlm_clip_baselines
用途：CONCH / UNI / PLIP
python: 3.10
主要依赖：torch, torchvision, transformers, timm, scikit-learn, pandas

conda env: llava-med
用途：LLaVA-Med v1.5
python: 3.10
主要依赖：按 LLaVA-Med pyproject.toml 固定版本安装
```

原因：

- LLaVA-Med 锁定较旧的 `transformers==4.36.2`、`accelerate==0.21.0`、`peft==0.4.0`。
- CONCH/UNI/PLIP 对依赖要求更轻，不应被 LLaVA-Med 的旧依赖牵制。

## 5. 下一步建议

### 5.1 先建轻量 CLIP-like baseline

优先级：高。

候选：PLIP、CONCH。

目标：不下载巨型 LLM，只用 image-text similarity 在 PathMMU 多选题上做零样本选项匹配。

输出：

```text
/home/wjy/revision_runs/04_baselines/outputs/plip_pathmmu_zeroshot.json
/home/wjy/revision_runs/04_baselines/outputs/conch_pathmmu_zeroshot.json
```

意义：

- 回应审稿人要求增加 pathology-specific baseline。
- 即使性能不高，也能说明病理图文基础模型在多选推理任务上的定位差异。

### 5.2 再跑 LLaVA-Med VQA baseline

优先级：高，但环境更重。

步骤：

1. 创建独立环境 `llava-med`。
2. 安装 `pip install -e /home/wjy/revision_runs/04_baselines/repos/LLaVA-Med`。
3. 下载或在线加载 `microsoft/llava-med-v1.5-mistral-7b`。
4. 写 PathMMU adapter。
5. 先跑 20 条 smoke test，再跑 500 条或 1385 条。

### 5.3 PathChat 处理方式

优先级：写作必须，实验视可获取性决定。

当前 GitHub 地址不可访问。建议：

- 在 related work 中详细讨论 PathChat。
- 查 HuggingFace/官方页面是否有 demo 或权重。
- 如果无法运行，明确说明“not directly evaluated due to unavailable public inference code/weights”，但仍做定位对比。

## 6. 风险提示

1. CONCH/UNI/PLIP 并非生成式多选 VQA 模型，不能与 PathVLM-R1 做完全同质比较。
2. LLaVA-Med 是医学通用 VLM，不是病理专用模型，但可作为强医学 VQA 基线。
3. 新增 baseline 前必须先固定 PathMMU canonical split，否则结果仍可能被数据泄漏问题影响。
4. 所有新增结果必须记录数据文件 SHA256、模型权重版本和运行命令。
