# 可复用基线环境命令草案

更新时间：2026-07-05

本文件只记录建议命令，尚未执行安装。执行前请确认磁盘空间、CUDA/PyTorch 版本和是否需要代理。

## 1. 目录

```bash
cd /home/wjy/revision_runs/04_baselines/repos
```

已克隆仓库：

```text
CONCH
UNI
PLIP
LLaVA-Med
```

## 2. 轻量 pathology CLIP-like baselines 环境

适用：CONCH、UNI、PLIP。

建议环境：

```bash
conda create -n pathvlm_clip_baselines python=3.10 -y
conda activate pathvlm_clip_baselines
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers tokenizers timm==0.9.8 numpy pandas scikit-learn tqdm pillow regex ftfy h5py
pip install -e /home/wjy/revision_runs/04_baselines/repos/CONCH
pip install -e /home/wjy/revision_runs/04_baselines/repos/UNI
pip install -e /home/wjy/revision_runs/04_baselines/repos/PLIP
```

注意：

- CONCH 和 UNI 许可证均为 CC BY-NC 4.0。
- CONCH 权重可能需要 HuggingFace 访问权限。
- UNI 更偏视觉 encoder，不是直接 VQA 模型。
- PLIP 可通过 `vinid/plip` 从 HuggingFace 加载。

## 3. LLaVA-Med 环境

适用：LLaVA-Med v1.5 医学 VQA baseline。

建议环境：

```bash
conda create -n llava-med python=3.10 -y
conda activate llava-med
pip install --upgrade pip
pip install -e /home/wjy/revision_runs/04_baselines/repos/LLaVA-Med
```

模型：

```text
microsoft/llava-med-v1.5-mistral-7b
```

README 推荐 24GB GPU 可使用多 GPU：

```bash
python -m llava.serve.controller --host 0.0.0.0 --port 10000
python -m llava.serve.model_worker \
  --host 0.0.0.0 \
  --controller http://localhost:10000 \
  --port 40000 \
  --worker http://localhost:40000 \
  --model-path microsoft/llava-med-v1.5-mistral-7b \
  --multi-modal \
  --num-gpus 2
```

注意：

- 不建议在当前主环境中直接安装 LLaVA-Med，因为它固定较旧依赖。
- 先做 smoke test，再写 PathMMU batch inference adapter。

## 4. PathMMU adapter 待实现要点

输入格式：

```text
image: 图片路径
problem/question: 多选问题，包含 Options A/B/C/D
```

输出格式建议：

```json
{
  "question": "...",
  "image": "...",
  "ground_truth": "...",
  "model_output": "...",
  "predicted_option": "A/B/C/D/null",
  "correct": true
}
```

解析规则：

1. 优先解析 `<answer>...</answer>`。
2. 若无标签，解析最后出现的 A/B/C/D。
3. 保存原始输出，避免只保存解析结果。
4. 记录 prompt 模板。

## 5. 建议执行顺序

1. 先实现 PLIP zero-shot option matching，成本最低。
2. 再尝试 CONCH zero-shot option matching。
3. UNI 主要做 related work 定位，除非设计好特征分类协议。
4. 最后跑 LLaVA-Med，因为环境和权重成本最高。
