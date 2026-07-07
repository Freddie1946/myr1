# Setup `wjy` Environment and Runtime Data

Use this on the training server after cloning the repository.

Basic environment setup without downloading large datasets:

```bash
bash scripts/setup_wjy_env_and_data.sh
conda activate wjy
```

Install optional flash-attn:

```bash
bash scripts/setup_wjy_env_and_data.sh --install-flash-attn
```

Install or clone LLaMA-Factory for SFT:

```bash
bash scripts/setup_wjy_env_and_data.sh --clone-llamafactory
```

Download VLM-R1 REC data from `om-ai-lab/VLM-R1`:

```bash
bash scripts/setup_wjy_env_and_data.sh --download-vlmr1-rec
```

Try to download/load PathMMU via Hugging Face datasets:

```bash
bash scripts/setup_wjy_env_and_data.sh --download-pathmmu
```

Try to download/load OmniMedVQA for out-of-domain metadata/cache:

```bash
bash scripts/setup_wjy_env_and_data.sh --download-omnimedvqa
```

Recommended runtime layout:

```text
DATA_ROOT/
  pathmmu/
    images/
    pathmmu_sft_train.json
    pathmmu_rl_train.json
    pathmmu_test.json
    split_manifest.json
  omnimedvqa/
  out_domain/
    chest_ct/
    isic2020/
    retinal_oct_c8/
    diabetic_retinopathy/
MODEL_ROOT/
  Qwen2.5-VL-7B-Instruct/
  checkpoints/
```

Large data and model weights are intentionally not tracked by Git.
