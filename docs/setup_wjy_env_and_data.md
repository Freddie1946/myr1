# Setup `wjy` Environment and Runtime Data

Use this on the training server after cloning the repository.

Default setup creates the `wjy` environment, installs local/open-r1 dependencies, prepares LLaMA-Factory, loads PathMMU and OmniMedVQA through Hugging Face `datasets`, and downloads Qwen2.5-VL-7B-Instruct weights. Hugging Face operations use the mirror `https://hf-mirror.com` by default:

```bash
bash scripts/setup_wjy_env_and_data.sh
conda activate wjy
```

Lightweight environment-only setup:

```bash
bash scripts/setup_wjy_env_and_data.sh \
  --no-install-llamafactory \
  --no-download-pathmmu \
  --no-download-omnimedvqa \
  --no-download-qwen-model
```

Install optional flash-attn:

```bash
bash scripts/setup_wjy_env_and_data.sh --install-flash-attn
```

LLaMA-Factory is installed by default. If no local copy is found, the script clones it under `external_repos/`. Disable this behavior with:

```bash
bash scripts/setup_wjy_env_and_data.sh --no-install-llamafactory
```

Also download VLM-R1 REC data from `om-ai-lab/VLM-R1` through the default HF mirror:

```bash
bash scripts/setup_wjy_env_and_data.sh --download-vlmr1-rec
```

PathMMU is loaded/downloaded by default via Hugging Face datasets. The explicit flag is kept for clarity:

```bash
bash scripts/setup_wjy_env_and_data.sh --download-pathmmu
```

OmniMedVQA is loaded/downloaded by default for out-of-domain metadata/cache. The explicit flag is kept for clarity:

```bash
bash scripts/setup_wjy_env_and_data.sh --download-omnimedvqa
```

Qwen model weights are downloaded by default:

```text
model id: Qwen/Qwen2.5-VL-7B-Instruct
local dir: MODEL_ROOT/<model basename>, e.g. MODEL_ROOT/Qwen2.5-VL-7B-Instruct
```

Use another Qwen checkpoint or skip the model download:

```bash
bash scripts/setup_wjy_env_and_data.sh \
  --qwen-model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --qwen-model-dir /path/to/Qwen2.5-VL-3B-Instruct

bash scripts/setup_wjy_env_and_data.sh --no-download-qwen-model
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


## Hugging Face Mirror

The setup script defaults to:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Override it if needed:

```bash
bash scripts/setup_wjy_env_and_data.sh --hf-endpoint https://huggingface.co
```

The mirror is used for direct `wget` downloads and exported for `huggingface_hub` / `datasets` calls.
