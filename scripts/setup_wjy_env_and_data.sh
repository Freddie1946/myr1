#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="wjy"
PYTHON_VERSION="3.10"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data_runtime}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/model_runtime}"
DOWNLOAD_VLMR1_REC=false
DOWNLOAD_PATHMMU=true
DOWNLOAD_OMNIMEDVQA=true
DOWNLOAD_QWEN_MODEL=true
INSTALL_FLASH_ATTN=false
INSTALL_LLAM_FACTORY=true
CLONE_LLAM_FACTORY=true
FORCE=false
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-}"

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_wjy_env_and_data.sh [options]

Creates conda environment `wjy`, installs VLM-R1/open-r1 dependencies, prepares
runtime data/model directories, and downloads the default revision assets.

Options:
  --env-name NAME              Conda env name (default: wjy)
  --python VERSION             Python version (default: 3.10)
  --data-root PATH             Runtime data root (default: ./data_runtime)
  --model-root PATH            Runtime model root (default: ./model_runtime)
  --download-vlmr1-rec         Download om-ai-lab/VLM-R1 REC train2014 + annotations
  --download-pathmmu           Download PathMMU dataset via huggingface/datasets (default)
  --no-download-pathmmu        Skip PathMMU loading/download
  --download-omnimedvqa        Download OmniMedVQA via huggingface/datasets (default)
  --no-download-omnimedvqa     Skip OmniMedVQA loading/download
  --download-qwen-model        Download Qwen model weights with huggingface_hub (default)
  --no-download-qwen-model     Skip Qwen model weight download
  --qwen-model-id ID           Hugging Face model id (default: Qwen/Qwen2.5-VL-7B-Instruct)
  --qwen-model-dir PATH        Local Qwen model directory (default: MODEL_ROOT/<model basename>)
  --hf-endpoint URL            Hugging Face endpoint mirror (default: https://hf-mirror.com)
  --install-flash-attn         Install flash-attn --no-build-isolation
  --install-llamafactory       pip install existing LLaMA-Factory if found next to project (default)
  --no-install-llamafactory    Skip LLaMA-Factory install
  --clone-llamafactory         Clone hiyouga/LLaMA-Factory under external_repos/ if missing (default)
  --no-clone-llamafactory      Do not clone LLaMA-Factory if no local copy is found
  --force                      Re-run downloads even if target files exist
  -h, --help                   Show this help

Environment variables:
  HF_HOME, HF_TOKEN, DATA_ROOT, MODEL_ROOT may be set before running.

Notes:
  - Default run prepares LLaMA-Factory, PathMMU metadata/cache, OmniMedVQA metadata/cache,
    and Qwen2.5-VL-7B-Instruct weights through the configured Hugging Face endpoint.
  - Use --no-download-qwen-model/--no-download-pathmmu/--no-download-omnimedvqa for a
    lightweight environment-only run.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --python) PYTHON_VERSION="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --model-root) MODEL_ROOT="$2"; shift 2 ;;
    --download-vlmr1-rec) DOWNLOAD_VLMR1_REC=true; shift ;;
    --download-pathmmu) DOWNLOAD_PATHMMU=true; shift ;;
    --no-download-pathmmu) DOWNLOAD_PATHMMU=false; shift ;;
    --download-omnimedvqa) DOWNLOAD_OMNIMEDVQA=true; shift ;;
    --no-download-omnimedvqa) DOWNLOAD_OMNIMEDVQA=false; shift ;;
    --download-qwen-model) DOWNLOAD_QWEN_MODEL=true; shift ;;
    --no-download-qwen-model) DOWNLOAD_QWEN_MODEL=false; shift ;;
    --qwen-model-id) QWEN_MODEL_ID="$2"; shift 2 ;;
    --qwen-model-dir) QWEN_MODEL_DIR="$2"; shift 2 ;;
    --hf-endpoint) HF_ENDPOINT="$2"; shift 2 ;;
    --install-flash-attn) INSTALL_FLASH_ATTN=true; shift ;;
    --install-llamafactory) INSTALL_LLAM_FACTORY=true; shift ;;
    --no-install-llamafactory) INSTALL_LLAM_FACTORY=false; shift ;;
    --clone-llamafactory) CLONE_LLAM_FACTORY=true; shift ;;
    --no-clone-llamafactory) CLONE_LLAM_FACTORY=false; shift ;;
    --force) FORCE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$QWEN_MODEL_DIR" ]]; then
  QWEN_MODEL_DIR="$MODEL_ROOT/${QWEN_MODEL_ID##*/}"
fi

log() { printf '\n[setup-wjy] %s\n' "$*"; }
run_in_env() {
  conda run -n "$ENV_NAME" bash -lc "$*"
}

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please install Miniconda/Anaconda first." >&2
  exit 1
fi

log "Project root: $PROJECT_ROOT"
log "Data root: $DATA_ROOT"
log "Model root: $MODEL_ROOT"
log "Hugging Face endpoint: $HF_ENDPOINT"
log "Qwen model id: $QWEN_MODEL_ID"
log "Qwen model dir: $QWEN_MODEL_DIR"
export HF_ENDPOINT
export QWEN_MODEL_ID QWEN_MODEL_DIR
mkdir -p "$DATA_ROOT" "$MODEL_ROOT" "$PROJECT_ROOT/external_repos"
mkdir -p "$DATA_ROOT/vlm-r1-rec" "$DATA_ROOT/pathmmu" "$DATA_ROOT/omnimedvqa" "$DATA_ROOT/out_domain"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "Conda env '$ENV_NAME' already exists"
else
  log "Creating conda env '$ENV_NAME' with python=$PYTHON_VERSION"
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

log "Installing base Python tooling"
run_in_env "python -m pip install --upgrade pip setuptools wheel"

log "Installing PyTorch stack if missing"
run_in_env "python - <<'PY' || pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
try:
    import torch
    print('torch exists', torch.__version__)
except Exception:
    raise SystemExit(1)
PY"

log "Installing open-r1 multimodal package"
run_in_env "cd '$PROJECT_ROOT/src/open-r1-multimodal' && pip install -e '.[dev]'"

log "Installing PathVLM revision core package requirements"
run_in_env "pip install qwen-vl-utils tensorboardX openai python-Levenshtein scipy statsmodels scikit-learn pandas pillow tqdm pyyaml huggingface_hub datasets"

if [[ "$INSTALL_FLASH_ATTN" == true ]]; then
  log "Installing flash-attn (this may take a while and requires matching CUDA/PyTorch)"
  run_in_env "pip install flash-attn --no-build-isolation"
else
  log "Skipping flash-attn install. Use --install-flash-attn if needed."
fi

if [[ "$INSTALL_LLAM_FACTORY" == true ]]; then
  if [[ -d "$PROJECT_ROOT/external_repos/LLaMA-Factory" ]]; then
    LLAMA_FACTORY_DIR="$PROJECT_ROOT/external_repos/LLaMA-Factory"
  elif [[ -d "$PROJECT_ROOT/../LLaMA-Factory" ]]; then
    LLAMA_FACTORY_DIR="$PROJECT_ROOT/../LLaMA-Factory"
  elif [[ -d "/home/wjy/LLaMA-Factory" ]]; then
    LLAMA_FACTORY_DIR="/home/wjy/LLaMA-Factory"
  elif [[ "$CLONE_LLAM_FACTORY" == true ]]; then
    LLAMA_FACTORY_DIR="$PROJECT_ROOT/external_repos/LLaMA-Factory"
    log "Cloning LLaMA-Factory to $LLAMA_FACTORY_DIR"
    git clone https://github.com/hiyouga/LLaMA-Factory.git "$LLAMA_FACTORY_DIR"
  else
    echo "LLaMA-Factory not found. Use --clone-llamafactory or set it manually." >&2
    exit 1
  fi
  log "Installing LLaMA-Factory from $LLAMA_FACTORY_DIR"
  run_in_env "cd '$LLAMA_FACTORY_DIR' && pip install -e '.[torch,metrics]'"
else
  log "Skipping LLaMA-Factory install. SFT uses LLaMA-Factory; install later if needed."
fi

if [[ "$DOWNLOAD_QWEN_MODEL" == true ]]; then
  log "Downloading Qwen model weights from $QWEN_MODEL_ID to $QWEN_MODEL_DIR"
  mkdir -p "$QWEN_MODEL_DIR"
  if [[ "$FORCE" == true || ! -f "$QWEN_MODEL_DIR/config.json" ]]; then
    run_in_env "python - <<'PY'
import os
from huggingface_hub import snapshot_download

model_id = os.environ['QWEN_MODEL_ID']
local_dir = os.environ['QWEN_MODEL_DIR']
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    resume_download=True,
)
print(f'Downloaded {model_id} to {local_dir}')
PY"
  else
    log "Qwen model directory already has config.json; use --force to refresh."
  fi
else
  log "Skipping Qwen model weight download."
fi

if [[ "$DOWNLOAD_VLMR1_REC" == true ]]; then
  log "Downloading VLM-R1 REC data to $DATA_ROOT/vlm-r1-rec"
  REC_ROOT="$DATA_ROOT/vlm-r1-rec"
  mkdir -p "$REC_ROOT"
  if [[ "$FORCE" == true || ! -f "$REC_ROOT/train2014.zip" ]]; then
    wget -c -O "$REC_ROOT/train2014.zip" ${HF_ENDPOINT%/}/datasets/omlab/VLM-R1/resolve/main/train2014.zip
  fi
  if [[ "$FORCE" == true || ! -f "$REC_ROOT/rec_jsons_processed.zip" ]]; then
    wget -c -O "$REC_ROOT/rec_jsons_processed.zip" ${HF_ENDPOINT%/}/datasets/omlab/VLM-R1/resolve/main/rec_jsons_processed.zip
  fi
  if [[ ! -d "$REC_ROOT/train2014" ]]; then unzip -q "$REC_ROOT/train2014.zip" -d "$REC_ROOT"; fi
  if [[ ! -d "$REC_ROOT/rec_jsons_processed" ]]; then unzip -q "$REC_ROOT/rec_jsons_processed.zip" -d "$REC_ROOT"; fi
fi

if [[ "$DOWNLOAD_PATHMMU" == true ]]; then
  log "Downloading/loading PathMMU with datasets into HF cache and exporting metadata"
  run_in_env "python - <<'PY'
from datasets import load_dataset
import json, os
out_dir = os.environ.get('PATHMMU_OUT', '$DATA_ROOT/pathmmu')
os.makedirs(out_dir, exist_ok=True)
# Try common identifiers; update here if official id changes.
ids = ['PathMMU/PathMMU', 'pathmmu/pathmmu', 'PathMMU']
last = None
for dataset_id in ids:
    try:
        ds = load_dataset(dataset_id)
        print('loaded', dataset_id, ds)
        with open(os.path.join(out_dir, 'dataset_info.json'), 'w', encoding='utf-8') as f:
            json.dump({k: str(v) for k, v in ds.items()}, f, indent=2, ensure_ascii=False)
        break
    except Exception as exc:
        print('failed', dataset_id, repr(exc))
        last = exc
else:
    raise SystemExit(f'Could not load PathMMU automatically. Please download manually. Last error: {last}')
PY"
fi

if [[ "$DOWNLOAD_OMNIMEDVQA" == true ]]; then
  log "Downloading/loading OmniMedVQA with datasets into HF cache and exporting metadata"
  run_in_env "python - <<'PY'
from datasets import load_dataset
import json, os
out_dir = os.environ.get('OMNIMEDVQA_OUT', '$DATA_ROOT/omnimedvqa')
os.makedirs(out_dir, exist_ok=True)
ds = load_dataset('foreverbeliever/OmniMedVQA')
print(ds)
with open(os.path.join(out_dir, 'dataset_info.json'), 'w', encoding='utf-8') as f:
    json.dump({k: str(v) for k, v in ds.items()}, f, indent=2, ensure_ascii=False)
PY"
fi

cat > "$PROJECT_ROOT/.env.example" <<EOF
HF_ENDPOINT=$HF_ENDPOINT
PROJECT_ROOT=$PROJECT_ROOT
DATA_ROOT=$DATA_ROOT
MODEL_ROOT=$MODEL_ROOT
OPEN_R1_ROOT=$PROJECT_ROOT/src/open-r1-multimodal
PATHVLM_RL_TRAIN_JSON=$DATA_ROOT/pathmmu/pathmmu_rl_train.json
EVAL_JSON=$DATA_ROOT/pathmmu/pathmmu_test.json
IMAGE_ROOT=$DATA_ROOT/pathmmu/images
MODEL_PATH=$QWEN_MODEL_DIR
QWEN_MODEL_ID=$QWEN_MODEL_ID
QWEN_MODEL_DIR=$QWEN_MODEL_DIR
OUTPUT_DIR=$PROJECT_ROOT/outputs/train/pathvlm_grpo
OUTPUT_JSON=$PROJECT_ROOT/outputs/eval/pathvlm_results.json
PATHVLM_ENABLE_GPT_REWARD=false
# OPENAI_API_KEY=
# OPENAI_BASE_URL=https://api.openai.com/v1
EOF

cat > "$DATA_ROOT/README_DATA.md" <<EOF
# Runtime Data Directory

Created by scripts/setup_wjy_env_and_data.sh.

- vlm-r1-rec/: optional om-ai-lab/VLM-R1 REC data.
- pathmmu/: place PathMMU images and image-level disjoint split JSON files here.
- omnimedvqa/: optional OmniMedVQA cache/metadata for out-of-domain evaluation.
- out_domain/: place ChestCT, ISIC2020, Retinal OCT-C8, Diabetic Retinopathy exports here.

Recommended PathMMU files:
- pathmmu_sft_train.json
- pathmmu_rl_train.json
- pathmmu_test.json
- split_manifest.json
EOF

log "Smoke checking important imports"
run_in_env "python - <<'PY'
import torch, transformers, datasets
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('datasets', datasets.__version__)
try:
    import open_r1
    print('open_r1 import ok')
except Exception as exc:
    print('open_r1 import warning:', exc)
PY"

log "Setup completed. Activate with: conda activate $ENV_NAME"
log "Review .env.example and DATA_ROOT/README_DATA.md before training."
