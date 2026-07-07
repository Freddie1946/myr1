#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="wjy"
PYTHON_VERSION="3.10"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data_runtime}"
MODEL_ROOT="${MODEL_ROOT:-$PROJECT_ROOT/model_runtime}"
DOWNLOAD_VLMR1_REC=false
DOWNLOAD_PATHMMU=false
DOWNLOAD_OMNIMEDVQA=false
INSTALL_FLASH_ATTN=false
INSTALL_LLAM_FACTORY=false
CLONE_LLAM_FACTORY=false
FORCE=false

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_wjy_env_and_data.sh [options]

Creates conda environment `wjy`, installs VLM-R1/open-r1 dependencies, prepares
runtime data/model directories, and optionally downloads datasets.

Options:
  --env-name NAME              Conda env name (default: wjy)
  --python VERSION             Python version (default: 3.10)
  --data-root PATH             Runtime data root (default: ./data_runtime)
  --model-root PATH            Runtime model root (default: ./model_runtime)
  --download-vlmr1-rec         Download om-ai-lab/VLM-R1 REC train2014 + annotations
  --download-pathmmu           Download PathMMU dataset via huggingface/datasets (metadata/cache)
  --download-omnimedvqa        Download OmniMedVQA via huggingface/datasets (metadata/cache)
  --install-flash-attn         Install flash-attn --no-build-isolation
  --install-llamafactory       pip install existing LLaMA-Factory if found next to project
  --clone-llamafactory         Clone hiyouga/LLaMA-Factory under external_repos/ and install it
  --force                      Re-run downloads even if target files exist
  -h, --help                   Show this help

Environment variables:
  HF_HOME, HF_TOKEN, DATA_ROOT, MODEL_ROOT may be set before running.

Notes:
  - Large downloads are opt-in. Default run only creates env/dirs and installs code.
  - Model weights are not downloaded by default.
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
    --download-omnimedvqa) DOWNLOAD_OMNIMEDVQA=true; shift ;;
    --install-flash-attn) INSTALL_FLASH_ATTN=true; shift ;;
    --install-llamafactory) INSTALL_LLAM_FACTORY=true; shift ;;
    --clone-llamafactory) CLONE_LLAM_FACTORY=true; shift ;;
    --force) FORCE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

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
run_in_env "pip install qwen-vl-utils tensorboardX openai python-Levenshtein scipy statsmodels scikit-learn pandas pillow tqdm pyyaml"

if [[ "$INSTALL_FLASH_ATTN" == true ]]; then
  log "Installing flash-attn (this may take a while and requires matching CUDA/PyTorch)"
  run_in_env "pip install flash-attn --no-build-isolation"
else
  log "Skipping flash-attn install. Use --install-flash-attn if needed."
fi

if [[ "$CLONE_LLAM_FACTORY" == true ]]; then
  if [[ ! -d "$PROJECT_ROOT/external_repos/LLaMA-Factory/.git" ]]; then
    log "Cloning LLaMA-Factory"
    git clone https://github.com/hiyouga/LLaMA-Factory.git "$PROJECT_ROOT/external_repos/LLaMA-Factory"
  fi
  INSTALL_LLAM_FACTORY=true
fi

if [[ "$INSTALL_LLAM_FACTORY" == true ]]; then
  if [[ -d "$PROJECT_ROOT/external_repos/LLaMA-Factory" ]]; then
    LLAMA_FACTORY_DIR="$PROJECT_ROOT/external_repos/LLaMA-Factory"
  elif [[ -d "$PROJECT_ROOT/../LLaMA-Factory" ]]; then
    LLAMA_FACTORY_DIR="$PROJECT_ROOT/../LLaMA-Factory"
  elif [[ -d "/home/wjy/LLaMA-Factory" ]]; then
    LLAMA_FACTORY_DIR="/home/wjy/LLaMA-Factory"
  else
    echo "LLaMA-Factory not found. Use --clone-llamafactory or set it manually." >&2
    exit 1
  fi
  log "Installing LLaMA-Factory from $LLAMA_FACTORY_DIR"
  run_in_env "cd '$LLAMA_FACTORY_DIR' && pip install -e '.[torch,metrics]'"
else
  log "Skipping LLaMA-Factory install. SFT uses LLaMA-Factory; install later if needed."
fi

if [[ "$DOWNLOAD_VLMR1_REC" == true ]]; then
  log "Downloading VLM-R1 REC data to $DATA_ROOT/vlm-r1-rec"
  REC_ROOT="$DATA_ROOT/vlm-r1-rec"
  mkdir -p "$REC_ROOT"
  if [[ "$FORCE" == true || ! -f "$REC_ROOT/train2014.zip" ]]; then
    wget -c -O "$REC_ROOT/train2014.zip" https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/train2014.zip
  fi
  if [[ "$FORCE" == true || ! -f "$REC_ROOT/rec_jsons_processed.zip" ]]; then
    wget -c -O "$REC_ROOT/rec_jsons_processed.zip" https://huggingface.co/datasets/omlab/VLM-R1/resolve/main/rec_jsons_processed.zip
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
PROJECT_ROOT=$PROJECT_ROOT
DATA_ROOT=$DATA_ROOT
MODEL_ROOT=$MODEL_ROOT
OPEN_R1_ROOT=$PROJECT_ROOT/src/open-r1-multimodal
PATHVLM_RL_TRAIN_JSON=$DATA_ROOT/pathmmu/pathmmu_rl_train.json
EVAL_JSON=$DATA_ROOT/pathmmu/pathmmu_test.json
IMAGE_ROOT=$DATA_ROOT/pathmmu/images
MODEL_PATH=$MODEL_ROOT/Qwen2.5-VL-7B-Instruct
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
