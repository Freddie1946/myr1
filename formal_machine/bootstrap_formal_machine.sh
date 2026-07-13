#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="${1:?usage: bootstrap_formal_machine.sh /path/to/formal_machine.env}"
source "$CONFIG_PATH"

: "${INSTALL_ROOT:?set INSTALL_ROOT}"
: "${SPLIT_ROOT:?set SPLIT_ROOT}"
: "${IMAGE_ROOT:?set IMAGE_ROOT}"
: "${CONDA_EXE:=conda}"
: "${ONLINE:=1}"
: "${WHEELHOUSE:=}"
: "${PYTORCH_INDEX_URL:=https://download.pytorch.org/whl/cu118}"
: "${TORCH_VERSION:=2.6.0}"
: "${TORCHVISION_VERSION:=0.21.0}"
: "${BASE_MODEL_ID:=Qwen/Qwen2.5-VL-7B-Instruct}"
: "${BASE_MODEL_REVISION:=cc594898137f460bfe9f0759e9844b3ce807cfb5}"
: "${BASE_MODEL_SOURCE:=}"
: "${PATHMMU_AUTO_DOWNLOAD:=1}"
: "${PATHMMU_DATASET_ID:=jamessyx/PathMMU}"
: "${PATHMMU_DATASET_REVISION:=main}"
: "${PATHMMU_ARCHIVE:=}"
: "${OVERWRITE_DATA:=0}"
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
: "${NPROC_PER_NODE:=4}"
: "${MASTER_PORT_BASE:=29600}"
: "${GRPO_PER_DEVICE_BATCH:=1}"
: "${HF_HUB_DISABLE_XET:=1}"
: "${HF_HUB_DOWNLOAD_TIMEOUT:=120}"
: "${HF_HUB_MAX_WORKERS:=1}"
export HF_HUB_DISABLE_XET HF_HUB_DOWNLOAD_TIMEOUT

LLAMAFACTORY_SRC="${LLAMAFACTORY_SRC:-$INSTALL_ROOT/sources/LLaMA-Factory}"
VLMR1_SRC="${VLMR1_SRC:-$REPO_ROOT/vendor/open-r1-multimodal}"
LLAMAFACTORY_GIT_URL="${LLAMAFACTORY_GIT_URL:-https://github.com/hiyouga/LLaMA-Factory.git}"
LLAMAFACTORY_REVISION="${LLAMAFACTORY_REVISION:-e2299e261be852304bb1d370515078193ab12bd8}"
LLAMAFACTORY_LAUNCHER_SHA256="${LLAMAFACTORY_LAUNCHER_SHA256:-8f16bb782a6da2122b5accd50ce1a01fd99284dffa20ff340850ec3b06927b77}"

ENV_ROOT="$INSTALL_ROOT/envs"
SFT_ENV="$ENV_ROOT/sft"
GRPO_ENV="$ENV_ROOT/grpo"
DATA_OUT="$INSTALL_ROOT/data/pathmmu_image_disjoint_v1"
MODEL_DIR="$INSTALL_ROOT/models/Qwen2.5-VL-7B-Instruct-$BASE_MODEL_REVISION"
REPORT_DIR="$INSTALL_ROOT/reports"
MODEL_SOURCE_MANIFEST="$REPORT_DIR/model_source_manifest.json"
LLAMAFACTORY_SOURCE_MANIFEST="$REPORT_DIR/llamafactory_source_manifest.json"
mkdir -p "$ENV_ROOT" "$INSTALL_ROOT/sources" "$INSTALL_ROOT/models" "$REPORT_DIR"

log() { printf '[formal-setup] %s\n' "$*"; }
die() { printf '[formal-setup] ERROR: %s\n' "$*" >&2; exit 1; }
require_command() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

require_command nvidia-smi
require_command git
require_command python3
if ! command -v "$CONDA_EXE" >/dev/null 2>&1 && [[ ! -x "$CONDA_EXE" ]]; then
  die "conda executable not found: $CONDA_EXE"
fi
[[ -d "$SPLIT_ROOT" ]] || die "split root not found: $SPLIT_ROOT"
mkdir -p "$IMAGE_ROOT"
if [[ "$ONLINE" == 0 ]]; then
  [[ -d "$WHEELHOUSE" ]] || die "ONLINE=0 requires WHEELHOUSE: $WHEELHOUSE"
  if [[ "$PATHMMU_AUTO_DOWNLOAD" == 1 && -z "$PATHMMU_ARCHIVE" ]]; then
    die "offline PathMMU setup requires PATHMMU_ARCHIVE or PATHMMU_AUTO_DOWNLOAD=0"
  fi
fi

log "Verifying repository code hashes"
python3 "$REPO_ROOT/scripts/verify_code_hash_manifest.py" \
  --repo-root "$REPO_ROOT" \
  --manifest "$REPO_ROOT/protocol/code_hash_manifest_20260714_001041.json"

log "Verifying frozen split hashes and image-disjoint invariants"
python3 "$REPO_ROOT/formal_machine/verify_frozen_splits.py" \
  --split-root "$SPLIT_ROOT" \
  --output "$REPORT_DIR/frozen_split_verification.json"

log "Hardware snapshot"
nvidia-smi
df -h "$INSTALL_ROOT"

if [[ -d "$LLAMAFACTORY_SRC/.git" ]]; then
  git -C "$LLAMAFACTORY_SRC" fetch --all --tags >/dev/null 2>&1 || true
  git -C "$LLAMAFACTORY_SRC" checkout --detach "$LLAMAFACTORY_REVISION"
elif [[ -f "$LLAMAFACTORY_SRC/pyproject.toml" || -f "$LLAMAFACTORY_SRC/setup.py" ]]; then
  log "Using copied LLaMA-Factory source tree without Git metadata: $LLAMAFACTORY_SRC"
elif [[ "$ONLINE" == 1 ]]; then
  log "Cloning LLaMA-Factory"
  git clone "$LLAMAFACTORY_GIT_URL" "$LLAMAFACTORY_SRC"
  git -C "$LLAMAFACTORY_SRC" checkout --detach "$LLAMAFACTORY_REVISION"
else
  die "offline mode requires an existing LLaMA-Factory source tree"
fi
LLAMAFACTORY_HEAD="$(git -C "$LLAMAFACTORY_SRC" rev-parse HEAD 2>/dev/null || true)"
[[ "$LLAMAFACTORY_HEAD" == "$LLAMAFACTORY_REVISION" ]] || \
  die "LLaMA-Factory revision mismatch: expected $LLAMAFACTORY_REVISION, got ${LLAMAFACTORY_HEAD:-no Git revision}"
ACTUAL_LLAMAFACTORY_LAUNCHER_SHA256="$(sha256sum "$LLAMAFACTORY_SRC/src/llamafactory/launcher.py" | awk '{print $1}')"
[[ "$ACTUAL_LLAMAFACTORY_LAUNCHER_SHA256" == "$LLAMAFACTORY_LAUNCHER_SHA256" ]] || \
  die "LLaMA-Factory launcher hash mismatch: expected $LLAMAFACTORY_LAUNCHER_SHA256, got $ACTUAL_LLAMAFACTORY_LAUNCHER_SHA256"
cat > "$LLAMAFACTORY_SOURCE_MANIFEST" <<EOF
{
  "git_url": "$LLAMAFACTORY_GIT_URL",
  "revision": "$LLAMAFACTORY_REVISION",
  "resolved_head": "$LLAMAFACTORY_HEAD",
  "release": "v0.9.2",
  "launcher_sha256": "$ACTUAL_LLAMAFACTORY_LAUNCHER_SHA256"
}
EOF
[[ -f "$VLMR1_SRC/setup.py" && -d "$VLMR1_SRC/src/open_r1" ]] || \
  die "vendored Open-R1 source missing: $VLMR1_SRC"

create_env() {
  local env_path="$1"
  if [[ ! -x "$env_path/bin/python" ]]; then
    log "Creating environment: $env_path"
    "$CONDA_EXE" create -y -p "$env_path" python=3.10 pip
  else
    log "Reusing existing environment: $env_path"
  fi
}

pip_install() {
  local python="$1"; shift
  if [[ "$ONLINE" == 1 ]]; then
    "$python" -m pip install --no-cache-dir "$@"
  else
    "$python" -m pip install --no-index --find-links "$WHEELHOUSE" "$@"
  fi
}

install_torch() {
  local python="$1"
  if [[ "$ONLINE" == 1 ]]; then
    "$python" -m pip install --no-cache-dir --index-url "$PYTORCH_INDEX_URL" \
      "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
  else
    "$python" -m pip install --no-index --find-links "$WHEELHOUSE" \
      "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
  fi
}

install_common() {
  local python="$1"
  pip_install "$python" \
    transformers==4.49.0 tokenizers==0.21.0 accelerate==1.4.0 datasets==3.3.2 \
    deepspeed==0.15.4 peft pillow safetensors sentencepiece qwen-vl-utils \
    math-verify pyyaml packaging psutil scipy
}

create_env "$SFT_ENV"
create_env "$GRPO_ENV"

log "Installing SFT environment"
install_torch "$SFT_ENV/bin/python"
install_common "$SFT_ENV/bin/python"
pip_install "$SFT_ENV/bin/python" trl==0.9.6
pip_install "$SFT_ENV/bin/python" -e "$LLAMAFACTORY_SRC"
# Re-pin the audited versions after editable dependency resolution.
pip_install "$SFT_ENV/bin/python" transformers==4.49.0 tokenizers==0.21.0 trl==0.9.6

log "Installing GRPO environment"
install_torch "$GRPO_ENV/bin/python"
install_common "$GRPO_ENV/bin/python"
pip_install "$GRPO_ENV/bin/python" trl==0.15.2 bitsandbytes liger-kernel==0.5.2 einops hf-transfer
"$GRPO_ENV/bin/python" -m pip install --no-deps -e "$VLMR1_SRC"

if [[ "$PATHMMU_AUTO_DOWNLOAD" == 1 ]]; then
  log "Downloading and extracting frozen-split PathMMU images"
  PATHMMU_ARGS=(
    --split-root "$SPLIT_ROOT"
    --image-root "$IMAGE_ROOT"
    --cache-dir "$INSTALL_ROOT/cache/pathmmu"
    --report "$REPORT_DIR/pathmmu_download_report.json"
    --dataset-id "$PATHMMU_DATASET_ID"
    --revision "$PATHMMU_DATASET_REVISION"
  )
  if [[ -n "$PATHMMU_ARCHIVE" ]]; then PATHMMU_ARGS+=(--archive "$PATHMMU_ARCHIVE"); fi
  "$GRPO_ENV/bin/python" "$REPO_ROOT/formal_machine/download_pathmmu.py" "${PATHMMU_ARGS[@]}"
else
  log "PATHMMU_AUTO_DOWNLOAD=0; existing images will be validated during data preparation"
fi

log "Preparing exact base-model snapshot"
if [[ -n "$BASE_MODEL_SOURCE" ]]; then
  [[ -f "$BASE_MODEL_SOURCE/config.json" ]] || die "invalid BASE_MODEL_SOURCE: $BASE_MODEL_SOURCE"
  MODEL_DIR="$(readlink -f "$BASE_MODEL_SOURCE")"
else
  [[ "$ONLINE" == 1 ]] || die "offline mode requires BASE_MODEL_SOURCE"
  "$GRPO_ENV/bin/python" - "$BASE_MODEL_ID" "$BASE_MODEL_REVISION" "$MODEL_DIR" "$HF_HUB_MAX_WORKERS" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=sys.argv[1],
    revision=sys.argv[2],
    local_dir=sys.argv[3],
    max_workers=int(sys.argv[4]),
)
PY
fi
cat > "$MODEL_SOURCE_MANIFEST" <<EOF
{
  "model_id": "$BASE_MODEL_ID",
  "revision": "$BASE_MODEL_REVISION",
  "source": "${BASE_MODEL_SOURCE:-huggingface_snapshot_download}"
}
EOF

log "Preparing and validating formal data adapters"
DATA_ARGS=(
  --split-root "$SPLIT_ROOT"
  --image-root "$IMAGE_ROOT"
  --output-root "$DATA_OUT"
)
if [[ "$OVERWRITE_DATA" == 1 ]]; then
  DATA_ARGS+=(--overwrite)
else
  DATA_ARGS+=(--reuse-if-valid)
fi
"$GRPO_ENV/bin/python" "$REPO_ROOT/formal_machine/prepare_formal_data.py" "${DATA_ARGS[@]}"

log "Rendering machine-resolved SFT and Outcome-GRPO configs"
"$GRPO_ENV/bin/python" "$REPO_ROOT/formal_machine/render_formal_configs.py" \
  --repo-root "$REPO_ROOT" --install-root "$INSTALL_ROOT" --model-dir "$MODEL_DIR" \
  --data-root "$DATA_OUT" --sft-python "$SFT_ENV/bin/python" --grpo-python "$GRPO_ENV/bin/python" \
  --llamafactory-src "$LLAMAFACTORY_SRC" --open-r1-src "$VLMR1_SRC" \
  --cuda-visible-devices "$CUDA_VISIBLE_DEVICES" --nproc-per-node "$NPROC_PER_NODE" \
  --master-port-base "$MASTER_PORT_BASE" --grpo-per-device-batch "$GRPO_PER_DEVICE_BATCH"

log "Running environment/model/data preflight"
"$GRPO_ENV/bin/python" "$REPO_ROOT/formal_machine/preflight_formal_machine.py" \
  --install-root "$INSTALL_ROOT" \
  --model-dir "$MODEL_DIR" \
  --model-source-manifest "$MODEL_SOURCE_MANIFEST" \
  --llamafactory-src "$LLAMAFACTORY_SRC" \
  --llamafactory-source-manifest "$LLAMAFACTORY_SOURCE_MANIFEST" \
  --expected-llamafactory-revision "$LLAMAFACTORY_REVISION" \
  --expected-llamafactory-launcher-sha256 "$LLAMAFACTORY_LAUNCHER_SHA256" \
  --base-model-manifest "$REPO_ROOT/protocol/base_model_manifest.json" \
  --expected-model-id "$BASE_MODEL_ID" \
  --expected-revision "$BASE_MODEL_REVISION" \
  --data-root "$DATA_OUT" \
  --split-verification "$REPORT_DIR/frozen_split_verification.json" \
  --output "$REPORT_DIR/preflight_report.json"

cat > "$INSTALL_ROOT/FORMAL_PATHS.env" <<EOF
export PATHVLM_REPO="$REPO_ROOT"
export PATHVLM_INSTALL_ROOT="$INSTALL_ROOT"
export PATHVLM_SFT_PYTHON="$SFT_ENV/bin/python"
export PATHVLM_GRPO_PYTHON="$GRPO_ENV/bin/python"
export PATHVLM_MODEL_DIR="$MODEL_DIR"
export PATHVLM_LLAMAFACTORY_SRC="$LLAMAFACTORY_SRC"
export PATHVLM_OPEN_R1_SRC="$VLMR1_SRC"
export PATHVLM_DATA_ROOT="$DATA_OUT"
export PATHVLM_LF_DATA="$DATA_OUT/llamafactory"
export PATHVLM_GRPO_DATA="$DATA_OUT/grpo"
EOF

log "SETUP COMPLETE"
log "Preflight report: $REPORT_DIR/preflight_report.json"
log "Persistent paths: $INSTALL_ROOT/FORMAL_PATHS.env"
log "Generated configs: $INSTALL_ROOT/generated_configs"
log "Next: review GPU allocation, then run scripts/launch_formal_sft_smoke.sh for the save/reload/resume gate."
