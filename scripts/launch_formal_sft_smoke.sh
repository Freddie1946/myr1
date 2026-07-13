#!/usr/bin/env bash
set -euo pipefail

: "${PATHVLM_REPO:?source FORMAL_PATHS.env first}"
: "${PATHVLM_INSTALL_ROOT:?source FORMAL_PATHS.env first}"
: "${PATHVLM_SFT_PYTHON:?source FORMAL_PATHS.env first}"
: "${PATHVLM_LLAMAFACTORY_SRC:?source FORMAL_PATHS.env first}"
: "${PATHVLM_SMOKE_CUDA_VISIBLE_DEVICES:?set explicitly allocated GPU indices, e.g. 1,2,3,4}"

NPROC_PER_NODE="${PATHVLM_SMOKE_NPROC_PER_NODE:-4}"
MASTER_PORT="${PATHVLM_SMOKE_MASTER_PORT:-29650}"
RUN_ID="${PATHVLM_SMOKE_RUN_ID:-formal_7b_sft_smoke_n0008_seed0042_$(date +%Y%m%d_%H%M%S)}"
CONFIG="${PATHVLM_SMOKE_CONFIG:-$PATHVLM_INSTALL_ROOT/generated_configs/sft/sft_smoke_n0008.yaml}"
RUN_DIR="$PATHVLM_INSTALL_ROOT/runs/stage1_sft_smoke/$RUN_ID"

exec "$PATHVLM_SFT_PYTHON" "$PATHVLM_REPO/formal_machine/run_formal_sft_smoke.py" \
  --repo-root "$PATHVLM_REPO" \
  --install-root "$PATHVLM_INSTALL_ROOT" \
  --llamafactory-src "$PATHVLM_LLAMAFACTORY_SRC" \
  --config "$CONFIG" \
  --run-dir "$RUN_DIR" \
  --cuda-visible-devices "$PATHVLM_SMOKE_CUDA_VISIBLE_DEVICES" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --master-port "$MASTER_PORT"
