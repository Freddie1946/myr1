#!/usr/bin/env bash
set -euo pipefail

: "${PATHVLM_REPO:?source FORMAL_PATHS.env first}"
: "${PATHVLM_INSTALL_ROOT:?source FORMAL_PATHS.env first}"
: "${PATHVLM_SFT_PYTHON:?source FORMAL_PATHS.env first}"
: "${PATHVLM_LLAMAFACTORY_SRC:?source FORMAL_PATHS.env first}"

VARIANT="${1:-z2_gpu_nogc_fused}"
if [[ "$VARIANT" != "z2_gpu_nogc_fused" && "$VARIANT" != "z2_gpu_gc_fused" ]]; then
  echo "unsupported speed-smoke variant: $VARIANT" >&2
  exit 2
fi
RUN_ID="sft_speed_smoke_${VARIANT}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$PATHVLM_INSTALL_ROOT/runs/stage1_sft_speed_smoke/$RUN_ID"

exec "$PATHVLM_SFT_PYTHON" "$PATHVLM_REPO/formal_machine/run_formal_sft_speed_smoke.py" \
  --repo-root "$PATHVLM_REPO" \
  --install-root "$PATHVLM_INSTALL_ROOT" \
  --llamafactory-src "$PATHVLM_LLAMAFACTORY_SRC" \
  --variant "$VARIANT" \
  --run-dir "$RUN_DIR" \
  --max-steps 20 \
  --master-port 29710
