#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--approval-check" ]]; then
  printf '%s\n' \
    'SCOPE: base plus exact n2000/n3000 seed42 epoch-1..10 snapshots' \
    'DATA: frozen validation_0385 only; test is forbidden' \
    'GPUS: physical 0-7, one independent deterministic job per GPU at a time' \
    'OUTPUTS: 21 job manifests, logs, metrics, and 385 raw predictions each' \
    'SELECTION: max validation accuracy; tie format; tie earliest epoch' \
    'NO MUTATION: no training, model update, RL, or test inference'
  exit 0
fi

: "${PATHVLM_REPO:?source FORMAL_PATHS.env first}"
: "${PATHVLM_INSTALL_ROOT:?source FORMAL_PATHS.env first}"
: "${PATHVLM_SFT_PYTHON:?source FORMAL_PATHS.env first}"

exec "$PATHVLM_SFT_PYTHON" \
  "$PATHVLM_REPO/formal_machine/run_formal_sft_validation_curve.py" \
  --repo-root "$PATHVLM_REPO" \
  --install-root "$PATHVLM_INSTALL_ROOT" \
  --gpus 0,1,2,3,4,5,6,7
