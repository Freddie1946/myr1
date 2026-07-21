#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--approval-check" ]]; then
  printf '%s\n' \
    'SCOPE: Base, SFT n500/n1000 finals, SFT n2000/n3000 epochs 1-10, Outcome-GRPO epochs 1-3' \
    'DATA: frozen validation_0385 only; test is forbidden' \
    'GENERATION: deterministic greedy decoding, max_new_tokens=1024, any cap hit fails the run' \
    'GPUS: physical 0-7, one independent model per GPU at a time' \
    'OUTPUTS: 26 manifests/logs/metrics and 10,010 raw predictions' \
    'SELECTION: max accuracy; tie format; tie earliest epoch' \
    'NO MUTATION: no training, model update, Stage 3, or test inference'
  exit 0
fi

: "${PATHVLM_REPO:?source FORMAL_PATHS.env first}"
: "${PATHVLM_INSTALL_ROOT:?source FORMAL_PATHS.env first}"
: "${PATHVLM_SFT_PYTHON:?source FORMAL_PATHS.env first}"

exec "$PATHVLM_SFT_PYTHON" \
  "$PATHVLM_REPO/formal_machine/run_formal_validation_1024_all_candidates.py" \
  --repo-root "$PATHVLM_REPO" \
  --install-root "$PATHVLM_INSTALL_ROOT" \
  --gpus 0,1,2,3,4,5,6,7
