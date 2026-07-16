#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--approval-check" ]]; then
  printf '%s\n' \
    'PARENT: exact formal_sft_n3000_seed0042_20260717_014544 only' \
    'TARGET: one fresh base-model n2000 seed42 ten-epoch SFT only' \
    'FAIL-CLOSED: parent gates, hashes, disk, GPU occupancy, or ambiguity stops launch' \
    'GPU POLICY: never kill a process; wait while any compute process exists' \
    'FORBIDDEN: validation, test, n500/n1000/n3000 relaunch, RL, Stage 2, Stage 3'
  exit 0
fi

: "${PATHVLM_REPO:?source FORMAL_PATHS.env first}"
: "${PATHVLM_INSTALL_ROOT:?source FORMAL_PATHS.env first}"
: "${PATHVLM_SFT_PYTHON:?source FORMAL_PATHS.env first}"

PARENT="$PATHVLM_INSTALL_ROOT/runs/stage1_sft/n3000_seed0042/formal_sft_n3000_seed0042_20260717_014544/run_manifest.yaml"

exec "$PATHVLM_SFT_PYTHON" \
  "$PATHVLM_REPO/formal_machine/run_n3000_to_n2000_handoff.py" \
  --repo-root "$PATHVLM_REPO" \
  --install-root "$PATHVLM_INSTALL_ROOT" \
  --parent-manifest "$PARENT" \
  --poll-seconds 60 \
  --daemon
