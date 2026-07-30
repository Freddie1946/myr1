#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=/home/dataset-assist-0/czy/wjy
REPO_ROOT="$WORK_ROOT/myr1"
INSTALL_ROOT="$WORK_ROOT/pathvlm_r1_v1_a100"
PARENT="$INSTALL_ROOT/transferred_checkpoints/sft_n3000_seed42_epoch03_step1125"
SFT_PYTHON="$INSTALL_ROOT/envs/sft/bin/python"
LLAMAFACTORY="$INSTALL_ROOT/sources/LLaMA-Factory"

if [[ $# -ne 1 ]]; then
  printf 'usage: %s /absolute/path/to/frozen_protocol.json\n' "$0" >&2
  exit 2
fi

exec "$SFT_PYTHON" "$REPO_ROOT/formal_machine/run_sft4000_control.py" \
  --repo-root "$REPO_ROOT" \
  --install-root "$INSTALL_ROOT" \
  --parent "$PARENT" \
  --llamafactory-src "$LLAMAFACTORY" \
  --python "$SFT_PYTHON" \
  --protocol "$1"
