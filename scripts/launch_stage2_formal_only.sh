#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_ROOT="/home/wjy/pathvlm_r1_v1_formal"
PYTHON="$INSTALL_ROOT/envs/grpo/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing pinned GRPO Python: $PYTHON" >&2
  exit 1
fi

exec "$PYTHON" "$REPO_ROOT/formal_machine/run_stage2_formal_only.py" "$@"
