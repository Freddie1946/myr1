#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${PATHVLM_WORKSPACE_ROOT:-/home/dataset-assist-0/czy/wjy}"
REPO_ROOT="${PATHVLM_REPO_ROOT:-$WORKSPACE_ROOT/myr1}"
CODEX_HOME="${CODEX_HOME:-$WORKSPACE_ROOT/.codex-wjy}"
CODEX_INSTALL_DIR="${CODEX_INSTALL_DIR:-$WORKSPACE_ROOT/codex-bin}"
CODEX_BIN="${CODEX_BIN:-$CODEX_INSTALL_DIR/codex}"

if [[ ! -d "$REPO_ROOT/.git" ]]; then
  printf 'Repository not found: %s\n' "$REPO_ROOT" >&2
  exit 2
fi
if [[ ! -d "$CODEX_HOME" ]]; then
  printf 'CODEX_HOME not found: %s\n' "$CODEX_HOME" >&2
  exit 2
fi
if [[ ! -x "$CODEX_BIN" ]]; then
  printf 'Personal Codex binary not found: %s\n' "$CODEX_BIN" >&2
  exit 2
fi

export CODEX_HOME
export PATH="$CODEX_INSTALL_DIR:$PATH"

exec "$CODEX_BIN" \
  --cd "$REPO_ROOT" \
  --add-dir "$WORKSPACE_ROOT" \
  --sandbox workspace-write \
  --ask-for-approval on-request \
  "$@"

