#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--approval-check" ]]; then
  printf '%s\n' \
    'APPROVAL SCOPE: deterministic frozen validation inference only' \
    'DATA: validation_0385 only; test is rejected' \
    'OUTPUTS: raw predictions, parser scores, checkpoint/config provenance' \
    'GUARDS: one fresh run directory; no model update'
  exit 0
fi

: "${PATHVLM_REPO:?source FORMAL_PATHS.env first}"
: "${PATHVLM_INSTALL_ROOT:?source FORMAL_PATHS.env first}"
: "${PATHVLM_SFT_PYTHON:?source FORMAL_PATHS.env first}"

CHECKPOINT=""
PARENT_MANIFEST=""
GPU="1"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="${2:?missing value for --checkpoint}"; shift 2 ;;
    --parent-manifest) PARENT_MANIFEST="${2:?missing value for --parent-manifest}"; shift 2 ;;
    --gpu) GPU="${2:?missing value for --gpu}"; shift 2 ;;
    *) printf 'unsupported argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

BACKEND="$PATHVLM_REPO/formal_machine/run_formal_validation.py"
[[ -f "$BACKEND" ]] || {
  printf 'formal validation backend is not yet implemented/audited: %s\n' "$BACKEND" >&2
  exit 3
}

exec "$PATHVLM_SFT_PYTHON" "$BACKEND" \
  --repo-root "$PATHVLM_REPO" \
  --install-root "$PATHVLM_INSTALL_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --parent-manifest "$PARENT_MANIFEST" \
  --gpu "$GPU"
