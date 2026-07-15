#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--approval-check" ]]; then
  printf '%s\n' \
    'APPROVAL SCOPE: attach n0500, then sequential validation/n1000/n2000/n3000 only' \
    'FAIL-CLOSED: any missing/failed/ambiguous gate stops the sequence' \
    'GPU POLICY: never kill processes; wait for all eight GPUs to be idle' \
    'DATA POLICY: SFT train plus validation_0385 only; test is forbidden' \
    'END: stop after n3000 validation; no Stage 2 or Stage 3'
  exit 0
fi

: "${PATHVLM_REPO:?source FORMAL_PATHS.env first}"
: "${PATHVLM_INSTALL_ROOT:?source FORMAL_PATHS.env first}"
: "${PATHVLM_SFT_PYTHON:?source FORMAL_PATHS.env first}"

CURRENT_MANIFEST=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --current-n0500-manifest)
      CURRENT_MANIFEST="${2:?missing value for --current-n0500-manifest}"
      shift 2
      ;;
    *) printf 'unsupported argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

EXPECTED_PREFIX="$PATHVLM_INSTALL_ROOT/runs/stage1_sft/n0500_seed0042/formal_sft_n0500_seed0042_"
case "$CURRENT_MANIFEST" in
  "$EXPECTED_PREFIX"*/run_manifest.yaml) ;;
  *) printf 'current manifest is outside the exact n0500 formal run layout: %s\n' "$CURRENT_MANIFEST" >&2; exit 2 ;;
esac

exec "$PATHVLM_SFT_PYTHON" \
  "$PATHVLM_REPO/formal_machine/run_formal_sft_scale_sequence.py" \
  --repo-root "$PATHVLM_REPO" \
  --install-root "$PATHVLM_INSTALL_ROOT" \
  --current-n0500-manifest "$CURRENT_MANIFEST" \
  --poll-seconds 60 \
  --validation-gpu 0 \
  --daemon
