#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--approval-check" ]]; then
  printf '%s\n' \
    'APPROVAL SCOPE: audited formal SFT runs only' \
    'ALLOWED CONFIGS: seed 42, n0500/n1000/n2000/n3000' \
    'FORMAL GPUS: physical devices 0,1,2,3,4,5,6,7 only' \
    'GUARDS: fresh occupancy check, validation/test isolation, stop on failure'
  exit 0
fi

: "${PATHVLM_REPO:?source FORMAL_PATHS.env first}"
: "${PATHVLM_INSTALL_ROOT:?source FORMAL_PATHS.env first}"
: "${PATHVLM_SFT_PYTHON:?source FORMAL_PATHS.env first}"
: "${PATHVLM_LLAMAFACTORY_SRC:?source FORMAL_PATHS.env first}"

CONFIG=""
CUDA_VISIBLE="0,1,2,3,4,5,6,7"
NPROC=8
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="${2:?missing value for --config}"; shift 2 ;;
    --cuda-visible-devices) CUDA_VISIBLE="${2:?missing value for --cuda-visible-devices}"; shift 2 ;;
    --nproc-per-node) NPROC="${2:?missing value for --nproc-per-node}"; shift 2 ;;
    *) printf 'unsupported argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

case "$(basename "$CONFIG")" in
  sft_n0500_seed0042.yaml|sft_n1000_seed0042.yaml|sft_n2000_seed0042.yaml|sft_n3000_seed0042.yaml) ;;
  *) printf 'config is outside the authorized seed-42 SFT scale matrix: %s\n' "$CONFIG" >&2; exit 2 ;;
esac

BACKEND="$PATHVLM_REPO/formal_machine/run_formal_sft.py"
[[ -f "$BACKEND" ]] || {
  printf 'formal SFT backend is not yet implemented/audited: %s\n' "$BACKEND" >&2
  exit 3
}

exec "$PATHVLM_SFT_PYTHON" "$BACKEND" \
  --repo-root "$PATHVLM_REPO" \
  --install-root "$PATHVLM_INSTALL_ROOT" \
  --llamafactory-src "$PATHVLM_LLAMAFACTORY_SRC" \
  --config "$CONFIG" \
  --cuda-visible-devices "$CUDA_VISIBLE" \
  --nproc-per-node "$NPROC"
