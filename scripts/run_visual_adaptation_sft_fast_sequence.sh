#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/visual_adaptation_sft_pilot_20260811/formal_fast_gbs96_e10}
REPO=/home/dataset-assist-0/czy/wjy/myr1
LAUNCHER=$REPO/scripts/launch_visual_adaptation_sft_arm.sh
VERIFY=$REPO/scripts/verify_visual_adaptation_sft_smoke.py
PYTHON=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/sft/bin/python

[[ -x "$PYTHON" && -f "$LAUNCHER" && -f "$VERIFY" ]] || exit 2
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "refusing to start: at least one GPU compute process is active" >&2
  exit 3
fi

mkdir -p "$ROOT/logs"
for arm in c0 a b2 b4; do
  config=$ROOT/$arm/config.yaml
  manifest=$ROOT/$arm/manifest.json
  output=$ROOT/$arm/output
  log=$ROOT/logs/${arm}.log
  [[ -f "$config" && -f "$manifest" && ! -e "$output" ]] || {
    echo "invalid pre-launch state for arm $arm" >&2
    exit 4
  }
  TMPDIR=/dev/shm \
  HF_DATASETS_CACHE=/home/dataset-assist-0/czy/wjy/.cache/huggingface-wjy/datasets_visual_adapt_fast \
    bash "$LAUNCHER" 0,1,2,3,4,5,6,7 8 "$((29620 + RANDOM % 1000))" \
      "$config" "$log"
  [[ -f "$output/train_results.json" ]] || {
    echo "missing train results for arm $arm" >&2
    exit 5
  }
  if [[ "$arm" == c0 ]]; then
    [[ -f "$output/model.safetensors.index.json" ]] || {
      echo "missing gathered C0 model" >&2
      exit 6
    }
  else
    "$PYTHON" "$VERIFY" \
      --manifest "$manifest" \
      --output-dir "$output" \
      --report "$ROOT/$arm/trainability_verification.json"
  fi
done

printf 'all visual-adaptation SFT arms completed\n'
