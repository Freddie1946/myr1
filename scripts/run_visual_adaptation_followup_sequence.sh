#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "usage: $0 <run-root>" >&2; exit 2; }
ROOT=$1
REPO=/home/dataset-assist-0/czy/wjy/myr1
LAUNCHER=$REPO/scripts/launch_visual_adaptation_sft_arm.sh
PYTHON=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/sft/bin/python

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "refusing to start: GPU compute process is active" >&2
  exit 3
fi
mkdir -p "$ROOT/logs"
for seed in 42; do
  for arm in l a; do
    name=${arm}_seed${seed}
    config=$ROOT/$name/config.yaml
    output=$ROOT/$name/output
    log=$ROOT/logs/$name.log
    [[ -f "$config" && ! -e "$output" ]] || {
      echo "invalid pre-launch state: $name" >&2
      exit 4
    }
    TMPDIR=/dev/shm \
    HF_DATASETS_CACHE=/home/dataset-assist-0/czy/wjy/.cache/huggingface-wjy/datasets_visual_adapt_followup \
      bash "$LAUNCHER" 0,1,2,3,4,5,6,7 8 "$((30620 + RANDOM % 1000))" "$config" "$log"
    [[ -f "$output/train_results.json" && -f "$output/adapter_model.safetensors" ]] || {
      echo "missing final output: $name" >&2
      exit 5
    }
    checkpoint_count=$(find "$output" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | wc -l)
    [[ "$checkpoint_count" -ge 5 ]] || {
      echo "too few retained checkpoints for $name: $checkpoint_count" >&2
      exit 6
    }
  done
done

printf '{"status":"training_completed","arms":["l_seed42","a_seed42"],"seed43_executed":false}\n' \
  >"$ROOT/training_sequence_complete.json"
