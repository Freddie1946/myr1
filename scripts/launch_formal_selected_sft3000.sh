#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "usage: $0 <run-dir>" >&2; exit 2; }
RUN_DIR=$1
REPO=/home/dataset-assist-0/czy/wjy/myr1
PYTHON=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/sft/bin/python
CONFIG=$RUN_DIR/config.yaml
OUTPUT=$RUN_DIR/output
LOG=$RUN_DIR/train.log
MIN_CHECKPOINT_COUNT=${MIN_CHECKPOINT_COUNT:-6}
[[ "$MIN_CHECKPOINT_COUNT" =~ ^[0-9]+$ ]] || exit 2

[[ -f "$CONFIG" && ! -e "$OUTPUT" ]] || { echo "invalid pre-launch state" >&2; exit 3; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU compute process is active" >&2
  exit 4
fi
TMPDIR=/dev/shm \
HF_DATASETS_CACHE=/home/dataset-assist-0/czy/wjy/.cache/huggingface-wjy/datasets_formal_selected_sft3000 \
  bash "$REPO/scripts/launch_visual_adaptation_sft_arm.sh" \
  0,1,2,3,4,5,6,7 8 32642 "$CONFIG" "$LOG"
[[ -f "$OUTPUT/train_results.json" && -f "$OUTPUT/adapter_model.safetensors" ]] || exit 5
checkpoint_count=$(find "$OUTPUT" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | wc -l)
[[ "$checkpoint_count" -ge "$MIN_CHECKPOINT_COUNT" ]] || { echo "too few checkpoints: $checkpoint_count" >&2; exit 6; }
