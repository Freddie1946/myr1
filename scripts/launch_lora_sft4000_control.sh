#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 4 ]] || {
  echo "usage: $0 <run-dir> <gpu-list> <nproc> <master-port>" >&2
  exit 2
}
RUN_DIR=$1
GPU_LIST=$2
NPROC=$3
MASTER_PORT=$4
REPO=/home/dataset-assist-0/czy/wjy/myr1
OUTPUT="$RUN_DIR/output"
LOG="$RUN_DIR/train.log"

[[ -f "$RUN_DIR/config.yaml" && -f "$RUN_DIR/manifest.json" ]] || exit 3
[[ ! -e "$OUTPUT" ]] || { echo "output already exists: $OUTPUT" >&2; exit 4; }
TMPDIR=/dev/shm \
HF_DATASETS_CACHE=/home/dataset-assist-0/czy/wjy/.cache/huggingface-wjy/datasets_lora_sft4000_control \
  bash "$REPO/scripts/launch_visual_adaptation_sft_arm.sh" \
  "$GPU_LIST" "$NPROC" "$MASTER_PORT" "$RUN_DIR/config.yaml" "$LOG"

[[ -f "$OUTPUT/train_results.json" && -f "$OUTPUT/adapter_model.safetensors" ]] || exit 5
