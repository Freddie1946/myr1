#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 5 ]] || {
  echo "usage: $0 <cuda-visible-devices> <nproc> <master-port> <config> <log>" >&2
  exit 2
}

GPU_LIST=$1
NPROC=$2
MASTER_PORT=$3
CONFIG=$4
LOG=$5

[[ -f "$CONFIG" ]] || { echo "missing config: $CONFIG" >&2; exit 2; }
[[ "$NPROC" =~ ^[1-8]$ ]] || { echo "invalid nproc: $NPROC" >&2; exit 2; }
mkdir -p "$(dirname "$LOG")"

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export PYTHONPATH="/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/sources/LLaMA-Factory/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

exec /home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/sft/bin/python \
  -m torch.distributed.run \
  --nproc_per_node="$NPROC" \
  --master_port="$MASTER_PORT" \
  /home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/sources/LLaMA-Factory/src/llamafactory/launcher.py \
  "$CONFIG" 2>&1 | tee "$LOG"
