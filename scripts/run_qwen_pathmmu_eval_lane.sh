#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -ne 5 ]]; then
  echo "usage: $0 GPU MODEL DATA OUTPUT SPLIT_ROLE" >&2
  exit 2
fi

GPU=$1
MODEL=$(readlink -f "$2")
DATA=$(readlink -f "$3")
OUTPUT=$(readlink -m "$4")
SPLIT_ROLE=$5
WORK_ROOT=/home/dataset-assist-0/czy/wjy
REPO=$WORK_ROOT/myr1
PYTHON=$WORK_ROOT/pathvlm_r1_v1_a100/envs/sft/bin/python

[[ "$GPU" =~ ^[0-7]$ ]] || { echo "invalid GPU: $GPU" >&2; exit 2; }
[[ "$SPLIT_ROLE" == "validation_smoke" || "$SPLIT_ROLE" == "test999_development" ]] || {
  echo "invalid split role: $SPLIT_ROLE" >&2; exit 2
}
[[ -f "$MODEL/config.json" && -f "$DATA" ]] || { echo "model or data is missing" >&2; exit 2; }
case "$OUTPUT" in
  "$WORK_ROOT/pathvlm_revision_eval_a100/runs"/*) ;;
  *) echo "output must be under the evaluation runs root" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES=$GPU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false PYTHONPATH=$REPO/scripts
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

RESUME=()
[[ ! -e "$OUTPUT" ]] || RESUME=(--resume)
exec "$PYTHON" "$REPO/scripts/run_pathmmu_qwen_diagnostic.py" \
  --model "$MODEL" --backend qwen2_5_vl \
  --data "$DATA" --output-dir "$OUTPUT" --split-role "$SPLIT_ROLE" \
  --batch-size 1 "${RESUME[@]}"
