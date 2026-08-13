#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 GPU_ID LABEL MODEL OUTPUT_ROOT [ADAPTER|-]" >&2
  exit 2
fi

GPU_ID="$1"
LABEL="$2"
MODEL="$3"
OUTPUT_ROOT="$4"
ADAPTER="${5:--}"
WORK=/home/dataset-assist-0/czy/wjy
REPO="$WORK/myr1"
PY="$WORK/pathvlm_r1_v1_a100/envs/grpo/bin/python"
DATA="$WORK/pathvlm_r1_v1_a100/data/pathmmu_image_disjoint_v2/rewritten_records/test_0999.json"
ROOT="$OUTPUT_ROOT/$LABEL"

cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export TOKENIZERS_PARALLELISM=false

for seed in 42 43 44 45 46; do
  run="$ROOT/seed_$seed"
  if [[ -f "$run/metrics.json" ]]; then continue; fi
  resume=()
  if [[ -f "$run/predictions.jsonl" ]]; then resume=(--resume); fi
  adapter=()
  if [[ "$ADAPTER" != "-" ]]; then adapter=(--adapter "$ADAPTER"); fi
  "$PY" scripts/run_pathmmu_qwen_diagnostic.py \
    --model "$MODEL" --backend qwen2_5_vl --data "$DATA" \
    --output-dir "$run" --split-role test999_development --batch-size 16 \
    --max-new-tokens 1024 --do-sample --seed "$seed" \
    --temperature 0.7 --top-p 0.9 --top-k 0 \
    "${adapter[@]}" "${resume[@]}"
done

"$PY" scripts/summarize_repeated_inference.py \
  --root "$ROOT" --output "$ROOT/summary.json" --expected-runs 5
