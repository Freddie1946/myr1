#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 JUDGE CAP_USD OUTPUT_DIR" >&2
  exit 2
fi

JUDGE=$1
CAP_USD=$2
OUTPUT_DIR=$3
WORK_ROOT=/home/dataset-assist-0/czy/wjy
REPO=$WORK_ROOT/myr1
SMOKE=$WORK_ROOT/pathvlm_revision_eval_a100/runs/multijudge_reasoning_eval_20260810/smoke_results.jsonl

set -a
source "$WORK_ROOT/.secrets/aigcbest.env"
set +a

cd "$REPO"
exec python3 scripts/run_multijudge_reasoning_evaluation.py \
  --manifest protocol/multijudge_reasoning_eval_current_models_20260810.json \
  --output-dir "$OUTPUT_DIR" \
  --mode full \
  --budget-limit-usd "$CAP_USD" \
  --judges "$JUDGE" \
  --reuse-smoke-results "$SMOKE"
