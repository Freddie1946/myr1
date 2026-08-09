#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU TASK" >&2
  exit 2
fi

GPU=$1
TASK=$2
ROOT=/home/dataset-assist-0/czy/wjy
REPO=$ROOT/myr1
INSTALL=$ROOT/pathvlm_r1_v1_a100
EVAL=$ROOT/pathvlm_revision_eval_a100
PYTHON=$INSTALL/envs/sft/bin/python
REVISION=ce7f51daa9731f106874ac2bee0e9a864f7a3636
MODEL=$EVAL/models/ChiPhan1110--ScaleReasoner-R1--$REVISION
RUN_ROOT=$EVAL/runs/pathology_baselines_contemporary_20260810/scalereasoner_r1
MODEL_SHA=$(sha256sum "$MODEL/config.json" | awk '{print $1}')

[[ "$GPU" =~ ^[0-7]$ ]] || { echo "invalid GPU" >&2; exit 2; }
[[ "$TASK" == pathvqa || "$TASK" == omnimedvqa ]] || {
  echo "task must be pathvqa or omnimedvqa" >&2
  exit 2
}

export CUDA_VISIBLE_DEVICES=$GPU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false PYTHONPATH=$REPO/scripts
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

if [[ "$TASK" == pathvqa ]]; then
  DATA=$EVAL/datasets/external_vqa_contract_v1_20260729/pathvqa_test_6719.json
  DATA_SHA=5fa7319784dd47c81027c2b99cbb0d0d472638d0dc5c5169dcbd5d43987784be
  FULL_COUNT=3362
  SCOPE_ARGS=(--pathvqa-answer-scope yes_no_only)
else
  DATA=$EVAL/datasets/external_vqa_contract_v1_20260729/omnimedvqa_four_sources_8518.json
  DATA_SHA=04ba0790410cd5de4c95689d349574b102034a153afa00d952e351d3c13991a7
  FULL_COUNT=8518
  SCOPE_ARGS=()
fi

SMOKE=$RUN_ROOT/${TASK}_smoke16
FULL=$RUN_ROOT/${TASK}_full${FULL_COUNT}
if [[ ! -f "$SMOKE/metrics.json" ]]; then
  RESUME=(); [[ ! -e "$SMOKE" ]] || RESUME=(--resume)
  "$PYTHON" "$REPO/scripts/run_external_vqa_qwen.py" \
    --task "$TASK" --model "$MODEL" --backend qwen2_5_vl --data "$DATA" \
    --output-dir "$SMOKE" --split-role adapter_smoke --batch-size 8 --limit 16 \
    "${SCOPE_ARGS[@]}" "${RESUME[@]}"
fi
VERIFY_SCOPE=()
[[ "$TASK" != pathvqa ]] || VERIFY_SCOPE=(--expected-pathvqa-answer-scope yes_no_only)
"$PYTHON" "$REPO/scripts/verify_local_baseline_smoke.py" \
  --task "$TASK" --metrics "$SMOKE/metrics.json" --predictions "$SMOKE/predictions.jsonl" \
  --expected-count 16 --expected-data-sha256 "$DATA_SHA" \
  --expected-model-config-sha256 "$MODEL_SHA" --minimum-nonempty-rate 0.80 \
  --minimum-parseable-rate 0.80 --maximum-cap-hit-rate 0.20 \
  "${VERIFY_SCOPE[@]}" --output "$SMOKE/smoke_gate.json"

if [[ ! -f "$FULL/metrics.json" ]]; then
  RESUME=(); [[ ! -e "$FULL" ]] || RESUME=(--resume)
  "$PYTHON" "$REPO/scripts/run_external_vqa_qwen.py" \
    --task "$TASK" --model "$MODEL" --backend qwen2_5_vl --data "$DATA" \
    --output-dir "$FULL" --split-role external_test --batch-size 8 \
    "${SCOPE_ARGS[@]}" "${RESUME[@]}"
fi
"$PYTHON" "$REPO/scripts/verify_local_baseline_full.py" \
  --task "$TASK" --metrics "$FULL/metrics.json" --predictions "$FULL/predictions.jsonl" \
  --run-config "$FULL/run_config.json" --expected-count "$FULL_COUNT" \
  --expected-split-role external_test --expected-data-sha256 "$DATA_SHA" \
  --expected-model-config-sha256 "$MODEL_SHA" "${VERIFY_SCOPE[@]}" \
  --output "$FULL/full_integrity_verified.json"
