#!/usr/bin/env bash
set -euo pipefail
umask 077

ROOT=/home/dataset-assist-0/czy/wjy
REPO=$ROOT/myr1
INSTALL=$ROOT/pathvlm_r1_v1_a100
EVAL=$ROOT/pathvlm_revision_eval_a100
PYTHON=$EVAL/envs/llama32_vision/bin/python
MODEL=$INSTALL/models/Llama-3.2-11B-Vision-Instruct-modelscope-master
DATA=$EVAL/datasets/external_vqa_contract_v1_20260729/pathvqa_test_6719.json
RUN=$EVAL/runs/local_gpu_baseline_gap_completion_20260810/llama32_11b/pathvqa_yesno
SMOKE=$RUN/smoke16
FULL=$RUN/full3362
DATA_SHA=5fa7319784dd47c81027c2b99cbb0d0d472638d0dc5c5169dcbd5d43987784be
MODEL_SHA=db708a72a246253fdf8aa3837a5f0b80ac1960692495ce05cb3101833128e845

export CUDA_VISIBLE_DEVICES=6
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false PYTHONPATH=$REPO/scripts
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

if [[ ! -f "$SMOKE/metrics.json" ]]; then
  RESUME=()
  [[ ! -e "$SMOKE" ]] || RESUME=(--resume)
  "$PYTHON" "$REPO/scripts/run_external_vqa_qwen.py" \
    --task pathvqa --model "$MODEL" --backend mllama --data "$DATA" \
    --output-dir "$SMOKE" --split-role adapter_smoke --batch-size 2 --limit 16 \
    --pathvqa-answer-scope yes_no_only "${RESUME[@]}"
fi

"$PYTHON" "$REPO/scripts/verify_local_baseline_smoke.py" \
  --task pathvqa --metrics "$SMOKE/metrics.json" --predictions "$SMOKE/predictions.jsonl" \
  --expected-count 16 --expected-data-sha256 "$DATA_SHA" \
  --expected-model-config-sha256 "$MODEL_SHA" --expected-pathvqa-answer-scope yes_no_only \
  --minimum-nonempty-rate 0.80 --minimum-parseable-rate 0.80 --maximum-cap-hit-rate 0.30 \
  --output "$SMOKE/smoke_gate.json"

if [[ ! -f "$FULL/metrics.json" ]]; then
  RESUME=()
  [[ ! -e "$FULL" ]] || RESUME=(--resume)
  "$PYTHON" "$REPO/scripts/run_external_vqa_qwen.py" \
    --task pathvqa --model "$MODEL" --backend mllama --data "$DATA" \
    --output-dir "$FULL" --split-role external_test --batch-size 2 \
    --pathvqa-answer-scope yes_no_only "${RESUME[@]}"
fi

"$PYTHON" "$REPO/scripts/verify_local_baseline_full.py" \
  --task pathvqa --metrics "$FULL/metrics.json" --predictions "$FULL/predictions.jsonl" \
  --run-config "$FULL/run_config.json" --expected-count 3362 --expected-split-role external_test \
  --expected-data-sha256 "$DATA_SHA" --expected-model-config-sha256 "$MODEL_SHA" \
  --expected-pathvqa-answer-scope yes_no_only --output "$FULL/full_integrity_verified.json"
