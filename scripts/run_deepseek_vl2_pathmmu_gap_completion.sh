#!/usr/bin/env bash
set -euo pipefail
umask 077

ROOT=/home/dataset-assist-0/czy/wjy
REPO=$ROOT/myr1
INSTALL=$ROOT/pathvlm_r1_v1_a100
EVAL=$ROOT/pathvlm_revision_eval_a100
PYTHON=$EVAL/envs/deepseek_vl2/bin/python
MODEL=$EVAL/models/deepseek-ai--deepseek-vl2--f363772d1c47f4239dd844015b4bd53beb87951b
VALIDATION=$INSTALL/data/pathmmu_image_disjoint_v2/rewritten_records/validation_0385.json
TEST=$INSTALL/data/pathmmu_image_disjoint_v2/rewritten_records/test_0999.json
RUN=$EVAL/runs/local_gpu_baseline_gap_completion_20260810/deepseek_vl2/pathmmu_letter_only_v2
SMOKE=$RUN/smoke16
FULL=$RUN/test999
MODEL_SHA=d5dadaf4d41af00d569ac376bf09de4e28324c853b3ae0c7f6fed66f19fdcf8c
VALIDATION_SHA=f52c9a412da579db8fb8bf52f6fa320e7c4ae9cb42b6cf1bdf62f77ab158dcf0
TEST_SHA=3420c8ff2f1642801a5ecebd21271f4e7de5877d6459814473ddbcdda996b549

export CUDA_VISIBLE_DEVICES=${DEEPSEEK_VISIBLE_GPUS:-3,4}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false PYTHONPATH=$REPO/scripts:$EVAL/sources/DeepSeek-VL2
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

if [[ ! -f "$SMOKE/metrics.json" ]]; then
  RESUME=(); [[ ! -e "$SMOKE" ]] || RESUME=(--resume)
  "$PYTHON" "$REPO/scripts/run_pathmmu_deepseek_vl2_diagnostic.py" \
    --model "$MODEL" --data "$VALIDATION" --output-dir "$SMOKE" \
    --split-role validation_smoke --limit 16 --prompt-contract letter_only_v2 \
    --max-new-tokens 32 --device-map-auto "${RESUME[@]}"
fi
"$PYTHON" "$REPO/scripts/verify_local_baseline_smoke.py" \
  --task pathmmu --metrics "$SMOKE/metrics.json" --predictions "$SMOKE/predictions.jsonl" \
  --expected-count 16 --expected-data-sha256 "$VALIDATION_SHA" \
  --expected-model-config-sha256 "$MODEL_SHA" --minimum-nonempty-rate 0.80 \
  --minimum-parseable-rate 0.80 --maximum-cap-hit-rate 0.20 \
  --output "$SMOKE/smoke_gate.json"

if [[ ! -f "$FULL/metrics.json" ]]; then
  RESUME=(); [[ ! -e "$FULL" ]] || RESUME=(--resume)
  "$PYTHON" "$REPO/scripts/run_pathmmu_deepseek_vl2_diagnostic.py" \
    --model "$MODEL" --data "$TEST" --output-dir "$FULL" \
    --split-role test999_development --prompt-contract letter_only_v2 \
    --max-new-tokens 32 --device-map-auto "${RESUME[@]}"
fi
"$PYTHON" "$REPO/scripts/verify_local_baseline_full.py" \
  --task pathmmu --metrics "$FULL/metrics.json" --predictions "$FULL/predictions.jsonl" \
  --run-config "$FULL/run_config.json" --expected-count 999 \
  --expected-split-role test999_development --expected-data-sha256 "$TEST_SHA" \
  --expected-model-config-sha256 "$MODEL_SHA" --output "$FULL/full_integrity_verified.json"
