#!/usr/bin/env bash
set -euo pipefail
umask 077

WORK_ROOT=/home/dataset-assist-0/czy/wjy
REPO=$WORK_ROOT/myr1
INSTALL=$WORK_ROOT/pathvlm_r1_v1_a100
EVAL=$WORK_ROOT/pathvlm_revision_eval_a100
PYTHON=$INSTALL/envs/sft/bin/python
REVISION=ce7f51daa9731f106874ac2bee0e9a864f7a3636
MODEL=$EVAL/models/ChiPhan1110--ScaleReasoner-R1--$REVISION
DATA=$INSTALL/data/pathmmu_image_disjoint_v2/rewritten_records/test_0999.json
VALIDATION=$INSTALL/data/pathmmu_image_disjoint_v2/rewritten_records/validation_0385.json
RUN_ROOT=$EVAL/runs/pathology_baselines_contemporary_20260810/scalereasoner_r1
SMOKE=$RUN_ROOT/validation_smoke16
FULL=$RUN_ROOT/pathmmu_test999

export HF_HOME=$WORK_ROOT/cache/huggingface
hf download ChiPhan1110/ScaleReasoner-R1 \
  --revision "$REVISION" \
  --local-dir "$MODEL"

export CUDA_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false PYTHONPATH=$REPO/scripts
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

if [[ ! -f "$SMOKE/metrics.json" ]]; then
  [[ ! -e "$SMOKE" ]] || { echo "Incomplete ScaleReasoner smoke exists: $SMOKE" >&2; exit 3; }
  "$PYTHON" "$REPO/scripts/run_pathmmu_qwen_diagnostic.py" \
    --model "$MODEL" --backend qwen2_5_vl \
    --data "$VALIDATION" --output-dir "$SMOKE" \
    --split-role validation_smoke --batch-size 1 --limit 16
fi

"$PYTHON" "$REPO/scripts/verify_local_baseline_smoke.py" \
  --task pathmmu \
  --metrics "$SMOKE/metrics.json" \
  --predictions "$SMOKE/predictions.jsonl" \
  --expected-count 16 \
  --expected-data-sha256 f52c9a412da579db8fb8bf52f6fa320e7c4ae9cb42b6cf1bdf62f77ab158dcf0 \
  --expected-model-config-sha256 "$(sha256sum "$MODEL/config.json" | awk '{print $1}')" \
  --minimum-nonempty-rate 0.80 \
  --minimum-parseable-rate 0.80 \
  --maximum-cap-hit-rate 0.20 \
  --output "$SMOKE/smoke_gate.json"

RESUME=()
[[ ! -e "$FULL" ]] || RESUME=(--resume)
if [[ ! -f "$FULL/metrics.json" ]]; then
  "$PYTHON" "$REPO/scripts/run_pathmmu_qwen_diagnostic.py" \
    --model "$MODEL" --backend qwen2_5_vl \
    --data "$DATA" --output-dir "$FULL" \
    --split-role test999_development --batch-size 1 \
    "${RESUME[@]}"
fi

"$PYTHON" "$REPO/scripts/verify_local_baseline_full.py" \
  --task pathmmu \
  --metrics "$FULL/metrics.json" \
  --predictions "$FULL/predictions.jsonl" \
  --run-config "$FULL/run_config.json" \
  --expected-count 999 \
  --expected-split-role test999_development \
  --expected-data-sha256 3420c8ff2f1642801a5ecebd21271f4e7de5877d6459814473ddbcdda996b549 \
  --expected-model-config-sha256 "$(sha256sum "$MODEL/config.json" | awk '{print $1}')" \
  --output "$FULL/full_integrity_verified.json"
