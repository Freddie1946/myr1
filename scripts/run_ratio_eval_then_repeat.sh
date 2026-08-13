#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 GPU_ID RATIO_LABEL RATIO_MODEL REPEAT_MODEL [REPEAT_ADAPTER|-]" >&2
  exit 2
fi

GPU_ID="$1"
RATIO_LABEL="$2"
RATIO_MODEL="$3"
REPEAT_MODEL="$4"
REPEAT_ADAPTER="${5:--}"

WORK=/home/dataset-assist-0/czy/wjy
REPO="$WORK/myr1"
PY="$WORK/pathvlm_r1_v1_a100/envs/grpo/bin/python"
DATA="$WORK/pathvlm_r1_v1_a100/data/pathmmu_image_disjoint_v2/rewritten_records"
EXT="$WORK/pathvlm_revision_eval_a100/datasets/external_vqa_contract_v1_20260729"
MMMU="$WORK/pathvlm_revision_eval_a100/datasets/mmmu_nonmedical_dev_retention_v1_20260811/panel.json"
OUT="$WORK/pathvlm_revision_eval_a100/runs/data_ratio_uniform_eval_20260814/$RATIO_LABEL"
REP="$WORK/pathvlm_revision_eval_a100/runs/repeated_inference_pathmmu_test999_20260814/$RATIO_LABEL"

cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

pathmmu() {
  local split="$1" role="$2" out="$3"
  if [[ -f "$out/metrics.json" ]]; then return; fi
  local resume=()
  if [[ -f "$out/predictions.jsonl" ]]; then resume=(--resume); fi
  "$PY" scripts/run_pathmmu_qwen_diagnostic.py \
    --model "$RATIO_MODEL" --backend qwen2_5_vl --data "$DATA/$split" \
    --output-dir "$out" --split-role "$role" --batch-size 16 --max-new-tokens 1024 \
    "${resume[@]}"
}

external() {
  local task="$1" data="$2" out="$3" cap="$4" contract="$5" scope="$6"
  if [[ -f "$out/metrics.json" ]]; then return; fi
  local resume=()
  if [[ -f "$out/predictions.jsonl" ]]; then resume=(--resume); fi
  "$PY" scripts/run_external_vqa_qwen.py \
    --task "$task" --model "$RATIO_MODEL" --backend qwen2_5_vl --data "$data" \
    --output-dir "$out" --split-role external_test --max-new-tokens "$cap" \
    --generation-contract "$contract" --batch-size 16 --pathvqa-answer-scope "$scope" \
    "${resume[@]}"
}

pathmmu validation_0385.json validation_smoke "$OUT/pathmmu_val385"
pathmmu test_0999.json test999_development "$OUT/pathmmu_test999"
external pathvqa "$EXT/pathvqa_test_6719.json" "$OUT/pathvqa_test3362_ab" 2048 pathvqa_pathmmu_ab_v6_2048 yes_no_only

if [[ ! -f "$OUT/mmmu_nonmedical116/metrics.json" ]]; then
  resume=()
  if [[ -f "$OUT/mmmu_nonmedical116/predictions.jsonl" ]]; then resume=(--resume); fi
  "$PY" scripts/run_mmmu_retention_diagnostic.py --model "$RATIO_MODEL" --panel "$MMMU" \
    --output-dir "$OUT/mmmu_nonmedical116" --batch-size 8 --max-new-tokens 4096 "${resume[@]}"
fi

external omnimedvqa "$EXT/omnimedvqa_four_sources_8518.json" "$OUT/omnimedvqa_8518" 1024 omnimed_domain_think_answer_v4_1024 all

for seed in 42 43 44 45 46; do
  run="$REP/seed_$seed"
  if [[ -f "$run/metrics.json" ]]; then continue; fi
  resume=()
  if [[ -f "$run/predictions.jsonl" ]]; then resume=(--resume); fi
  adapter=()
  if [[ "$REPEAT_ADAPTER" != "-" ]]; then adapter=(--adapter "$REPEAT_ADAPTER"); fi
  "$PY" scripts/run_pathmmu_qwen_diagnostic.py \
    --model "$REPEAT_MODEL" --backend qwen2_5_vl --data "$DATA/test_0999.json" \
    --output-dir "$run" --split-role test999_development --batch-size 16 \
    --max-new-tokens 1024 --do-sample --seed "$seed" --temperature 0.7 --top-p 0.9 --top-k 0 \
    "${adapter[@]}" \
    "${resume[@]}"
done
"$PY" scripts/summarize_repeated_inference.py --root "$REP" --output "$REP/summary.json" --expected-runs 5
