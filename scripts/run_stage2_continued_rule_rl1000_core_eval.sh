#!/usr/bin/env bash
set -euo pipefail

WORK=/home/dataset-assist-0/czy/wjy
REPO="$WORK/myr1"
PY="$WORK/pathvlm_r1_v1_a100/envs/grpo/bin/python"
MODEL="$WORK/pathvlm_r1_v1_a100/runs/data_ratio_rule_rl_ablation_v1/formal_sequence/tasks/stage2_continue_rule_rl1000/output/checkpoint-1500"
PATHMMU="$WORK/pathvlm_r1_v1_a100/data/pathmmu_image_disjoint_v2/rewritten_records"
EXT="$WORK/pathvlm_revision_eval_a100/datasets/external_vqa_contract_v1_20260729"
OUT="$WORK/pathvlm_revision_eval_a100/runs/stage2_continued_rule_rl1000_core_eval_20260815"

mkdir -p "$OUT"
cd "$REPO"

resume_arg() {
  local out=$1
  if [[ -f "$out/predictions.jsonl" ]]; then
    printf '%s\n' --resume
  fi
}

run_pathmmu() {
  local gpu=$1 split=$2 role=$3 out=$4
  [[ -f "$out/metrics.json" ]] && return 0
  mapfile -t resume < <(resume_arg "$out")
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_pathmmu_qwen_diagnostic.py \
    --model "$MODEL" --backend qwen2_5_vl --data "$PATHMMU/$split" \
    --output-dir "$out" --split-role "$role" --batch-size 16 \
    --max-new-tokens 1024 "${resume[@]}"
}

run_pathvqa() {
  local gpu=$1 out=$2 contract=$3
  [[ -f "$out/metrics.json" ]] && return 0
  mapfile -t resume < <(resume_arg "$out")
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_external_vqa_qwen.py \
    --task pathvqa --model "$MODEL" --backend qwen2_5_vl \
    --data "$EXT/pathvqa_test_6719.json" --output-dir "$out" \
    --split-role external_test --batch-size 16 --max-new-tokens 2048 \
    --generation-contract "$contract" --pathvqa-answer-scope yes_no_only \
    "${resume[@]}"
}

# OmniMedVQA is deliberately excluded until the manuscript-faithful common
# contract is frozen. These four lanes are already stable across core models.
run_pathmmu 0 validation_0385.json validation_smoke "$OUT/pathmmu_val385" &
pids=("$!")
run_pathmmu 1 test_0999.json test999_development "$OUT/pathmmu_test999" &
pids+=("$!")
run_pathvqa 3 "$OUT/pathvqa_test3362_yesno" pathvqa_domain_think_answer_v5_2048 &
pids+=("$!")
run_pathvqa 5 "$OUT/pathvqa_test3362_ab" pathvqa_pathmmu_ab_v6_2048 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "One or more Stage2 continued-rule-RL evaluation lanes failed; completed lanes remain resumable." >&2
  exit "$status"
fi

printf 'completed_at=%(%Y-%m-%dT%H:%M:%S%z)T\n' -1 > "$OUT/CORE_COMPLETED"
