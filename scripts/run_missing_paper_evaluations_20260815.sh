#!/usr/bin/env bash
set -euo pipefail

WORK=/home/dataset-assist-0/czy/wjy
REPO="$WORK/myr1"
PY="$WORK/pathvlm_r1_v1_a100/envs/grpo/bin/python"
EXT="$WORK/pathvlm_revision_eval_a100/datasets/external_vqa_contract_v1_20260729"
STAGE3="$WORK/pathvlm_r1_v1_a100/runs/stage3_process_grpo/gpt4o_n8_parent_full_20260813/epoch_model_snapshots"
STAGE2_CONT="$WORK/pathvlm_r1_v1_a100/runs/data_ratio_rule_rl_ablation_v1/formal_sequence/tasks/stage2_continue_rule_rl1000/output/checkpoint-1500"
OUT="$WORK/pathvlm_revision_eval_a100/runs/missing_paper_evaluations_20260815"
LOG="$OUT/logs"
mkdir -p "$LOG"
cd "$REPO"

resume_args() {
  local directory=$1
  if [[ -f "$directory/predictions.jsonl" ]]; then
    printf '%s\n' --resume
  fi
}

validate_smoke() {
  local directory=$1 expected_contract=$2
  "$PY" - "$directory" "$expected_contract" <<'PY'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
expected_contract = sys.argv[2]
metrics = json.loads((directory / "metrics.json").read_text())
config = json.loads((directory / "run_config.json").read_text())
if metrics.get("count") != 16:
    raise SystemExit(f"unexpected smoke count: {metrics.get('count')}")
if config.get("generation_contract") != expected_contract:
    raise SystemExit("generation contract mismatch")
if metrics.get("empty_completion_count", 0) != 0:
    raise SystemExit("empty completion in smoke")
if metrics.get("generation_cap_hit_count", 0) != 0:
    raise SystemExit("generation cap hit in smoke")
if expected_contract.startswith("pathvqa_"):
    parsed = metrics.get("paper_yes_no_contract_aligned_parsed")
    if parsed is not None and parsed < 15:
        raise SystemExit(f"insufficient PathVQA parseability: {parsed}/16")
print({"smoke": str(directory), "status": "passed", "metrics": metrics})
PY
}

run_stage3_yesno() {
  local gpu=$1 step=$2
  local model="$STAGE3/checkpoint-$step"
  local root="$OUT/stage3_gpt4o_checkpoint${step}_pathvqa_yesno_v5"
  local smoke="$root/smoke16"
  local full="$root/full3362"
  if [[ ! -f "$smoke/metrics.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_external_vqa_qwen.py \
      --task pathvqa --model "$model" --backend qwen2_5_vl \
      --data "$EXT/pathvqa_test_6719.json" --output-dir "$smoke" \
      --split-role adapter_smoke --limit 16 --batch-size 16 --max-new-tokens 2048 \
      --generation-contract pathvqa_domain_think_answer_v5_2048 \
      --pathvqa-answer-scope yes_no_only
  fi
  validate_smoke "$smoke" pathvqa_domain_think_answer_v5_2048
  if [[ ! -f "$full/metrics.json" ]]; then
    mapfile -t resume < <(resume_args "$full")
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_external_vqa_qwen.py \
      --task pathvqa --model "$model" --backend qwen2_5_vl \
      --data "$EXT/pathvqa_test_6719.json" --output-dir "$full" \
      --split-role external_test --batch-size 16 --max-new-tokens 2048 \
      --generation-contract pathvqa_domain_think_answer_v5_2048 \
      --pathvqa-answer-scope yes_no_only "${resume[@]}"
  fi
}

run_stage2_omni() {
  local gpu=$1
  local root="$OUT/stage2_continued_rule_rl1000_omnimedvqa_v4"
  local smoke="$root/smoke16"
  local full="$root/full8518"
  if [[ ! -f "$smoke/metrics.json" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_external_vqa_qwen.py \
      --task omnimedvqa --model "$STAGE2_CONT" --backend qwen2_5_vl \
      --data "$EXT/omnimedvqa_four_sources_8518.json" --output-dir "$smoke" \
      --split-role adapter_smoke --limit 16 --batch-size 16 --max-new-tokens 1024 \
      --generation-contract omnimed_domain_think_answer_v4_1024 \
      --pathvqa-answer-scope all
  fi
  validate_smoke "$smoke" omnimed_domain_think_answer_v4_1024
  if [[ ! -f "$full/metrics.json" ]]; then
    mapfile -t resume < <(resume_args "$full")
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/run_external_vqa_qwen.py \
      --task omnimedvqa --model "$STAGE2_CONT" --backend qwen2_5_vl \
      --data "$EXT/omnimedvqa_four_sources_8518.json" --output-dir "$full" \
      --split-role external_test --batch-size 16 --max-new-tokens 1024 \
      --generation-contract omnimed_domain_think_answer_v4_1024 \
      --pathvqa-answer-scope all "${resume[@]}"
  fi
}

pids=()
run_stage3_yesno 0 500 >"$LOG/stage3_step500_yesno.log" 2>&1 & pids+=("$!")
run_stage3_yesno 1 1000 >"$LOG/stage3_step1000_yesno.log" 2>&1 & pids+=("$!")
run_stage3_yesno 2 1500 >"$LOG/stage3_step1500_yesno.log" 2>&1 & pids+=("$!")
run_stage2_omni 3 >"$LOG/stage2_continue_omni.log" 2>&1 & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  printf 'One or more lanes failed; completed predictions remain resumable.\n' >&2
  exit "$status"
fi

printf 'completed_at=%(%Y-%m-%dT%H:%M:%S%z)T\n' -1 > "$OUT/COMPLETED"
