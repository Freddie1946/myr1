#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT=/home/dataset-assist-0/czy/wjy
REPO="$WORK_ROOT/myr1"
TRAIN_ROOT="$WORK_ROOT/pathvlm_r1_v1_a100/runs/visual_adaptation_sft_pilot_20260811/formal_fast_gbs96_e10"
BASE_MODEL="$WORK_ROOT/pathvlm_r1_v1_a100/models/Qwen2.5-VL-7B-Instruct-cc594898137f460bfe9f0759e9844b3ce807cfb5"
PATHVQA_DATA="$WORK_ROOT/pathvlm_revision_eval_a100/datasets/pathvqa_architecture_validation_v1_20260811/pathvqa_validation_balanced_image_unique_512.json"
PATHMMU_DATA="$WORK_ROOT/pathvlm_r1_v1_a100/data/pathmmu_image_disjoint_v2/rewritten_records/validation_0385.json"
RUN_ROOT="$WORK_ROOT/pathvlm_revision_eval_a100/runs/visual_adaptation_architecture_validation_20260811"
PYTHON="$WORK_ROOT/pathvlm_r1_v1_a100/envs/sft/bin/python"

mkdir -p "$RUN_ROOT/logs"
if [[ -e "$RUN_ROOT/sequence_complete.json" ]]; then
  echo "validation sequence is already complete" >&2
  exit 1
fi

models=(base c0 l a b2 b4)
gpus=(0 1 2 3 4 5)

model_path() {
  case "$1" in
    base|l|a|b2|b4) printf '%s\n' "$BASE_MODEL" ;;
    c0) printf '%s\n' "$TRAIN_ROOT/c0/output" ;;
  esac
}

adapter_args() {
  case "$1" in
    l|a|b2|b4) printf '%s\n' "--adapter" "$TRAIN_ROOT/$1/output" ;;
  esac
}

run_pathvqa() {
  local index="$1" name="${models[$1]}" gpu="${gpus[$1]}"
  local output="$RUN_ROOT/pathvqa/$name/original"
  [[ ! -e "$output" ]] || { echo "refusing existing output: $output" >&2; return 1; }
  mapfile -t adapter < <(adapter_args "$name")
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO/scripts" "$PYTHON" \
    "$REPO/scripts/run_pathvqa_forced_binary_logits.py" \
    --model "$(model_path "$name")" "${adapter[@]}" \
    --data "$PATHVQA_DATA" --output-dir "$output" --batch-size 32 \
    --split-role validation_diagnostic \
    >"$RUN_ROOT/logs/pathvqa_${name}.log" 2>&1
}

run_pathmmu() {
  local index="$1" name="${models[$1]}" gpu="${gpus[$1]}"
  local output="$RUN_ROOT/pathmmu/$name"
  [[ ! -e "$output" ]] || { echo "refusing existing output: $output" >&2; return 1; }
  mapfile -t adapter < <(adapter_args "$name")
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO/scripts" "$PYTHON" \
    "$REPO/scripts/run_pathmmu_qwen_diagnostic.py" \
    --model "$(model_path "$name")" "${adapter[@]}" --backend qwen2_5_vl \
    --data "$PATHMMU_DATA" --output-dir "$output" --batch-size 8 \
    --split-role validation_smoke --max-new-tokens 1024 \
    >"$RUN_ROOT/logs/pathmmu_${name}.log" 2>&1
}

wait_phase() {
  local phase="$1" failed=0 pid
  shift
  for pid in "$@"; do
    if ! wait "$pid"; then
      echo "$phase job PID $pid failed" >&2
      failed=1
    fi
  done
  (( failed == 0 ))
}

pathvqa_pids=()
for index in "${!models[@]}"; do
  run_pathvqa "$index" &
  pathvqa_pids+=("$!")
done
wait_phase pathvqa "${pathvqa_pids[@]}"

PYTHONPATH="$REPO/scripts" "$PYTHON" "$REPO/scripts/summarize_pathvqa_forced_binary.py" \
  --run-root "$RUN_ROOT/pathvqa" --base-name base \
  --output "$RUN_ROOT/pathvqa/paired_summary.json" --seed 42 \
  >"$RUN_ROOT/logs/pathvqa_summary.log" 2>&1

pathmmu_pids=()
for index in "${!models[@]}"; do
  run_pathmmu "$index" &
  pathmmu_pids+=("$!")
done
wait_phase pathmmu "${pathmmu_pids[@]}"

for name in c0 l a b2 b4; do
  PYTHONPATH="$REPO/scripts" "$PYTHON" "$REPO/scripts/analyze_paired_closed_benchmark.py" \
    --left "$RUN_ROOT/pathmmu/base/predictions.jsonl" \
    --right "$RUN_ROOT/pathmmu/$name/predictions.jsonl" \
    --left-label base --right-label "$name" --benchmark pathmmu_validation385 \
    --correctness-field accuracy_reward --expected-count 385 --replicates 10000 --seed 42 \
    --output-dir "$RUN_ROOT/pathmmu/paired_base_vs_$name"
done

"$PYTHON" - "$RUN_ROOT" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "status": "completed",
    "formal_result": False,
    "role": "rapid_architecture_validation",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "pathvqa_summary": str((root / "pathvqa/paired_summary.json").resolve()),
    "pathmmu_models": ["base", "c0", "l", "a", "b2", "b4"],
}
(root / "sequence_complete.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
