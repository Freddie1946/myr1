#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT=${1:-/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/formal_selected_rule_rl1000_20260811/formal_z2_gc0_mbs10_len384}
EVAL_ROOT=${2:-/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/formal_selected_rule_rl1000_20260811}
REPO=/home/dataset-assist-0/czy/wjy/myr1
# The RL adapters were written by PEFT 0.19.1 and contain fields unknown to the
# older SFT evaluation environment (PEFT 0.12.0). Use the training-compatible
# environment so adapter loading is lossless rather than rewriting its config.
PYTHON=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/grpo/bin/python
MODEL=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/formal_selected_rule_rl1000_20260811/parents/sft_step080_merged
PATHMMU_VAL=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/pathmmu_image_disjoint_v2/rewritten_records/validation_0385.json
PATHMMU_TEST=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/pathmmu_image_disjoint_v2/rewritten_records/test_0999.json
PATHVQA_VAL=/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/pathvqa_architecture_validation_v1_20260811/pathvqa_validation_balanced_image_unique_512.json
PATHVQA_TEST=/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/external_vqa_contract_v1_20260729/pathvqa_test_6719.json
MMMU=/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/mmmu_nonmedical_dev_retention_v1_20260811/panel.json

[[ -f "$TRAIN_ROOT/output/train_results.json" && -f "$TRAIN_ROOT/output/pathvlm_train_state_audit.json" ]] || exit 2
for step in 100 150; do
  [[ -f "$TRAIN_ROOT/output/checkpoint-$step/adapter_model.safetensors" ]] || exit 3
done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU compute process is active" >&2
  exit 4
fi
[[ ! -e "$EVAL_ROOT" ]] || { echo "existing evaluation root: $EVAL_ROOT" >&2; exit 5; }
mkdir -p "$EVAL_ROOT/pathmmu_val" "$EVAL_ROOT/pathvqa_val" "$EVAL_ROOT/mmmu" "$EVAL_ROOT/logs"

wait_all() {
  local failed=0 pid
  for pid in "$@"; do
    if ! wait "$pid"; then failed=1; fi
  done
  ((failed == 0))
}

# Phase 1: validation and retention for both saved RL checkpoints.
pids=()
for spec in 100:0 150:1; do
  step=${spec%%:*}; gpu=${spec#*:}; adapter=$TRAIN_ROOT/output/checkpoint-$step
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO/scripts "$PYTHON" \
    "$REPO/scripts/run_pathmmu_qwen_diagnostic.py" \
    --model "$MODEL" --adapter "$adapter" --backend qwen2_5_vl \
    --data "$PATHMMU_VAL" --output-dir "$EVAL_ROOT/pathmmu_val/step$step" \
    --batch-size 32 --split-role validation_smoke --max-new-tokens 1024 \
    >"$EVAL_ROOT/logs/step${step}_pathmmu_val.log" 2>&1 &
  pids+=("$!")
done
for spec in 100:2 150:3; do
  step=${spec%%:*}; gpu=${spec#*:}; adapter=$TRAIN_ROOT/output/checkpoint-$step
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO/scripts "$PYTHON" \
    "$REPO/scripts/run_pathvqa_forced_binary_logits.py" \
    --model "$MODEL" --adapter "$adapter" --data "$PATHVQA_VAL" \
    --output-dir "$EVAL_ROOT/pathvqa_val/step$step/original" --batch-size 64 \
    --split-role validation_diagnostic --image-mode original \
    >"$EVAL_ROOT/logs/step${step}_pathvqa_val.log" 2>&1 &
  pids+=("$!")
done
for spec in 100:4 150:5; do
  step=${spec%%:*}; gpu=${spec#*:}; adapter=$TRAIN_ROOT/output/checkpoint-$step
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO/scripts "$PYTHON" \
    "$REPO/scripts/run_mmmu_retention_diagnostic.py" \
    --model "$MODEL" --adapter "$adapter" --panel "$MMMU" \
    --output-dir "$EVAL_ROOT/mmmu/step$step" --batch-size 4 --max-new-tokens 4096 \
    >"$EVAL_ROOT/logs/step${step}_mmmu.log" 2>&1 &
  pids+=("$!")
done
wait_all "${pids[@]}" || exit 6

# Freeze the primary reporting checkpoint using validation only. First keep checkpoints within
# one percentage point of best PathMMU validation, then maximize PathVQA retention; earliest step
# breaks exact ties. Test results cannot alter this choice.
"$PYTHON" - "$EVAL_ROOT" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for step in (100, 150):
    pm = json.loads((root / "pathmmu_val" / f"step{step}" / "metrics.json").read_text())
    pv = json.loads((root / "pathvqa_val" / f"step{step}" / "original" / "metrics.json").read_text())
    mm = json.loads((root / "mmmu" / f"step{step}" / "metrics.json").read_text())
    rows.append({
        "step": step,
        "pathmmu_correct": pm["correct"], "pathmmu_count": pm["count"],
        "pathmmu_accuracy": pm["accuracy"], "pathmmu_format_correct": pm["format_correct"],
        "pathmmu_cap_hits": pm["generation_cap_hit_count"],
        "pathvqa_correct": pv["correct"], "pathvqa_count": pv["count"],
        "pathvqa_accuracy": pv["accuracy"],
        "mmmu_correct": mm["correct"], "mmmu_count": mm["count"],
        "mmmu_accuracy": mm["accuracy"], "mmmu_cap_hits": mm["generation_cap_hit_count"],
    })
best = max(row["pathmmu_accuracy"] for row in rows)
eligible = [row for row in rows if row["pathmmu_accuracy"] >= best - 0.01]
selected = max(eligible, key=lambda row: (row["pathvqa_accuracy"], -row["step"]))
payload = {
    "schema_version": 1,
    "status": "validation_complete_reporting_checkpoint_frozen_before_test",
    "selection_rule": "PathMMU validation within 1 pp of best, then highest PathVQA validation retention, earliest exact tie",
    "test_used_for_selection": False,
    "rows": rows,
    "eligible_steps": [row["step"] for row in eligible],
    "selected_step": selected["step"],
}
(root / "validation_selection.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

# Phase 2: reporting-only complete tests for both checkpoints. Two PathMMU lanes use GPUs 0-1;
# three PathVQA shards per checkpoint use GPUs 2-7.
mkdir -p "$EVAL_ROOT/pathmmu_test999" "$EVAL_ROOT/pathvqa_test_yesno3362"
pids=()
for spec in 100:0 150:1; do
  step=${spec%%:*}; gpu=${spec#*:}; adapter=$TRAIN_ROOT/output/checkpoint-$step
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO/scripts "$PYTHON" \
    "$REPO/scripts/run_pathmmu_qwen_diagnostic.py" \
    --model "$MODEL" --adapter "$adapter" --backend qwen2_5_vl \
    --data "$PATHMMU_TEST" --output-dir "$EVAL_ROOT/pathmmu_test999/step$step" \
    --batch-size 32 --split-role test999_development --max-new-tokens 1024 \
    >"$EVAL_ROOT/logs/step${step}_pathmmu_test999.log" 2>&1 &
  pids+=("$!")
done
gpu=2
for step in 100 150; do
  adapter=$TRAIN_ROOT/output/checkpoint-$step
  mkdir -p "$EVAL_ROOT/pathvqa_test_yesno3362/step$step/shards"
  for shard in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO/scripts "$PYTHON" \
      "$REPO/scripts/run_pathvqa_forced_binary_logits.py" \
      --model "$MODEL" --adapter "$adapter" --data "$PATHVQA_TEST" \
      --output-dir "$EVAL_ROOT/pathvqa_test_yesno3362/step$step/shards/shard_$shard" \
      --batch-size 64 --shard-index "$shard" --shard-count 3 \
      --split-role post_hoc_test_sensitivity --image-mode original \
      >"$EVAL_ROOT/logs/step${step}_pathvqa_test_shard_${shard}.log" 2>&1 &
    pids+=("$!"); gpu=$((gpu + 1))
  done
done
wait_all "${pids[@]}" || exit 7
for step in 100 150; do
  "$PYTHON" "$REPO/scripts/merge_pathvqa_forced_binary_shards.py" \
    --shard-root "$EVAL_ROOT/pathvqa_test_yesno3362/step$step/shards" \
    --output-dir "$EVAL_ROOT/pathvqa_test_yesno3362/step$step/merged"
done

# Phase 3: visual-dependence controls for the validation-selected checkpoint.
selected_step=$("$PYTHON" -c "import json; print(json.load(open('$EVAL_ROOT/validation_selection.json'))['selected_step'])")
adapter=$TRAIN_ROOT/output/checkpoint-$selected_step
pids=()
for spec in cyclic_mismatch:0 global_mean_blank:1; do
  mode=${spec%%:*}; gpu=${spec#*:}
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO/scripts "$PYTHON" \
    "$REPO/scripts/run_pathvqa_forced_binary_logits.py" \
    --model "$MODEL" --adapter "$adapter" --data "$PATHVQA_VAL" \
    --output-dir "$EVAL_ROOT/pathvqa_val/step$selected_step/$mode" --batch-size 64 \
    --split-role validation_diagnostic --image-mode "$mode" \
    >"$EVAL_ROOT/logs/step${selected_step}_pathvqa_${mode}.log" 2>&1 &
  pids+=("$!")
done
wait_all "${pids[@]}" || exit 8

"$PYTHON" - "$EVAL_ROOT" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
selection = json.loads((root / "validation_selection.json").read_text())
rows = []
for step in (100, 150):
    pmv = json.loads((root / "pathmmu_val" / f"step{step}" / "metrics.json").read_text())
    pvv = json.loads((root / "pathvqa_val" / f"step{step}" / "original" / "metrics.json").read_text())
    mmmu = json.loads((root / "mmmu" / f"step{step}" / "metrics.json").read_text())
    pmt = json.loads((root / "pathmmu_test999" / f"step{step}" / "metrics.json").read_text())
    pvt = json.loads((root / "pathvqa_test_yesno3362" / f"step{step}" / "merged" / "metrics.json").read_text())
    rows.append({
        "step": step,
        "pathmmu_val": {"correct": pmv["correct"], "count": pmv["count"], "accuracy": pmv["accuracy"], "format_correct": pmv["format_correct"], "cap_hits": pmv["generation_cap_hit_count"]},
        "pathvqa_val": {"correct": pvv["correct"], "count": pvv["count"], "accuracy": pvv["accuracy"]},
        "mmmu": {"correct": mmmu["correct"], "count": mmmu["count"], "accuracy": mmmu["accuracy"], "cap_hits": mmmu["generation_cap_hit_count"]},
        "pathmmu_test999": {"correct": pmt["correct"], "count": pmt["count"], "accuracy": pmt["accuracy"], "format_correct": pmt["format_correct"], "cap_hits": pmt["generation_cap_hit_count"]},
        "pathvqa_test_yesno3362": {"correct": pvt["correct"], "count": pvt["count"], "accuracy": pvt["accuracy"], "coverage": pvt["coverage"]},
    })
step = selection["selected_step"]
normal = json.loads((root / "pathvqa_val" / f"step{step}" / "original" / "metrics.json").read_text())
shuffle = json.loads((root / "pathvqa_val" / f"step{step}" / "cyclic_mismatch" / "metrics.json").read_text())
blank = json.loads((root / "pathvqa_val" / f"step{step}" / "global_mean_blank" / "metrics.json").read_text())
payload = {
    "schema_version": 1,
    "status": "formal_selected_rule_rl1000_evaluation_complete",
    "test_used_for_selection": False,
    "selected_step": step,
    "selection": selection,
    "rows": rows,
    "selected_visual_dependence": {
        "normal_accuracy": normal["accuracy"],
        "shuffle_accuracy": shuffle["accuracy"],
        "blank_accuracy": blank["accuracy"],
        "normal_minus_shuffle": normal["accuracy"] - shuffle["accuracy"],
        "normal_minus_blank": normal["accuracy"] - blank["accuracy"],
    },
    "pathvqa_open_question_results_excluded": True,
}
(root / "complete_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

echo "formal selected rule-RL1000 evaluation complete: $EVAL_ROOT"
