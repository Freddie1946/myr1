#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT=${1:-/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/formal_selected_rule_rl1000_corrected_20260811/formal_z2_mbs10_lr1e6_len384}
EVAL_ROOT=${2:-/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/formal_selected_rule_rl1000_corrected_20260811}
REPO=/home/dataset-assist-0/czy/wjy/myr1
PYTHON=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/grpo/bin/python
MODEL=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/formal_selected_rule_rl1000_20260811/parents/sft_step080_merged
PATHMMU_TEST=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/pathmmu_image_disjoint_v2/rewritten_records/test_0999.json
PATHVQA_TEST=/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/external_vqa_contract_v1_20260729/pathvqa_test_6719.json
OMNIMED_TEST=/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/external_vqa_contract_v1_20260729/omnimedvqa_four_sources_8518.json

SELECTION=$EVAL_ROOT/validation_selection.json
[[ -f "$SELECTION" ]] || { echo "validation selection is not frozen" >&2; exit 2; }
selected_step=$("$PYTHON" -c "import json; print(json.load(open('$SELECTION'))['selected_step'])")
[[ "$selected_step" == 100 || "$selected_step" == 150 ]] || exit 2
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU compute process is active" >&2
  exit 3
fi
TEST_ROOT=$EVAL_ROOT/full_tests
[[ ! -e "$TEST_ROOT" ]] || { echo "existing full-test root: $TEST_ROOT" >&2; exit 4; }
mkdir -p "$TEST_ROOT/pathmmu_test999" "$TEST_ROOT/pathvqa_test_yesno3362_generated" \
  "$TEST_ROOT/pathvqa_test_yesno3362_forced_logits" "$TEST_ROOT/omnimedvqa_test8518" \
  "$TEST_ROOT/logs"

wait_all() {
  local failed=0 pid
  for pid in "$@"; do
    if ! wait "$pid"; then failed=1; fi
  done
  ((failed == 0))
}

# Both saved checkpoints are reported. Test is strictly post-selection and cannot change
# selected_step. The eight independent evaluations use one A100 each.
pids=()
for spec in 100:0 150:1; do
  step=${spec%%:*}; gpu=${spec#*:}; adapter=$TRAIN_ROOT/output/checkpoint-$step
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO/scripts "$PYTHON" \
    "$REPO/scripts/run_pathmmu_qwen_diagnostic.py" \
    --model "$MODEL" --adapter "$adapter" --backend qwen2_5_vl \
    --data "$PATHMMU_TEST" --output-dir "$TEST_ROOT/pathmmu_test999/step$step" \
    --batch-size 32 --split-role test999_development --max-new-tokens 1024 \
    >"$TEST_ROOT/logs/step${step}_pathmmu_test999.log" 2>&1 &
  pids+=("$!")
done
for spec in 100:2 150:3; do
  step=${spec%%:*}; gpu=${spec#*:}; adapter=$TRAIN_ROOT/output/checkpoint-$step
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO/scripts "$PYTHON" \
    "$REPO/scripts/run_external_vqa_qwen.py" \
    --task pathvqa --model "$MODEL" --adapter "$adapter" --backend qwen2_5_vl \
    --data "$PATHVQA_TEST" --output-dir "$TEST_ROOT/pathvqa_test_yesno3362_generated/step$step" \
    --split-role external_test --pathvqa-answer-scope yes_no_only \
    --generation-contract pathvqa_domain_think_answer_v5_2048 \
    --max-new-tokens 2048 --batch-size 16 \
    >"$TEST_ROOT/logs/step${step}_pathvqa_generated.log" 2>&1 &
  pids+=("$!")
done
for spec in 100:4 150:5; do
  step=${spec%%:*}; gpu=${spec#*:}; adapter=$TRAIN_ROOT/output/checkpoint-$step
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO/scripts "$PYTHON" \
    "$REPO/scripts/run_external_vqa_qwen.py" \
    --task omnimedvqa --model "$MODEL" --adapter "$adapter" --backend qwen2_5_vl \
    --data "$OMNIMED_TEST" --output-dir "$TEST_ROOT/omnimedvqa_test8518/step$step" \
    --split-role external_test \
    --generation-contract omnimed_domain_think_answer_v4_1024 \
    --max-new-tokens 1024 --batch-size 16 \
    >"$TEST_ROOT/logs/step${step}_omnimedvqa.log" 2>&1 &
  pids+=("$!")
done
for spec in 100:6 150:7; do
  step=${spec%%:*}; gpu=${spec#*:}; adapter=$TRAIN_ROOT/output/checkpoint-$step
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO/scripts "$PYTHON" \
    "$REPO/scripts/run_pathvqa_forced_binary_logits.py" \
    --model "$MODEL" --adapter "$adapter" --data "$PATHVQA_TEST" \
    --output-dir "$TEST_ROOT/pathvqa_test_yesno3362_forced_logits/step$step" \
    --batch-size 64 --split-role post_hoc_test_sensitivity --image-mode original \
    >"$TEST_ROOT/logs/step${step}_pathvqa_forced.log" 2>&1 &
  pids+=("$!")
done
wait_all "${pids[@]}" || exit 5

"$PYTHON" - "$EVAL_ROOT" "$selected_step" <<'PY'
import json, sys
from pathlib import Path

root, selected_step = Path(sys.argv[1]), int(sys.argv[2])
test = root / "full_tests"
rows = []
for step in (100, 150):
    pm = json.loads((test / "pathmmu_test999" / f"step{step}" / "metrics.json").read_text())
    pv = json.loads((test / "pathvqa_test_yesno3362_generated" / f"step{step}" / "metrics.json").read_text())
    pf = json.loads((test / "pathvqa_test_yesno3362_forced_logits" / f"step{step}" / "metrics.json").read_text())
    om = json.loads((test / "omnimedvqa_test8518" / f"step{step}" / "metrics.json").read_text())
    rows.append({
        "step": step,
        "selected_before_test": step == selected_step,
        "pathmmu_test999": {k: pm[k] for k in ("correct", "count", "accuracy", "format_correct", "generation_cap_hit_count")},
        "pathvqa_test_yesno3362_generated": {
            "correct": pv["contract_aligned_exact_correct"],
            "count": pv["count"],
            "accuracy": pv["paper_yes_no_contract_aligned_accuracy"],
            "generation_cap_hit_count": pv["generation_cap_hit_count"],
            "maximum_generated_tokens": pv["maximum_generated_tokens"],
        },
        "pathvqa_test_yesno3362_forced_logits_diagnostic": {k: pf[k] for k in ("correct", "count", "accuracy")},
        "omnimedvqa_test8518": {
            "correct": om["strict_final_correct"],
            "count": om["count"],
            "accuracy": om["strict_final_accuracy"],
            "answer_coverage": om["strict_final_answer_coverage"],
            "generation_cap_hit_count": om["generation_cap_hit_count"],
            "maximum_generated_tokens": om["maximum_generated_tokens"],
        },
    })
payload = {
    "schema_version": 1,
    "status": "corrected_rule_rl_full_tests_complete",
    "test_used_for_checkpoint_selection": False,
    "selected_step_frozen_before_test": selected_step,
    "pathvqa_open_questions_excluded_from_primary_table": True,
    "rows": rows,
}
(test / "complete_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

echo "corrected rule-RL full tests complete: $TEST_ROOT"
