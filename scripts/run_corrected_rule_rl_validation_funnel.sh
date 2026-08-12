#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT=${1:-/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/formal_selected_rule_rl1000_corrected_20260811/formal_z2_mbs10_lr1e6_len384}
EVAL_ROOT=${2:-/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/formal_selected_rule_rl1000_corrected_20260811}
REPO=/home/dataset-assist-0/czy/wjy/myr1
PYTHON=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/grpo/bin/python
MODEL=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/formal_selected_rule_rl1000_20260811/parents/sft_step080_merged
PATHMMU_VAL=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/pathmmu_image_disjoint_v2/rewritten_records/validation_0385.json
PATHVQA_VAL=/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/pathvqa_architecture_validation_v1_20260811/pathvqa_validation_balanced_image_unique_512.json
MMMU=/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/mmmu_nonmedical_dev_retention_v1_20260811/panel.json
CONTRACT=domain_think_answer_v2_2048

[[ -f "$TRAIN_ROOT/output/train_results.json" ]] || exit 2
[[ -f "$TRAIN_ROOT/reward_alignment_verification.json" ]] || exit 2
for step in 100 150; do
  [[ -f "$TRAIN_ROOT/output/checkpoint-$step/adapter_model.safetensors" ]] || exit 3
done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU compute process is active" >&2
  exit 4
fi
[[ ! -e "$EVAL_ROOT" ]] || { echo "existing evaluation root: $EVAL_ROOT" >&2; exit 5; }
mkdir -p "$EVAL_ROOT/pathmmu_val" "$EVAL_ROOT/pathvqa_val_generated" \
  "$EVAL_ROOT/pathvqa_val_forced_logits" "$EVAL_ROOT/mmmu" "$EVAL_ROOT/logs"

wait_all() {
  local failed=0 pid
  for pid in "$@"; do
    if ! wait "$pid"; then failed=1; fi
  done
  ((failed == 0))
}

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
    "$REPO/scripts/run_mmmu_retention_diagnostic.py" \
    --model "$MODEL" --adapter "$adapter" --panel "$MMMU" \
    --output-dir "$EVAL_ROOT/mmmu/step$step" --batch-size 4 --max-new-tokens 4096 \
    >"$EVAL_ROOT/logs/step${step}_mmmu.log" 2>&1 &
  pids+=("$!")
done
for spec in 100:4 150:5; do
  step=${spec%%:*}; gpu=${spec#*:}; adapter=$TRAIN_ROOT/output/checkpoint-$step
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO/scripts "$PYTHON" \
    "$REPO/scripts/run_pathvqa_yesno_prompt_calibration.py" \
    --model "$MODEL" --adapter "$adapter" --panel "$PATHVQA_VAL" \
    --panel-role architecture_validation_512 \
    --output-root "$EVAL_ROOT/pathvqa_val_generated/step$step" --batch-size 16 \
    --image-mode original --prompt-contracts "$CONTRACT" \
    >"$EVAL_ROOT/logs/step${step}_pathvqa_generated.log" 2>&1 &
  pids+=("$!")
done
for spec in 100:6 150:7; do
  step=${spec%%:*}; gpu=${spec#*:}; adapter=$TRAIN_ROOT/output/checkpoint-$step
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO/scripts "$PYTHON" \
    "$REPO/scripts/run_pathvqa_forced_binary_logits.py" \
    --model "$MODEL" --adapter "$adapter" --data "$PATHVQA_VAL" \
    --output-dir "$EVAL_ROOT/pathvqa_val_forced_logits/step$step" --batch-size 64 \
    --split-role validation_diagnostic --image-mode original \
    >"$EVAL_ROOT/logs/step${step}_pathvqa_forced.log" 2>&1 &
  pids+=("$!")
done
wait_all "${pids[@]}" || exit 6

"$PYTHON" - "$EVAL_ROOT" "$CONTRACT" <<'PY'
import json, sys
from pathlib import Path

root, contract = Path(sys.argv[1]), sys.argv[2]
rows = []
for step in (100, 150):
    pm = json.loads((root / "pathmmu_val" / f"step{step}" / "metrics.json").read_text())
    pg = json.loads((root / "pathvqa_val_generated" / f"step{step}" / contract / "metrics.json").read_text())
    pf = json.loads((root / "pathvqa_val_forced_logits" / f"step{step}" / "metrics.json").read_text())
    mm = json.loads((root / "mmmu" / f"step{step}" / "metrics.json").read_text())
    row = {
        "step": step,
        "pathmmu": {k: pm[k] for k in ("correct", "count", "accuracy", "format_correct", "generation_cap_hit_count")},
        "pathvqa_generated": {k: pg[k] for k in ("correct", "count", "accuracy", "parseable_count", "parseable_rate", "generation_cap_hit_count", "maximum_generated_tokens")},
        "pathvqa_forced_logits_diagnostic": {k: pf[k] for k in ("correct", "count", "accuracy")},
        "mmmu": {k: mm[k] for k in ("correct", "count", "accuracy", "choice_extracted", "generation_cap_hit_count")},
    }
    row["generation_gate_passed"] = (
        row["pathmmu"]["format_correct"] == row["pathmmu"]["count"]
        and row["pathmmu"]["generation_cap_hit_count"] == 0
        and row["pathvqa_generated"]["parseable_rate"] >= 0.99
        and row["pathvqa_generated"]["generation_cap_hit_count"] == 0
    )
    rows.append(row)
if not all(row["generation_gate_passed"] for row in rows):
    status = "validation_complete_generation_gate_failed_no_selection"
    eligible, selected = [], None
else:
    best = max(row["pathmmu"]["accuracy"] for row in rows)
    eligible = [row for row in rows if row["pathmmu"]["accuracy"] >= best - 0.01]
    selected = max(eligible, key=lambda row: (row["pathvqa_generated"]["accuracy"], -row["step"]))
    status = "validation_complete_reporting_checkpoint_frozen_before_test"
payload = {
    "schema_version": 2,
    "status": status,
    "selection_rule": "PathMMU generated validation within 1 pp of best, then highest PathVQA generated semantic accuracy, earliest exact tie",
    "pathvqa_forced_logits_used_for_selection": False,
    "test_used_for_selection": False,
    "rows": rows,
    "eligible_steps": [row["step"] for row in eligible],
    "selected_step": selected["step"] if selected else None,
}
(root / "validation_selection.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
if selected is None:
    raise SystemExit(7)
PY

echo "corrected rule-RL validation funnel complete: $EVAL_ROOT"
