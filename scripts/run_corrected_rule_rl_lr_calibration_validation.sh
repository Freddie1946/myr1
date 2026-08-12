#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/corrected_rule_rl_lr_calibration_20260811
EVAL_ROOT=/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/corrected_rule_rl_lr_calibration_20260811
REPO=/home/dataset-assist-0/czy/wjy/myr1
PYTHON=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/grpo/bin/python
MODEL=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/formal_selected_rule_rl1000_20260811/parents/sft_step080_merged
PATHMMU_VAL=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/pathmmu_image_disjoint_v2/rewritten_records/validation_0385.json
PATHVQA_VAL=/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/pathvqa_architecture_validation_v1_20260811/pathvqa_validation_balanced_image_unique_512.json

[[ ! -e "$EVAL_ROOT" ]] || { echo "existing evaluation root: $EVAL_ROOT" >&2; exit 2; }
for arm in lr1e6 lr3e6; do
  [[ -f "$TRAIN_ROOT/$arm/output/checkpoint-10/adapter_model.safetensors" ]] || exit 3
  [[ -f "$TRAIN_ROOT/$arm/reward_alignment_verification.json" ]] || exit 3
done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU compute process is active" >&2
  exit 4
fi
mkdir -p "$EVAL_ROOT/pathmmu_val" "$EVAL_ROOT/pathvqa_val" "$EVAL_ROOT/logs"

pids=()
for specification in lr1e6:0 lr3e6:1; do
  IFS=: read -r arm gpu <<<"$specification"
  adapter=$TRAIN_ROOT/$arm/output/checkpoint-10
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO/scripts "$PYTHON" \
    "$REPO/scripts/run_pathmmu_qwen_diagnostic.py" \
    --model "$MODEL" --adapter "$adapter" --backend qwen2_5_vl \
    --data "$PATHMMU_VAL" --output-dir "$EVAL_ROOT/pathmmu_val/$arm" \
    --batch-size 32 --split-role validation_smoke --max-new-tokens 1024 \
    >"$EVAL_ROOT/logs/${arm}_pathmmu_val.log" 2>&1 &
  pids+=("$!")
done
for specification in lr1e6:2 lr3e6:3; do
  IFS=: read -r arm gpu <<<"$specification"
  adapter=$TRAIN_ROOT/$arm/output/checkpoint-10
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO/scripts "$PYTHON" \
    "$REPO/scripts/run_pathvqa_forced_binary_logits.py" \
    --model "$MODEL" --adapter "$adapter" --data "$PATHVQA_VAL" \
    --output-dir "$EVAL_ROOT/pathvqa_val/$arm" --batch-size 64 \
    --split-role validation_diagnostic --image-mode original \
    >"$EVAL_ROOT/logs/${arm}_pathvqa_val.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
(( failed == 0 )) || exit 5

"$PYTHON" - "$TRAIN_ROOT" "$EVAL_ROOT" <<'PY'
import json, statistics, sys
from pathlib import Path

train_root, eval_root = map(Path, sys.argv[1:])
rows = []
for arm in ("lr1e6", "lr3e6"):
    audit = json.loads((train_root / arm / "output" / "pathvlm_train_state_audit.json").read_text())
    alignment = json.loads((train_root / arm / "reward_alignment_verification.json").read_text())
    pm = json.loads((eval_root / "pathmmu_val" / arm / "metrics.json").read_text())
    pv = json.loads((eval_root / "pathvqa_val" / arm / "metrics.json").read_text())
    logs = [row for row in audit["log_history"] if "grad_norm" in row]
    rows.append({
        "arm": arm,
        "learning_rate": 1e-6 if arm == "lr1e6" else 3e-6,
        "completed_steps": audit["global_step"],
        "alignment_passed": alignment["passed"],
        "mean_accuracy_reward": statistics.mean(row["rewards/audited_accuracy_reward"] for row in logs),
        "mean_format_reward": statistics.mean(row["rewards/audited_format_reward"] for row in logs),
        "mean_reward_std": statistics.mean(row["reward_std"] for row in logs),
        "mean_kl_steps2_10": statistics.mean(row["kl"] for row in logs[1:]),
        "final_kl": logs[-1]["kl"],
        "mean_grad_norm": statistics.mean(row["grad_norm"] for row in logs),
        "max_grad_norm": max(row["grad_norm"] for row in logs),
        "pathmmu_val": {"correct": pm["correct"], "count": pm["count"], "accuracy": pm["accuracy"], "format_correct": pm["format_correct"], "cap_hits": pm["generation_cap_hit_count"]},
        "pathvqa_val": {"correct": pv["correct"], "count": pv["count"], "accuracy": pv["accuracy"], "coverage": pv["coverage"]},
    })
payload = {
    "schema_version": 1,
    "status": "calibration_training_and_validation_complete_selection_pending",
    "formal_result": False,
    "test_used": False,
    "rows": rows,
}
(eval_root / "calibration_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
