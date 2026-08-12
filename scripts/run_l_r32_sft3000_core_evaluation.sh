#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT=${1:-/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/rule_rl_mechanism_funnel_20260811/formal_v1/phase3/l_r32_sft3000}
EVAL_ROOT=${2:-/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/rule_rl_mechanism_funnel_20260811/formal_v1/l_r32_sft_core}
REPO=/home/dataset-assist-0/czy/wjy/myr1
PYTHON=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/sft/bin/python
MODEL=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/models/Qwen2.5-VL-7B-Instruct-cc594898137f460bfe9f0759e9844b3ce807cfb5
VAL=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/pathmmu_image_disjoint_v2/rewritten_records/validation_0385.json
L16_EVAL=/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/formal_selected_sft3000_20260811/pathmmu

[[ -f "$TRAIN_ROOT/output/train_results.json" ]] || { echo "missing L-r32 train result" >&2; exit 2; }
for step in 16 32 48 64 80 96; do
  [[ -f "$TRAIN_ROOT/output/checkpoint-$step/adapter_model.safetensors" ]] || exit 3
done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU compute process is active" >&2
  exit 4
fi
mkdir -p "$EVAL_ROOT/pathmmu_val" "$EVAL_ROOT/logs"

# Bound cold-start concurrency to avoid a six-process filesystem/model-loading
# spike. Completed metrics are resumable sentinels; partial outputs fail closed.
steps=(16 32 48 64 80 96)
for ((start=0; start<${#steps[@]}; start+=3)); do
  pids=()
  for ((offset=0; offset<3 && start+offset<${#steps[@]}; offset++)); do
    step=${steps[$((start+offset))]}
    id=step$(printf '%03d' "$step")
    output=$EVAL_ROOT/pathmmu_val/$id
    if [[ -f "$output/metrics.json" ]]; then
      continue
    fi
    [[ ! -e "$output" ]] || { echo "partial output blocks $id" >&2; exit 5; }
    CUDA_VISIBLE_DEVICES=$offset PYTHONPATH=$REPO/scripts "$PYTHON" \
      "$REPO/scripts/run_pathmmu_qwen_diagnostic.py" \
      --model "$MODEL" --adapter "$TRAIN_ROOT/output/checkpoint-$step" \
      --backend qwen2_5_vl --data "$VAL" \
      --output-dir "$output" --batch-size 32 \
      --split-role validation_smoke --max-new-tokens 1024 \
      >"$EVAL_ROOT/logs/${id}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  ((failed == 0)) || exit 6
done

"$PYTHON" - "$TRAIN_ROOT" "$EVAL_ROOT" "$L16_EVAL" <<'PY'
import json
import sys
from pathlib import Path

train_root, eval_root, l16_root = map(Path, sys.argv[1:])
final_state = json.loads((train_root / "output/trainer_state.json").read_text())
loss_by_step = {
    int(row["step"]): float(row["loss"])
    for row in final_state["log_history"]
    if "step" in row and "loss" in row
}
rows = []
for step in (16, 32, 48, 64, 80, 96):
    metrics = json.loads((eval_root / "pathmmu_val" / f"step{step:03d}/metrics.json").read_text())
    l16 = json.loads((l16_root / f"step{step:03d}/metrics.json").read_text())
    rows.append({
        "step": step,
        "epoch": step / 32,
        "l_r32_training_loss_at_step": loss_by_step.get(step),
        "l_r32_correct": metrics["correct"],
        "l_r32_count": metrics["count"],
        "l_r32_accuracy": metrics["accuracy"],
        "l_r32_format_rate": metrics["format_correct"] / metrics["count"],
        "l_r32_generation_cap_hits": metrics["generation_cap_hit_count"],
        "l_r16_correct": l16["correct"],
        "l_r16_accuracy": l16["accuracy"],
        "r32_minus_r16_accuracy": metrics["accuracy"] - l16["accuracy"],
    })
best = max(row["l_r32_accuracy"] for row in rows)
eligible = [row for row in rows if row["l_r32_accuracy"] >= best - 0.01 - 1e-12]
shortlist = [row["step"] for row in sorted(eligible, key=lambda row: (-row["l_r32_accuracy"], row["step"]))[:3]]
payload = {
    "schema_version": 1,
    "status": "l_r32_pathmmu_core_complete",
    "selection_data": "PathMMU validation385 only",
    "test_accessed": False,
    "shortlist_rule": "up to three checkpoints within 1 percentage point of the L-r32 PathMMU-validation best",
    "shortlisted_steps": shortlist,
    "rows": rows,
}
(eval_root / "core_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "L-r32 core evaluation complete: $EVAL_ROOT/core_summary.json"
