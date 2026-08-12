#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT=${1:-/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/formal_a_sft3000_mechanism_20260811/formal}
EVAL_ROOT=${2:-/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/formal_a_sft3000_mechanism_20260811}
REPO=/home/dataset-assist-0/czy/wjy/myr1
PYTHON=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/sft/bin/python
MODEL=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/models/Qwen2.5-VL-7B-Instruct-cc594898137f460bfe9f0759e9844b3ce807cfb5
PATHMMU_VAL=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/pathmmu_image_disjoint_v2/rewritten_records/validation_0385.json
PATHMMU_TEST=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/pathmmu_image_disjoint_v2/rewritten_records/test_0999.json
PATHVQA_VAL=/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/pathvqa_architecture_validation_v1_20260811/pathvqa_validation_balanced_image_unique_512.json
PATHVQA_TEST=/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/external_vqa_contract_v1_20260729/pathvqa_test_6719.json
TRAIN_PROBE=/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/visual_adaptation_followup_train_probe500_20260811/records_n0500.json
MMMU=/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/mmmu_nonmedical_dev_retention_v1_20260811/panel.json

[[ -f "$TRAIN_ROOT/output/train_results.json" && -f "$TRAIN_ROOT/trainability_verification.json" ]] || exit 2
for step in 16 32 48 64 80 96; do
  [[ -f "$TRAIN_ROOT/output/checkpoint-$step/adapter_model.safetensors" ]] || exit 3
done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU compute process is active" >&2
  exit 4
fi
mkdir -p "$EVAL_ROOT/pathmmu_val" "$EVAL_ROOT/pathvqa_val" "$EVAL_ROOT/logs"

wait_all() {
  local failed=0 pid
  for pid in "$@"; do
    if ! wait "$pid"; then failed=1; fi
  done
  ((failed == 0))
}

# Phase 1: the complete six-point PathMMU validation curve, one checkpoint per GPU.
pids=()
gpu=0
for step in 16 32 48 64 80 96; do
  adapter=$TRAIN_ROOT/output/checkpoint-$step
  output=$EVAL_ROOT/pathmmu_val/step$(printf '%03d' "$step")
  [[ ! -e "$output" ]] || { echo "existing PathMMU val output: $output" >&2; exit 5; }
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO/scripts "$PYTHON" \
    "$REPO/scripts/run_pathmmu_qwen_diagnostic.py" \
    --model "$MODEL" --adapter "$adapter" --backend qwen2_5_vl \
    --data "$PATHMMU_VAL" --output-dir "$output" --batch-size 32 \
    --split-role validation_smoke --max-new-tokens 1024 \
    >"$EVAL_ROOT/logs/step$(printf '%03d' "$step")_pathmmu_val.log" 2>&1 &
  pids+=("$!")
  gpu=$((gpu + 1))
done
wait_all "${pids[@]}" || exit 6

# Phase 2: format-neutral PathVQA normal-image trajectory for all six checkpoints.
pids=()
gpu=0
for step in 16 32 48 64 80 96; do
  adapter=$TRAIN_ROOT/output/checkpoint-$step
  output=$EVAL_ROOT/pathvqa_val/step$(printf '%03d' "$step")/original
  [[ ! -e "$output" ]] || { echo "existing PathVQA val output: $output" >&2; exit 7; }
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO/scripts "$PYTHON" \
    "$REPO/scripts/run_pathvqa_forced_binary_logits.py" \
    --model "$MODEL" --adapter "$adapter" --data "$PATHVQA_VAL" \
    --output-dir "$output" --batch-size 64 --split-role validation_diagnostic \
    --image-mode original >"$EVAL_ROOT/logs/step$(printf '%03d' "$step")_pathvqa_val_original.log" 2>&1 &
  pids+=("$!")
  gpu=$((gpu + 1))
done
wait_all "${pids[@]}" || exit 8

# Select exactly one reporting checkpoint using PathMMU validation only.
"$PYTHON" - "$EVAL_ROOT" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for step in (16, 32, 48, 64, 80, 96):
    pmmu = json.loads((root / "pathmmu_val" / f"step{step:03d}" / "metrics.json").read_text())
    pqa_dir = root / "pathvqa_val" / f"step{step:03d}" / "original"
    pqa = json.loads((pqa_dir / "metrics.json").read_text())
    pred = [json.loads(x) for x in (pqa_dir / "predictions.jsonl").read_text().splitlines()]
    rows.append({
        "step": step,
        "pathmmu_correct": pmmu["correct"],
        "pathmmu_count": pmmu["count"],
        "pathmmu_accuracy": pmmu["accuracy"],
        "pathmmu_format_correct": pmmu["format_correct"],
        "pathmmu_cap_hits": pmmu["generation_cap_hit_count"],
        "pathvqa_correct": pqa["correct"],
        "pathvqa_count": pqa["count"],
        "pathvqa_accuracy": pqa["accuracy"],
        "pathvqa_yes_rate": sum(x["forced_binary_answer"] == "yes" for x in pred) / len(pred),
        "pathvqa_mean_target_margin": sum(float(x["target_margin"]) for x in pred) / len(pred),
    })
selected = max(rows, key=lambda x: (x["pathmmu_accuracy"], -x["step"]))
payload = {
    "schema_version": 1,
    "status": "validation_complete_test_reporting_checkpoint_frozen",
    "selection_rule": "highest PathMMU validation385 accuracy; earliest step breaks exact ties",
    "test_used_for_selection": False,
    "rows": rows,
    "selected_step": selected["step"],
}
(root / "validation_and_test_checkpoint_selection.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
print(selected["step"])
PY
selected_step=$("$PYTHON" -c "import json; print(json.load(open('$EVAL_ROOT/validation_and_test_checkpoint_selection.json'))['selected_step'])")
adapter=$TRAIN_ROOT/output/checkpoint-$selected_step
selected_id=step$(printf '%03d' "$selected_step")

# Phase 3: post-hoc tests. PathMMU occupies GPU0; seven exact PathVQA shards occupy GPUs1-7.
mkdir -p "$EVAL_ROOT/pathmmu_test999" "$EVAL_ROOT/pathvqa_test_yesno3362/$selected_id/shards"
pathmmu_test_output=$EVAL_ROOT/pathmmu_test999/$selected_id
[[ ! -e "$pathmmu_test_output" ]] || exit 9
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$REPO/scripts "$PYTHON" \
  "$REPO/scripts/run_pathmmu_qwen_diagnostic.py" \
  --model "$MODEL" --adapter "$adapter" --backend qwen2_5_vl \
  --data "$PATHMMU_TEST" --output-dir "$pathmmu_test_output" --batch-size 32 \
  --split-role test999_development --max-new-tokens 1024 \
  >"$EVAL_ROOT/logs/${selected_id}_pathmmu_test999.log" 2>&1 &
pids=("$!")
for shard in 0 1 2 3 4 5 6; do
  output=$EVAL_ROOT/pathvqa_test_yesno3362/$selected_id/shards/shard_$shard
  [[ ! -e "$output" ]] || exit 10
  CUDA_VISIBLE_DEVICES=$((shard + 1)) PYTHONPATH=$REPO/scripts "$PYTHON" \
    "$REPO/scripts/run_pathvqa_forced_binary_logits.py" \
    --model "$MODEL" --adapter "$adapter" --data "$PATHVQA_TEST" \
    --output-dir "$output" --batch-size 64 --shard-index "$shard" --shard-count 7 \
    --split-role post_hoc_test_sensitivity --image-mode original \
    >"$EVAL_ROOT/logs/${selected_id}_pathvqa_test_shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
wait_all "${pids[@]}" || exit 11
"$PYTHON" "$REPO/scripts/merge_pathvqa_forced_binary_shards.py" \
  --shard-root "$EVAL_ROOT/pathvqa_test_yesno3362/$selected_id/shards" \
  --output-dir "$EVAL_ROOT/pathvqa_test_yesno3362/$selected_id/merged"

# Phase 4: representative retention/intervention diagnostics.
mkdir -p "$EVAL_ROOT/train_probe" "$EVAL_ROOT/mmmu"
pids=()
for spec in cyclic_mismatch:0 global_mean_blank:1; do
  mode=${spec%%:*}; gpu=${spec#*:}
  output=$EVAL_ROOT/pathvqa_val/$selected_id/$mode
  [[ ! -e "$output" ]] || exit 12
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO/scripts "$PYTHON" \
    "$REPO/scripts/run_pathvqa_forced_binary_logits.py" \
    --model "$MODEL" --adapter "$adapter" --data "$PATHVQA_VAL" \
    --output-dir "$output" --batch-size 64 --split-role validation_diagnostic \
    --image-mode "$mode" >"$EVAL_ROOT/logs/${selected_id}_pathvqa_val_${mode}.log" 2>&1 &
  pids+=("$!")
done
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$REPO/scripts "$PYTHON" \
  "$REPO/scripts/run_pathmmu_qwen_diagnostic.py" \
  --model "$MODEL" --adapter "$adapter" --backend qwen2_5_vl \
  --data "$TRAIN_PROBE" --output-dir "$EVAL_ROOT/train_probe/$selected_id" --batch-size 32 \
  --split-role train_memorization_probe --max-new-tokens 1024 \
  >"$EVAL_ROOT/logs/${selected_id}_train_probe.log" 2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$REPO/scripts "$PYTHON" \
  "$REPO/scripts/run_mmmu_retention_diagnostic.py" \
  --model "$MODEL" --adapter "$adapter" --panel "$MMMU" \
  --output-dir "$EVAL_ROOT/mmmu/$selected_id" --batch-size 4 --max-new-tokens 4096 \
  >"$EVAL_ROOT/logs/${selected_id}_mmmu.log" 2>&1 &
pids+=("$!")
wait_all "${pids[@]}" || exit 13

"$PYTHON" - "$EVAL_ROOT" "$selected_id" <<'PY'
import json, sys
from pathlib import Path

root, selected_id = Path(sys.argv[1]), sys.argv[2]
selection = json.loads((root / "validation_and_test_checkpoint_selection.json").read_text())
original = json.loads((root / "pathvqa_val" / selected_id / "original" / "metrics.json").read_text())
shuffle = json.loads((root / "pathvqa_val" / selected_id / "cyclic_mismatch" / "metrics.json").read_text())
blank = json.loads((root / "pathvqa_val" / selected_id / "global_mean_blank" / "metrics.json").read_text())
payload = {
    "schema_version": 1,
    "status": "formal_a_sft3000_mechanism_evaluation_complete",
    "selected_step": selection["selected_step"],
    "test_used_for_selection": False,
    "pathmmu_test": json.loads((root / "pathmmu_test999" / selected_id / "metrics.json").read_text()),
    "pathvqa_test": json.loads((root / "pathvqa_test_yesno3362" / selected_id / "merged" / "metrics.json").read_text()),
    "train_probe": json.loads((root / "train_probe" / selected_id / "metrics.json").read_text()),
    "mmmu": json.loads((root / "mmmu" / selected_id / "metrics.json").read_text()),
    "pathvqa_val_original": original,
    "pathvqa_val_shuffle": shuffle,
    "pathvqa_val_blank": blank,
    "pathvqa_val_normal_minus_shuffle": original["accuracy"] - shuffle["accuracy"],
    "pathvqa_val_normal_minus_blank": original["accuracy"] - blank["accuracy"],
}
(root / "complete_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

echo "formal A-SFT3000 mechanism evaluation complete: $EVAL_ROOT"
