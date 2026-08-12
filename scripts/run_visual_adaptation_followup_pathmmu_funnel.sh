#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT=${1:-/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/visual_adaptation_followup_n1500_e3_2seed_20260811}
EVAL_ROOT=${2:-/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/visual_adaptation_followup_n1500_e3_2seed_20260811}
REPO=/home/dataset-assist-0/czy/wjy/myr1
PYTHON=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/sft/bin/python
MODEL=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/models/Qwen2.5-VL-7B-Instruct-cc594898137f460bfe9f0759e9844b3ce807cfb5
VAL=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/pathmmu_image_disjoint_v2/rewritten_records/validation_0385.json

[[ -f "$TRAIN_ROOT/training_sequence_complete.json" ]] || {
  echo "training sequence is not complete" >&2
  exit 2
}
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU compute process is active" >&2
  exit 3
fi
mkdir -p "$EVAL_ROOT/pathmmu" "$EVAL_ROOT/logs"

candidates=()
for seed in 42; do
  for arm in l a; do
    for step in 8 16 24 32 40 48; do
      id=${arm}_seed${seed}_step$(printf '%03d' "$step")
      adapter=$TRAIN_ROOT/${arm}_seed${seed}/output/checkpoint-$step
      [[ -f "$adapter/adapter_model.safetensors" ]] || {
        echo "missing candidate adapter: $adapter" >&2
        exit 4
      }
      candidates+=("$id|$adapter")
    done
  done
done

run_one() {
  local spec=$1 gpu=$2 id=${1%%|*} adapter=${1#*|}
  local output=$EVAL_ROOT/pathmmu/$id
  if [[ -f "$output/metrics.json" ]]; then
    return 0
  fi
  [[ ! -e "$output" ]] || {
    echo "incomplete existing output blocks $id" >&2
    return 1
  }
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO/scripts "$PYTHON" \
    "$REPO/scripts/run_pathmmu_qwen_diagnostic.py" \
    --model "$MODEL" --adapter "$adapter" --backend qwen2_5_vl \
    --data "$VAL" --output-dir "$output" --batch-size 8 \
    --split-role validation_smoke --max-new-tokens 1024 \
    >"$EVAL_ROOT/logs/${id}_pathmmu.log" 2>&1
}

for ((start=0; start<${#candidates[@]}; start+=8)); do
  pids=()
  for ((offset=0; offset<8 && start+offset<${#candidates[@]}; offset++)); do
    run_one "${candidates[$((start+offset))]}" "$offset" &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  ((failed == 0)) || exit 5
done

"$PYTHON" - "$EVAL_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for metrics_path in sorted((root / "pathmmu").glob("*/metrics.json")):
    name = metrics_path.parent.name
    arm, seed_text, step_text = name.split("_")
    metrics = json.loads(metrics_path.read_text())
    rows.append({
        "id": name,
        "arm": arm,
        "seed": int(seed_text.removeprefix("seed")),
        "step": int(step_text.removeprefix("step")),
        "correct": metrics["correct"],
        "count": metrics["count"],
        "accuracy": metrics["accuracy"],
        "format_correct": metrics["format_correct"],
        "generation_cap_hit_count": metrics["generation_cap_hit_count"],
        "predictions_sha256": metrics["predictions_sha256"],
    })
if len(rows) != 12:
    raise SystemExit(f"expected 12 completed candidates, got {len(rows)}")
shortlist = []
for arm in ("l", "a"):
    for seed in (42,):
        group = [row for row in rows if row["arm"] == arm and row["seed"] == seed]
        best = max(row["accuracy"] for row in group)
        shortlist.extend(row["id"] for row in group if row["accuracy"] >= best - 0.01 - 1e-12)
payload = {
    "schema_version": 1,
    "status": "pathmmu_phase_completed",
    "candidate_rule": "within 1.0 percentage point of arm-and-seed best",
    "rows": rows,
    "shortlist": shortlist,
}
(root / "pathmmu_funnel_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
