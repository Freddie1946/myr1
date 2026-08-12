#!/usr/bin/env bash
set -euo pipefail

REPO=/home/dataset-assist-0/czy/wjy/myr1
PYTHON=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/grpo/bin/python
ROOT=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/full_language_rule_rl_capacity_20260811
PARENT=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/formal_selected_rule_rl1000_20260811/parents/sft_step080_merged
PROBE=/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/rule_rl_mechanism_train_probe256_20260811/records_n0256.json
SMOKE=$ROOT/smoke_z3_gpu_gc1_step002
FORMAL=$ROOT/formal_g2_lr1_step100
STATE=$ROOT/sequence_state.json

mkdir -p "$ROOT"
[[ ! -e "$STATE" ]] || { echo "sequence state already exists: $STATE" >&2; exit 3; }

state() {
  local status=$1 stage=$2
  "$PYTHON" - "$STATE" "$status" "$stage" <<'PY'
import json,os,sys
from datetime import datetime,timezone
from pathlib import Path
p=Path(sys.argv[1])
payload={
  "schema_version":1,
  "status":sys.argv[2],
  "stage":sys.argv[3],
  "updated_at":datetime.now(timezone.utc).isoformat(),
  "pid":os.getppid(),
  "automatic_retry":False,
  "formal_restarts_from_original_parent":True,
}
p.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
}

trap 'rc=$?; state failed "${CURRENT_STAGE:-unknown}"; exit "$rc"' ERR

CURRENT_STAGE=smoke_train
state running "$CURRENT_STAGE"
MODEL_PATH=$PARENT RUN_DIR=$SMOKE MODE=smoke \
RUN_ROLE=full_language_rule_rl_capacity_smoke \
MAX_STEPS=500 EXPLICIT_SAVE_STEPS=2 STOP_AFTER_SAVED_STEP=2 \
LEARNING_RATE=1.0e-6 MASTER_PORT=34201 SAVE_LIMIT=1 \
bash "$REPO/scripts/launch_full_language_rule_rl.sh"

CURRENT_STAGE=smoke_checkpoint_reload
state running "$CURRENT_STAGE"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$REPO/scripts/run_pathmmu_qwen_diagnostic.py" \
  --model "$SMOKE/output/checkpoint-2" --backend qwen2_5_vl \
  --data "$PROBE" --output-dir "$SMOKE/checkpoint_reload_probe8" \
  --split-role rl_train_probe --max-new-tokens 1024 --batch-size 1 --limit 8

CURRENT_STAGE=smoke_verification
state running "$CURRENT_STAGE"
"$PYTHON" "$REPO/scripts/verify_full_language_rule_rl_smoke.py" "$SMOKE" \
  --expected-step 2 --load-metrics "$SMOKE/checkpoint_reload_probe8/metrics.json"

CURRENT_STAGE=formal_train_to_step100
state running "$CURRENT_STAGE"
MODEL_PATH=$PARENT RUN_DIR=$FORMAL MODE=formal \
RUN_ROLE=full_language_rule_rl_capacity_formal_step100 \
MAX_STEPS=500 EXPLICIT_SAVE_STEPS=100 STOP_AFTER_SAVED_STEP=100 \
LEARNING_RATE=1.0e-6 MASTER_PORT=34202 SAVE_LIMIT=1 \
bash "$REPO/scripts/launch_full_language_rule_rl.sh"

CURRENT_STAGE=formal_step100_ready_for_evaluation
state completed "$CURRENT_STAGE"
