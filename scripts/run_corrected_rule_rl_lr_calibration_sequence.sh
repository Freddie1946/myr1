#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/corrected_rule_rl_lr_calibration_20260811
MODEL=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/formal_selected_rule_rl1000_20260811/parents/sft_step080_merged
LAUNCHER=/home/dataset-assist-0/czy/wjy/myr1/scripts/launch_formal_selected_rule_rl1000.sh

for specification in lr1e6:1.0e-6:33731 lr3e6:3.0e-6:33732; do
  IFS=: read -r label learning_rate master_port <<<"$specification"
  run_dir=$ROOT/$label
  [[ ! -e "$run_dir/output" ]] || {
    echo "calibration output already exists: $run_dir/output" >&2
    exit 3
  }
  MODEL_PATH=$MODEL \
  RUN_DIR=$run_dir \
  MODE=formal \
  RUN_ROLE=optimizer_calibration \
  PER_DEVICE_BATCH=10 \
  GRADIENT_CHECKPOINTING=false \
  MAX_COMPLETION_LENGTH=384 \
  LEARNING_RATE=$learning_rate \
  MAX_STEPS_OVERRIDE=150 \
  STOP_AFTER_SAVED_STEP=10 \
  SAVE_STRATEGY_OVERRIDE=steps \
  SAVE_STEPS_OVERRIDE=10 \
  SAVE_LIMIT_OVERRIDE=1 \
  PATHVLM_SKIP_FINAL_MODEL_SAVE=true \
  MASTER_PORT=$master_port \
  bash "$LAUNCHER"
done
