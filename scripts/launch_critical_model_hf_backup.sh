#!/usr/bin/env bash
set -euo pipefail

INSTALL_ROOT=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100
REPO_ROOT=/home/dataset-assist-0/czy/wjy/myr1
FORMAL_ROOT="$INSTALL_ROOT/runs/data_ratio_rule_rl_ablation_v1/formal_sequence"
GROK_OUTPUT="$INSTALL_ROOT/runs/stage3_process_grpo/grok43_full3epoch_seed42_20260805_attempt02/output"
STATE_ROOT="$INSTALL_ROOT/reports/critical_model_hf_backup_20260810"
LOG_PATH="$STATE_ROOT/backup.log"
PID_PATH="$STATE_ROOT/worker.pid"

mkdir -p "$STATE_ROOT"
cd "$REPO_ROOT"
echo "$$" >"$PID_PATH"
exec nice -n 10 "$INSTALL_ROOT/envs/grpo/bin/python" \
  scripts/backup_critical_models_to_hf.py \
  --formal-root "$FORMAL_ROOT" \
  --grok-output "$GROK_OUTPUT" \
  --state-root "$STATE_ROOT" \
  --poll-seconds 300 \
  >>"$LOG_PATH" 2>&1
