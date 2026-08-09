#!/usr/bin/env bash
set -euo pipefail
umask 077

WORK_ROOT="${WJY_WORK_ROOT:-/home/dataset-assist-0/czy/wjy}"
REPO_ROOT="$WORK_ROOT/myr1"
INSTALL_ROOT="$WORK_ROOT/pathvlm_r1_v1_a100"
FORMAL_ROOT="$INSTALL_ROOT/runs/data_ratio_rule_rl_ablation_v1/formal_sequence"
HANDOFF_PID_FILE="$FORMAL_ROOT/optimized_handoff.pid"
HANDOFF_LAUNCH_LOG="$FORMAL_ROOT/optimized_handoff_launcher.log"

if [[ -s "$HANDOFF_PID_FILE" ]]; then
  existing="$(cat "$HANDOFF_PID_FILE")"
  if kill -0 "$existing" 2>/dev/null; then
    echo "optimized handoff already active pid=$existing" >&2
    exit 2
  fi
fi

nohup setsid bash "$REPO_ROOT/scripts/handoff_data_ratio_sequence_to_optimized_tail.sh" \
  >> "$HANDOFF_LAUNCH_LOG" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" > "$HANDOFF_PID_FILE"
sleep 2
kill -0 "$pid"
echo "optimized handoff started pid=$pid log=$HANDOFF_LAUNCH_LOG"
