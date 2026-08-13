#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "usage: $0 <formal-run-dir>" >&2; exit 2; }
RUN_DIR=$1
REPO=/home/dataset-assist-0/czy/wjy/myr1
LOG="$RUN_DIR/queue.log"
POLL_SECONDS=${POLL_SECONDS:-60}
REQUIRED_IDLE_POLLS=${REQUIRED_IDLE_POLLS:-2}
MAX_USED_MIB=${MAX_USED_MIB:-1024}
idle_count=0

exec >>"$LOG" 2>&1
echo "$(date -Is) queue_started run_dir=$RUN_DIR"
while true; do
  mapfile -t used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  compute=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' || true)
  memory_ok=1
  [[ ${#used[@]} -eq 8 ]] || memory_ok=0
  for value in "${used[@]}"; do
    value=${value// /}
    [[ "$value" =~ ^[0-9]+$ && "$value" -le "$MAX_USED_MIB" ]] || memory_ok=0
  done
  if [[ -z "$compute" && "$memory_ok" -eq 1 ]]; then
    idle_count=$((idle_count + 1))
    echo "$(date -Is) idle_poll=$idle_count/$REQUIRED_IDLE_POLLS memory=${used[*]}"
  else
    idle_count=0
    echo "$(date -Is) waiting compute_pids=${compute//$'\n'/,} memory=${used[*]}"
  fi
  if [[ "$idle_count" -ge "$REQUIRED_IDLE_POLLS" ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

echo "$(date -Is) launching_formal"
exec bash "$REPO/scripts/launch_lora_sft4000_control.sh" "$RUN_DIR" 0,1,2,3,4,5,6,7 8 32940
