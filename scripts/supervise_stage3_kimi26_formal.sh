#!/usr/bin/env bash
set -euo pipefail
umask 077

WORKSPACE_ROOT="/home/dataset-assist-0/czy/wjy"
REPO_ROOT="$WORKSPACE_ROOT/myr1"
INSTALL_ROOT="$WORKSPACE_ROOT/pathvlm_r1_v1_a100"
PYTHON="$INSTALL_ROOT/envs/grpo/bin/python"
LAUNCHER="$REPO_ROOT/scripts/launch_stage3_kimi26_formal.sh"
RECOVERY="$REPO_ROOT/scripts/stage3_checkpoint_recovery.py"
MAX_UNIQUE_REQUESTS=12361
RESERVE_USD=0.05

: "${PATHVLM_STAGE3_RUN_DIR:?Set a dedicated Stage3 run directory}"
: "${PATHVLM_STAGE3_FORMAL_BUDGET_USD:?Set the separately approved run budget}"

RUN_DIR="$(readlink -m "$PATHVLM_STAGE3_RUN_DIR")"
OUTPUT_DIR="$RUN_DIR/output"
JUDGE_LEDGER="$RUN_DIR/judge/budget_ledger.json"
SMOKE_MARKER="$RUN_DIR/kimi26_formal_contract_smoke_passed.json"
SUPERVISOR_AUDIT="$RUN_DIR/supervisor_recovery_audit.jsonl"
SAVE_STEPS="${PATHVLM_STAGE3_SAVE_STEPS:-100}"
MAX_RECOVERIES="${PATHVLM_STAGE3_MAX_RECOVERIES:-3}"
START_LAUNCH_INDEX="${PATHVLM_STAGE3_START_LAUNCH_INDEX:-0}"
BASE_PORT="${PATHVLM_STAGE3_MASTER_PORT:-29641}"

expected_parent="$INSTALL_ROOT/runs/stage3_process_grpo"
case "$RUN_DIR" in
  "$expected_parent"/*) ;;
  *)
    echo "Run directory must be a child of $expected_parent" >&2
    exit 2
    ;;
esac
if [[ "$SAVE_STEPS" != "100" ]]; then
  echo "This recovery contract requires PATHVLM_STAGE3_SAVE_STEPS=100" >&2
  exit 2
fi
if [[ ! "$MAX_RECOVERIES" =~ ^[0-9]+$ ]] || (( MAX_RECOVERIES > 3 )); then
  echo "PATHVLM_STAGE3_MAX_RECOVERIES must be an integer from 0 through 3" >&2
  exit 2
fi
if [[ ! "$START_LAUNCH_INDEX" =~ ^[0-9]+$ ]] || (( START_LAUNCH_INDEX > MAX_RECOVERIES )); then
  echo "PATHVLM_STAGE3_START_LAUNCH_INDEX must be an integer from 0 through PATHVLM_STAGE3_MAX_RECOVERIES" >&2
  exit 2
fi
if [[ ! "$BASE_PORT" =~ ^[1-9][0-9]*$ ]] || (( BASE_PORT + MAX_RECOVERIES > 65535 )); then
  echo "PATHVLM_STAGE3_MASTER_PORT cannot support the bounded recovery range" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" || ! -f "$LAUNCHER" || ! -f "$RECOVERY" ]]; then
  echo "Stage3 Python, launcher, or recovery helper is missing" >&2
  exit 2
fi
if [[ ! -f "$SMOKE_MARKER" ]]; then
  echo "Paid formal smoke must pass before the supervisor can launch: $SMOKE_MARKER" >&2
  exit 2
fi

record_event() {
  local event_json="$1"
  "$PYTHON" "$RECOVERY" event --audit "$SUPERVISOR_AUDIT" --json "$event_json"
}

archive_incomplete_output() {
  local segment="$1"
  if [[ ! -e "$OUTPUT_DIR" ]]; then
    return
  fi
  local archive_root="$RUN_DIR/failed_precheckpoint_outputs"
  local archive="$archive_root/${segment}-output"
  if [[ -e "$archive" ]]; then
    echo "Refusing to overwrite preserved failed output: $archive" >&2
    exit 2
  fi
  mkdir -p "$archive_root"
  mv -- "$OUTPUT_DIR" "$archive"
  record_event "{\"event\":\"incomplete_output_preserved\",\"segment\":\"$segment\",\"archive\":\"$archive\"}"
}

resume_from=""
for (( launch_index=START_LAUNCH_INDEX; launch_index<=MAX_RECOVERIES; launch_index++ )); do
  segment="segment$(printf '%02d' "$launch_index")"
  if [[ -d "$OUTPUT_DIR" ]]; then
    set +e
    resume_from="$($PYTHON "$RECOVERY" latest --output-dir "$OUTPUT_DIR" --world-size 8 --path-only)"
    latest_status=$?
    set -e
    if (( latest_status == 3 )); then
      resume_from=""
      archive_incomplete_output "$segment-prelaunch"
    elif (( latest_status != 0 )); then
      echo "Checkpoint discovery failed with status $latest_status" >&2
      exit "$latest_status"
    fi
  else
    resume_from=""
  fi

  export PATHVLM_STAGE3_SAVE_STEPS="$SAVE_STEPS"
  export PATHVLM_STAGE3_SEGMENT_ID="$segment"
  export PATHVLM_STAGE3_MASTER_PORT="$((BASE_PORT + launch_index))"
  if [[ -n "$resume_from" ]]; then
    export PATHVLM_STAGE3_RESUME_FROM_CHECKPOINT="$resume_from"
  else
    unset PATHVLM_STAGE3_RESUME_FROM_CHECKPOINT || true
  fi

  record_event "{\"event\":\"segment_start\",\"segment\":\"$segment\",\"launch_index\":$launch_index,\"resume_from\":\"$resume_from\",\"save_steps\":$SAVE_STEPS}"
  set +e
  bash "$LAUNCHER"
  launch_status=$?
  set -e
  if (( launch_status == 0 )); then
    record_event "{\"event\":\"training_completed\",\"segment\":\"$segment\",\"launch_index\":$launch_index}"
    echo "Stage3 completed under bounded recovery supervision"
    exit 0
  fi

  train_log="$RUN_DIR/train_${segment}.log"
  set +e
  failure_classification="$($PYTHON "$RECOVERY" classify --log "$train_log")"
  classification_status=$?
  set -e
  if (( classification_status != 0 )); then
    failure_classification="{\"action\":\"stop\",\"category\":\"failure_classifier_error\",\"classifier_exit_status\":$classification_status,\"log\":\"$train_log\"}"
  fi
  recovery_action="$($PYTHON - "$failure_classification" <<'PY'
import json
import sys
print(json.loads(sys.argv[1])["action"])
PY
)"

  settlement="$($PYTHON "$RECOVERY" settle \
    --ledger "$JUDGE_LEDGER" \
    --limit-usd "$PATHVLM_STAGE3_FORMAL_BUDGET_USD" \
    --reserve-usd "$RESERVE_USD" \
    --max-unique-requests "$MAX_UNIQUE_REQUESTS" \
    --reason "$segment exited with status $launch_status")"

  if [[ "$recovery_action" != "recover" ]]; then
    event_json="$($PYTHON - "$segment" "$launch_index" "$launch_status" \
      "$resume_from" "$settlement" "$failure_classification" <<'PY'
import json
import sys
print(json.dumps({
    "event": "segment_failed_terminal",
    "segment": sys.argv[1],
    "launch_index": int(sys.argv[2]),
    "exit_status": int(sys.argv[3]),
    "resumed_from": sys.argv[4],
    "settlement": json.loads(sys.argv[5]),
    "failure_classification": json.loads(sys.argv[6]),
}, separators=(",", ":")))
PY
)"
    record_event "$event_json"
    echo "Stage3 stopped without automatic retry: $failure_classification" >&2
    exit "$launch_status"
  fi

  set +e
  next_resume="$($PYTHON "$RECOVERY" latest --output-dir "$OUTPUT_DIR" --world-size 8 --path-only 2>/dev/null)"
  latest_status=$?
  set -e
  if (( latest_status == 3 )); then
    next_resume=""
  elif (( latest_status != 0 )); then
    echo "Checkpoint discovery after failure returned status $latest_status" >&2
    exit "$latest_status"
  fi
  event_json="$($PYTHON - "$segment" "$launch_index" "$launch_status" "$resume_from" "$next_resume" "$settlement" "$failure_classification" <<'PY'
import json
import sys
print(json.dumps({
    "event": "segment_failed",
    "segment": sys.argv[1],
    "launch_index": int(sys.argv[2]),
    "exit_status": int(sys.argv[3]),
    "resumed_from": sys.argv[4],
    "next_resume": sys.argv[5],
    "settlement": json.loads(sys.argv[6]),
    "failure_classification": json.loads(sys.argv[7]),
}, separators=(",", ":")))
PY
)"
  record_event "$event_json"

  if (( launch_index == MAX_RECOVERIES )); then
    echo "Stage3 stopped after $((MAX_RECOVERIES + 1)) bounded launches; no further automatic retry" >&2
    exit "$launch_status"
  fi
  if [[ -z "$next_resume" ]]; then
    archive_incomplete_output "$segment-postfailure"
  fi
  echo "Segment $segment failed; conservatively settled and will recover from ${next_resume:-the frozen parent}"
  sleep 15
done
