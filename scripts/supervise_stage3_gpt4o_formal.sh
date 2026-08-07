#!/usr/bin/env bash
set -euo pipefail
umask 077

WORKSPACE_ROOT="/home/dataset-assist-0/czy/wjy"
REPO_ROOT="$WORKSPACE_ROOT/myr1"
INSTALL_ROOT="$WORKSPACE_ROOT/pathvlm_r1_v1_a100"
PYTHON="$INSTALL_ROOT/envs/grpo/bin/python"
LAUNCHER="${PATHVLM_STAGE3_LAUNCHER:-$REPO_ROOT/scripts/launch_stage3_gpt4o_formal.sh}"
RECOVERY="$REPO_ROOT/scripts/stage3_checkpoint_recovery.py"
MAX_HTTP_ATTEMPTS=12360
RESERVE_USD="${PATHVLM_AIGCBEST_RESERVE_USD:-0.02}"
ACCOUNTING_CAPACITY_USD="${PATHVLM_AIGCBEST_LIMIT_USD:-247.20}"

: "${PATHVLM_STAGE3_RUN_DIR:?Set a dedicated GPT-4o formal run directory}"
RUN_DIR="$(readlink -m "$PATHVLM_STAGE3_RUN_DIR")"
OUTPUT_DIR="$RUN_DIR/output"
JUDGE_LEDGER="$RUN_DIR/judge/budget_ledger.json"
SUPERVISOR_AUDIT="$RUN_DIR/supervisor_recovery_audit.jsonl"
MAX_RECOVERIES="${PATHVLM_STAGE3_MAX_RECOVERIES:-3}"
START_INDEX="${PATHVLM_STAGE3_START_INDEX:-0}"
BASE_PORT="${PATHVLM_STAGE3_MASTER_PORT:-29740}"
[[ "$MAX_RECOVERIES" =~ ^[0-9]+$ ]] && (( MAX_RECOVERIES <= 6 )) || {
  echo "Recoveries must be an integer from 0 through 6" >&2
  exit 2
}
[[ "$START_INDEX" =~ ^[0-9]+$ ]] && (( START_INDEX <= 6 )) || {
  echo "Start index must be an integer from 0 through 6" >&2
  exit 2
}
(( START_INDEX <= MAX_RECOVERIES )) || { echo "Start index exceeds recovery limit" >&2; exit 2; }
[[ "$BASE_PORT" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ -x "$PYTHON" && -f "$LAUNCHER" && -f "$RECOVERY" ]] || exit 2

record_event() {
  "$PYTHON" "$RECOVERY" event --audit "$SUPERVISOR_AUDIT" --json "$1"
}
archive_incomplete_output() {
  local segment="$1"
  [[ -e "$OUTPUT_DIR" ]] || return
  local archive="$RUN_DIR/failed_precheckpoint_outputs/${segment}-output"
  [[ ! -e "$archive" ]] || { echo "Refusing to overwrite $archive" >&2; exit 2; }
  mkdir -p "$(dirname "$archive")"
  mv -- "$OUTPUT_DIR" "$archive"
  record_event "{\"event\":\"incomplete_output_preserved\",\"segment\":\"$segment\",\"archive\":\"$archive\"}"
}

resume_from=""
for (( launch_index=START_INDEX; launch_index<=MAX_RECOVERIES; launch_index++ )); do
  segment="segment$(printf '%02d' "$launch_index")"
  if [[ -d "$OUTPUT_DIR" ]]; then
    set +e
    resume_from="$($PYTHON "$RECOVERY" latest --output-dir "$OUTPUT_DIR" --world-size 8 --path-only)"
    latest_status=$?
    set -e
    if (( latest_status == 3 )); then resume_from=""; archive_incomplete_output "$segment-prelaunch"
    elif (( latest_status != 0 )); then exit "$latest_status"; fi
  else
    resume_from=""
  fi
  export PATHVLM_STAGE3_SEGMENT_ID="$segment"
  export PATHVLM_STAGE3_MASTER_PORT="$((BASE_PORT + launch_index))"
  export PATHVLM_STAGE3_SAVE_STEPS=100
  if [[ -n "$resume_from" ]]; then export PATHVLM_STAGE3_RESUME_FROM_CHECKPOINT="$resume_from"; else unset PATHVLM_STAGE3_RESUME_FROM_CHECKPOINT || true; fi
  record_event "{\"event\":\"segment_start\",\"segment\":\"$segment\",\"launch_index\":$launch_index,\"resume_from\":\"$resume_from\"}"
  launch_capture="$RUN_DIR/launch_${segment}.log"
  set +e
  bash "$LAUNCHER" 2>&1 | tee "$launch_capture"
  launch_status=${PIPESTATUS[0]}
  set -e
  if (( launch_status == 0 )); then
    record_event "{\"event\":\"training_completed\",\"segment\":\"$segment\",\"launch_index\":$launch_index}"
    exit 0
  fi
  train_log="$RUN_DIR/train_${segment}.log"
  classification_log="$train_log"
  # The launcher performs read-only preflight before creating train_segment*.log.
  # Preserve and classify that output instead of mislabeling a preflight failure
  # as an untraceable missing-training-log terminal failure.
  [[ -s "$classification_log" ]] || classification_log="$launch_capture"
  set +e
  failure_classification="$($PYTHON "$RECOVERY" classify --log "$classification_log")"
  classification_status=$?
  set -e
  if (( classification_status != 0 )); then
    event="$($PYTHON - "$segment" "$launch_status" "$classification_status" <<'PY'
import json,sys
print(json.dumps({"event":"failure_classifier_error","segment":sys.argv[1],"exit_status":int(sys.argv[2]),"classifier_status":int(sys.argv[3])},separators=(",",":")))
PY
)"
    record_event "$event"
    exit "$launch_status"
  fi
  recovery_action="$($PYTHON - "$failure_classification" <<'PY'
import json,sys
print(json.loads(sys.argv[1])["action"])
PY
)"
  settlement="$($PYTHON "$RECOVERY" settle --ledger "$JUDGE_LEDGER" \
    --limit-usd "$ACCOUNTING_CAPACITY_USD" --reserve-usd "$RESERVE_USD" \
    --max-unique-requests "$MAX_HTTP_ATTEMPTS" --reason "$segment exited with status $launch_status")"
  if [[ "$recovery_action" != "recover" ]]; then
    event="$($PYTHON - "$segment" "$launch_status" "$settlement" "$failure_classification" <<'PY'
import json,sys
print(json.dumps({"event":"segment_failed_terminal","segment":sys.argv[1],"exit_status":int(sys.argv[2]),"settlement":json.loads(sys.argv[3]),"failure_classification":json.loads(sys.argv[4])},separators=(",",":")))
PY
)"
    record_event "$event"
    exit "$launch_status"
  fi
  set +e
  next_resume="$($PYTHON "$RECOVERY" latest --output-dir "$OUTPUT_DIR" --world-size 8 --path-only 2>/dev/null)"
  latest_status=$?
  set -e
  [[ "$latest_status" == 0 || "$latest_status" == 3 ]] || exit "$latest_status"
  [[ "$latest_status" == 0 ]] || next_resume=""
  event="$($PYTHON - "$segment" "$launch_index" "$launch_status" "$resume_from" "$next_resume" "$settlement" "$failure_classification" <<'PY'
import json,sys
print(json.dumps({"event":"segment_failed","segment":sys.argv[1],"launch_index":int(sys.argv[2]),"exit_status":int(sys.argv[3]),"resumed_from":sys.argv[4],"next_resume":sys.argv[5],"settlement":json.loads(sys.argv[6]),"failure_classification":json.loads(sys.argv[7])},separators=(",",":")))
PY
)"
  record_event "$event"
  (( launch_index < MAX_RECOVERIES )) || exit "$launch_status"
  [[ -n "$next_resume" ]] || archive_incomplete_output "$segment-postfailure"
  sleep 15
done
