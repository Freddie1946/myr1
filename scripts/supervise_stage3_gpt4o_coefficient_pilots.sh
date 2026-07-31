#!/usr/bin/env bash
set -euo pipefail
umask 077

WORKSPACE="/home/dataset-assist-0/czy/wjy"
REPO="$WORKSPACE/myr1"
INSTALL="$WORKSPACE/pathvlm_r1_v1_a100"
PYTHON="$INSTALL/envs/grpo/bin/python"
RUN_ROOT="$INSTALL/runs/stage3_process_grpo"
STATUS="$RUN_ROOT/gpt4o_coefficient_pilots_supervision_20260801.json"
LAUNCHER="$REPO/scripts/launch_stage3_gpt4o_penalty_pilot.sh"
CHECKPOINT_TOOL="$REPO/scripts/stage3_checkpoint_recovery.py"

[[ ! -e "$STATUS" ]] || { echo "Refusing to reuse supervision status: $STATUS" >&2; exit 2; }

write_status() {
  local state="$1"
  local detail="$2"
  "$PYTHON" - "$STATUS" "$state" "$detail" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path, state, detail = sys.argv[1:]
prior = {}
if Path(path).exists():
    prior = json.loads(Path(path).read_text(encoding="utf-8"))
events = prior.get("events", [])
events.append({"timestamp": datetime.now(timezone.utc).isoformat(), "state": state, "detail": detail})
value = {
    "schema_version": 1,
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "state": state,
    "detail": detail,
    "automatic_same_arm_restarts": 0,
    "events": events,
}
temporary = Path(path).with_suffix(".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

on_error() {
  local line="$1"
  local status="$2"
  set +e
  write_status "failed_stopped" "line=$line exit_status=$status; no arm was automatically restarted"
  exit "$status"
}
trap 'on_error "$LINENO" "$?"' ERR

run_dir() {
  printf '%s/gpt4o_penalty_%s_100step_seed42_20260801' "$RUN_ROOT" "$1"
}

wait_for_existing_arm() {
  local tag="$1"
  local run
  run="$(run_dir "$tag")"
  local audit="$run/output/gpt4o_penalty_${tag}_train_state_audit.json"
  write_status "monitoring_${tag}" "waiting for the already-started arm; no restart is permitted"
  while [[ ! -f "$audit" ]]; do
    if ! pgrep -f -- "$run/output" >/dev/null; then
      echo "Arm $tag stopped without a complete train-state audit" >&2
      return 1
    fi
    sleep 30
  done
}

wait_for_idle_gpus() {
  local attempts=0
  while true; do
    mapfile -t memory < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [[ "${#memory[@]}" -eq 8 ]] || return 1
    local busy=0
    for value in "${memory[@]}"; do
      value="${value//[[:space:]]/}"
      (( value <= 10 )) || busy=1
    done
    (( busy == 0 )) && return 0
    attempts=$((attempts + 1))
    (( attempts <= 40 )) || { echo "GPUs did not become idle within 10 minutes" >&2; return 1; }
    sleep 15
  done
}

validate_arm() {
  local tag="$1"
  local penalty="$2"
  local run
  run="$(run_dir "$tag")"
  "$PYTHON" "$CHECKPOINT_TOOL" validate "$run/output/checkpoint-100" --world-size 8 >/dev/null
  "$PYTHON" - "$run" "$tag" "$penalty" <<'PY'
import glob
import json
import os
import sys
from pathlib import Path

run = Path(sys.argv[1])
tag = sys.argv[2]
penalty = float(sys.argv[3])
output = run / "output"
audit_path = output / f"gpt4o_penalty_{tag}_train_state_audit.json"
audit = json.loads(audit_path.read_text(encoding="utf-8"))
if audit.get("global_step") != 100 or audit.get("resume_from_checkpoint") is not None:
    raise SystemExit("train-state audit is not a fresh complete 100-step arm")
for name in ("model.safetensors.index.json", "train_results.json"):
    if not (output / name).is_file() or (output / name).stat().st_size == 0:
        raise SystemExit(f"final output file is missing or empty: {name}")
ledger_path = run / "judge" / "budget_ledger.json"
ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
if ledger.get("budget_breached") is not False:
    raise SystemExit("budget ledger is breached")
if ledger.get("reservations"):
    raise SystemExit("budget ledger has unresolved reservations")
attempts = int(ledger.get("completed_unique_requests", -1))
spend = float(ledger.get("committed_spend_usd", 99))
if not 0 < attempts <= 800 or not 0 <= spend <= 6.0:
    raise SystemExit(f"attempt/spend cap mismatch: attempts={attempts}, spend={spend}")
cache_files = glob.glob(str(run / "judge" / "cache" / "**" / "*.json"), recursive=True)
if not cache_files:
    raise SystemExit("Judge cache is empty")
for path in cache_files:
    row = json.loads(Path(path).read_text(encoding="utf-8"))
    if row.get("served_model") != "gpt-4o-2024-08-06":
        raise SystemExit(f"served model mismatch in {path}")
    if row.get("scores", {}).get("penalty") != penalty:
        raise SystemExit(f"penalty mismatch in {path}")
fallback_path = run / "judge" / "rule_fallback_ledger.json"
fallback_total = 0
if fallback_path.exists():
    fallback = json.loads(fallback_path.read_text(encoding="utf-8"))
    fallback_total = int(fallback.get("total_used", -1))
    if fallback_total < 0 or fallback_total > 24:
        raise SystemExit("fallback total exceeds contract")
result = {
    "schema_version": 1,
    "status": "passed",
    "penalty": penalty,
    "global_step": 100,
    "http_attempts": attempts,
    "committed_spend_usd": spend,
    "cache_records": len(cache_files),
    "fallback_total": fallback_total,
    "test_accessed": False,
}
path = run / "supervision_validation.json"
path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, sort_keys=True))
PY
  write_status "validated_${tag}" "complete checkpoint, exact Judge, attempt and budget gates passed"
}

write_status "started" "sequential success-only supervision for 0.3, 0.4 and 0.5"
wait_for_existing_arm "0p3"
wait_for_idle_gpus
validate_arm "0p3" "0.3"

write_status "launching_0p4" "0.3 passed; starting independently from the fixed Stage2 parent"
"$LAUNCHER" 0.4
wait_for_idle_gpus
validate_arm "0p4" "0.4"

write_status "launching_0p5" "0.4 passed; starting independently from the fixed Stage2 parent"
"$LAUNCHER" 0.5
wait_for_idle_gpus
validate_arm "0p5" "0.5"

write_status "training_complete" "all three matched 100-step arms passed; validation385 remains separate"
trap - ERR
echo "All three GPT-4o coefficient pilots completed and passed supervision gates."
