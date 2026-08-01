#!/usr/bin/env bash
set -euo pipefail
umask 077

WORKSPACE="/home/dataset-assist-0/czy/wjy"
REPO="$WORKSPACE/myr1"
INSTALL="$WORKSPACE/pathvlm_r1_v1_a100"
PYTHON="$INSTALL/envs/grpo/bin/python"
RUN_ROOT="$INSTALL/runs/stage3_process_grpo"
RECOVERY="$RUN_ROOT/gpt4o_penalty_0p5_100step_seed42_20260801_recovery_fresh01"
RECOVERY_AUDIT="$RECOVERY/output/gpt4o_penalty_0p5_recovery_fresh01_train_state_audit.json"
STATUS="$RUN_ROOT/gpt4o_0p5_recovery_and_validation_supervision_20260801.json"
CHECKPOINT_TOOL="$REPO/scripts/stage3_checkpoint_recovery.py"
VALIDATION="$REPO/scripts/run_stage3_validation385_parallel.sh"

[[ ! -e "$STATUS" ]] || { echo "Refusing to reuse recovery supervision status" >&2; exit 2; }

write_status() {
  "$PYTHON" - "$STATUS" "$1" "$2" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path, state, detail = sys.argv[1:]
prior = json.loads(Path(path).read_text()) if Path(path).exists() else {}
events = prior.get("events", [])
events.append({"timestamp": datetime.now(timezone.utc).isoformat(), "state": state, "detail": detail})
value = {
    "schema_version": 1,
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "state": state,
    "detail": detail,
    "automatic_training_restarts": 0,
    "events": events,
}
temporary = Path(path).with_suffix(".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
}

on_error() {
  local line="$1"
  local status="$2"
  set +e
  write_status "failed_stopped" "line=$line exit_status=$status; recovery was not restarted"
  exit "$status"
}
trap 'on_error "$LINENO" "$?"' ERR

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

validate_recovery() {
  "$PYTHON" "$CHECKPOINT_TOOL" validate "$RECOVERY/output/checkpoint-100" --world-size 8 >/dev/null
  "$PYTHON" - "$RECOVERY" <<'PY'
import glob
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
audit = json.loads(
    (run / "output" / "gpt4o_penalty_0p5_recovery_fresh01_train_state_audit.json").read_text()
)
if audit.get("global_step") != 100 or audit.get("resume_from_checkpoint") is not None:
    raise SystemExit("recovery train-state audit is not a fresh complete 100-step run")
ledger = json.loads((run / "judge" / "budget_ledger.json").read_text())
attempts = int(ledger.get("completed_unique_requests", -1))
spend = float(ledger.get("committed_spend_usd", 99))
if ledger.get("budget_breached") is not False or ledger.get("reservations"):
    raise SystemExit("recovery budget ledger is breached or unresolved")
if not 800 <= attempts <= 824 or not 0 <= spend <= 5.5:
    raise SystemExit(f"recovery attempt/spend mismatch: {attempts}, {spend}")
caches = [
    json.loads(Path(path).read_text())
    for path in glob.glob(str(run / "judge" / "cache" / "**" / "*.json"), recursive=True)
]
if not caches:
    raise SystemExit("recovery Judge cache is empty")
for row in caches:
    if row.get("served_model") != "gpt-4o-2024-08-06" or row.get("scores", {}).get("penalty") != 0.5:
        raise SystemExit("recovery Judge identity or penalty mismatch")
process_rewards = 0
for path in glob.glob(str(run / "reward_audit" / "*.jsonl")):
    for line in Path(path).read_text().splitlines():
        process_rewards += json.loads(line).get("reward_type") == "process"
if process_rewards != 800:
    raise SystemExit(f"expected 800 process rewards, found {process_rewards}")
fallback_path = run / "judge" / "rule_fallback_ledger.json"
fallbacks = 0
if fallback_path.exists():
    fallbacks = int(json.loads(fallback_path.read_text()).get("total_used", -1))
    if not 0 <= fallbacks <= 24:
        raise SystemExit("recovery fallback count exceeds contract")
result = {
    "schema_version": 1,
    "status": "passed",
    "penalty": 0.5,
    "global_step": 100,
    "http_attempts": attempts,
    "committed_spend_usd": spend,
    "cache_records": len(caches),
    "process_rewards": process_rewards,
    "fallback_total": fallbacks,
    "cache_reuse": False,
    "test_accessed": False,
}
(run / "supervision_validation.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(result, sort_keys=True))
PY
}

run_validation() {
  local label="$1"
  local model="$2"
  local destination="$3"
  write_status "validating_${label}" "PathMMU validation385 greedy inference; test remains excluded"
  PATHVLM_STAGE3_VALIDATION_MODEL="$model" \
  PATHVLM_STAGE3_VALIDATION_RUN_ROOT="$destination" \
    "$VALIDATION"
}

write_status "monitoring_recovery" "waiting for fresh penalty-0.5 completion; no automatic rerun"
while [[ ! -f "$RECOVERY_AUDIT" ]]; do
  if ! pgrep -f -- "$RECOVERY/output" >/dev/null; then
    echo "Recovery stopped without a complete train-state audit" >&2
    exit 1
  fi
  sleep 30
done
wait_for_idle_gpus
validate_recovery
write_status "recovery_validated" "complete checkpoint, Judge, logical reward, attempt and budget gates passed"

RUN_03="$RUN_ROOT/gpt4o_penalty_0p3_100step_seed42_20260801"
RUN_04="$RUN_ROOT/gpt4o_penalty_0p4_100step_seed42_20260801"
run_validation "0p3" "$RUN_03/output" "$RUN_03/validation_0385_greedy_20260801"
run_validation "0p4" "$RUN_04/output" "$RUN_04/validation_0385_greedy_20260801"
run_validation "0p5_recovery" "$RECOVERY/output" "$RECOVERY/validation_0385_greedy_20260801"

write_status "complete" "fresh recovery and all three matched validation385 runs completed"
trap - ERR
echo "Recovery and three validation385 runs completed."
