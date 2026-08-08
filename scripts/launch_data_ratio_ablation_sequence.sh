#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WJY_WORK_ROOT:-/home/dataset-assist-0/czy/wjy}"
REPO_ROOT="$WORK_ROOT/myr1"
INSTALL_ROOT="$WORK_ROOT/pathvlm_r1_v1_a100"
PYTHON="$INSTALL_ROOT/envs/grpo/bin/python"
PROTOCOL="$REPO_ROOT/protocol/data_ratio_rule_rl_ablation_v1_20260809.json"

MODE=""
DETACH=false
PREFLIGHT_ONLY=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:?missing value for --mode}"; shift 2 ;;
    --detach) DETACH=true; shift ;;
    --preflight-only) PREFLIGHT_ONLY=true; shift ;;
    *) printf 'unsupported argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

case "$MODE" in smoke|formal) ;; *) printf '%s\n' '--mode must be smoke or formal' >&2; exit 2 ;; esac
[[ -x "$PYTHON" ]] || { printf 'missing GRPO Python: %s\n' "$PYTHON" >&2; exit 2; }
[[ -f "$PROTOCOL" ]] || { printf 'missing protocol: %s\n' "$PROTOCOL" >&2; exit 2; }

RUN_ROOT="$INSTALL_ROOT/runs/data_ratio_rule_rl_ablation_v1/${MODE}_sequence"
mkdir -p "$RUN_ROOT"
COMMAND=(
  "$PYTHON" "$REPO_ROOT/scripts/run_data_ratio_ablation_sequence.py"
  --mode "$MODE"
  --repo-root "$REPO_ROOT"
  --install-root "$INSTALL_ROOT"
  --protocol "$PROTOCOL"
)
if [[ "$PREFLIGHT_ONLY" == true ]]; then
  COMMAND+=(--preflight-only)
fi

if [[ "$DETACH" == false ]]; then
  exec "${COMMAND[@]}"
fi
if [[ "$PREFLIGHT_ONLY" == true ]]; then
  printf '%s\n' '--detach and --preflight-only are mutually exclusive' >&2
  exit 2
fi

PID_FILE="$RUN_ROOT/supervisor.pid"
if [[ -s "$PID_FILE" ]]; then
  EXISTING_PID="$(tr -cd '0-9' < "$PID_FILE")"
  if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    printf 'sequence supervisor is already active: pid=%s\n' "$EXISTING_PID" >&2
    exit 3
  fi
fi

SUPERVISOR_LOG="$RUN_ROOT/supervisor.log"
nohup setsid "${COMMAND[@]}" >>"$SUPERVISOR_LOG" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  printf 'detached supervisor exited during launch; inspect %s\n' "$SUPERVISOR_LOG" >&2
  exit 4
fi
printf 'started mode=%s pid=%s log=%s state=%s\n' \
  "$MODE" "$PID" "$SUPERVISOR_LOG" "$RUN_ROOT/state.json"
