#!/usr/bin/env bash
set -euo pipefail
umask 077

WORK_ROOT="${WJY_WORK_ROOT:-/home/dataset-assist-0/czy/wjy}"
REPO_ROOT="$WORK_ROOT/myr1"
INSTALL_ROOT="$WORK_ROOT/pathvlm_r1_v1_a100"
FORMAL_ROOT="$INSTALL_ROOT/runs/data_ratio_rule_rl_ablation_v1/formal_sequence"
PROTOCOL="$REPO_ROOT/protocol/data_ratio_rule_rl_ablation_v1_20260809.json"
REPORT="$INSTALL_ROOT/runs/data_ratio_rule_rl_ablation_v1/throughput_smoke/performance_report.json"
HANDOFF_LOG="$FORMAL_ROOT/optimized_handoff.log"
OLD_PID="$(cat "$FORMAL_ROOT/supervisor.pid")"
OLD_KILLED=false

restore_old_supervisor() {
  if [[ "$OLD_KILLED" != true ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    kill -CONT "$OLD_PID" 2>/dev/null || true
  fi
}
trap restore_old_supervisor EXIT

python3 - "$FORMAL_ROOT/state.json" <<'PY'
import json, sys
state=json.load(open(sys.argv[1], encoding="utf-8"))
assert state["status"] == "running", state["status"]
assert state["current_task"] == "sft0250_rl0750_rule_rl", state["current_task"]
assert state["tasks"][state["current_task"]]["status"] == "running"
assert state["test_accessed"] is False
PY

CHILD_PID="$(pgrep -P "$OLD_PID" -f 'torch.distributed.run.*grpo_pathmmu.py' | head -n 1)"
[[ -n "$CHILD_PID" ]] || { echo "active rule-RL child is missing" >&2; exit 2; }
kill -STOP "$OLD_PID"
sleep 2
OLD_STAT="$(ps -o stat= -p "$OLD_PID" | tr -d ' ')"
[[ "$OLD_STAT" == T* ]] || { echo "old supervisor did not stop: $OLD_STAT" >&2; exit 2; }
echo "$(date -Is) paused old supervisor pid=$OLD_PID; current child pid=$CHILD_PID continues" >> "$HANDOFF_LOG"

while true; do
  CHILD_STAT="$(ps -o stat= -p "$CHILD_PID" 2>/dev/null | tr -d ' ' || true)"
  if [[ -z "$CHILD_STAT" || "$CHILD_STAT" == Z* ]]; then
    break
  fi
  sleep 60
done
sleep 10

"$INSTALL_ROOT/envs/grpo/bin/python" - "$REPO_ROOT" "$FORMAL_ROOT" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "scripts"))
from run_data_ratio_ablation_sequence import Task, gate_rule_rl
root=Path(sys.argv[2])
task_dir=root / "tasks/sft0250_rl0750_rule_rl"
task=Task(
    "sft0250_rl0750_rule_rl", "rule_rl",
    dataset_yaml="grpo/pathvlm_ratio_sft0250_rl0750_rl.yaml",
    parent_kind="task", parent_task="sft0250_rl0750_sft",
    sample_count=750, max_steps=1125,
)
result=gate_rule_rl(task, task_dir, task_dir / "output", task_dir / "online_reward_events_attempt01")
assert result["model"]["weight_file_count"] > 0
print(json.dumps(result, indent=2))
PY

kill -KILL "$OLD_PID"
OLD_KILLED=true
for _ in $(seq 1 30); do
  kill -0 "$OLD_PID" 2>/dev/null || break
  sleep 1
done
echo "$(date -Is) current task verified; old supervisor retired" >> "$HANDOFF_LOG"

if ! "$INSTALL_ROOT/envs/grpo/bin/python" "$REPO_ROOT/scripts/run_data_ratio_rule_rl_throughput_smoke.py" \
    --repo-root "$REPO_ROOT" \
    --install-root "$INSTALL_ROOT" \
    --report "$REPORT" \
    --max-steps 20 >> "$HANDOFF_LOG" 2>&1; then
  echo "$(date -Is) optimization smoke infrastructure failed; freezing baseline fallback" >> "$HANDOFF_LOG"
  "$INSTALL_ROOT/envs/grpo/bin/python" "$REPO_ROOT/scripts/run_data_ratio_rule_rl_throughput_smoke.py" \
    --repo-root "$REPO_ROOT" \
    --install-root "$INSTALL_ROOT" \
    --report "$REPORT" \
    --max-steps 20 \
    --baseline-only >> "$HANDOFF_LOG" 2>&1
fi

printf '%s\n' "$$" > "$FORMAL_ROOT/supervisor.pid"
echo "$(date -Is) throughput smoke passed; starting optimized tail pid=$$" >> "$HANDOFF_LOG"
exec "$INSTALL_ROOT/envs/grpo/bin/python" \
  "$REPO_ROOT/scripts/run_data_ratio_ablation_optimized_tail.py" \
  --mode formal \
  --repo-root "$REPO_ROOT" \
  --install-root "$INSTALL_ROOT" \
  --protocol "$PROTOCOL" \
  --performance-report "$REPORT" >> "$HANDOFF_LOG" 2>&1
