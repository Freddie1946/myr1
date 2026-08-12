#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/dataset-assist-0/czy/wjy/myr1
PYTHON=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/grpo/bin/python
N4_RUN=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/full_language_rule_rl_capacity_20260811/formal_g2_lr1_step100
N4_CHECKPOINT=$N4_RUN/output/checkpoint-500
N4_SNAPSHOT=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/full_language_rule_rl_capacity_20260811/model_snapshots/checkpoint-500
N4_EVAL=/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/full_language_rule_rl_capacity_20260811/step500_gate
N8_RUN=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/full_language_rule_rl_n8_capacity_20260812
SPLIT_AUDIT=/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/pathmmu_split_passk_audit_20260812
STATE=$N4_RUN/post_step500_n4_eval_n8_split_audit_state.json
RECOVERY=$N4_RUN/step500_recovered_completion_verification.json

[[ ! -e "$STATE" ]] || { echo "post-step500 state already exists: $STATE" >&2; exit 3; }
[[ ! -e "$N4_EVAL" && ! -e "$N8_RUN" && ! -e "$SPLIT_AUDIT" ]] || exit 4
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU compute process is active before recovery sequence" >&2
  exit 5
fi

state() {
  "$PYTHON" - "$STATE" "$1" "$2" <<'PY'
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "pid": os.getppid(),
    "order": ["n4_step500_evaluation", "n8_from_same_sft_step500", "three_model_split_audit"],
    "n4_training_restarted": False,
}, indent=2, sort_keys=True) + "\n")
PY
}

CURRENT_STAGE=verify_n4_saved_completion
trap 'rc=$?; state failed "${CURRENT_STAGE:-unknown}"; exit "$rc"' ERR
state running "$CURRENT_STAGE"

"$PYTHON" - "$N4_RUN" "$N4_CHECKPOINT" "$N4_SNAPSHOT" "$RECOVERY" <<'PY'
import collections, json, math, sys
from datetime import datetime, timezone
from pathlib import Path

run, checkpoint, snapshot, output = map(Path, sys.argv[1:])
trainer = json.loads((checkpoint / "trainer_state.json").read_text())
history = [row for row in trainer["log_history"] if "grad_norm" in row]
if trainer.get("global_step") != 500 or trainer.get("max_steps") != 500:
    raise RuntimeError("saved trainer state did not reach the declared 500/500 steps")
if len(history) != 500:
    raise RuntimeError(f"expected 500 optimizer rows, found {len(history)}")
for row in history:
    for key in ("loss", "grad_norm", "kl"):
        if key in row and not math.isfinite(float(row[key])):
            raise RuntimeError(f"non-finite {key} in saved history")

def verify_model(root: Path) -> dict:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    sizes = {name: (root / name).stat().st_size for name in shards}
    if not shards or any(size <= 0 for size in sizes.values()):
        raise RuntimeError(f"incomplete model shards in {root}")
    return {"root": str(root), "shard_count": len(shards), "shard_sizes": sizes}

segments = []
segment_rows = []
for name, expected_unique in (
    ("online_reward_events", 200),
    ("online_reward_events_resume100_250_attempt02", 300),
    ("online_reward_events_resume250_500_attempt01", 500),
):
    events = []
    for path in (run / name).glob("rank_*.jsonl"):
        events.extend(json.loads(line) for line in path.open() if line.strip())
    counts = collections.Counter(int(event["record_index"]) for event in events)
    if len(counts) != expected_unique or set(counts.values()) != {4}:
        raise RuntimeError(f"bad n4 reward-event coverage in {name}")
    segments.append(set(counts))
    segment_rows.append({"name": name, "unique_prompts": len(counts), "events": len(events)})
overlap = sum(len(segments[i] & segments[j]) for i in range(3) for j in range(i + 1, 3))
union = set.union(*segments)
if overlap or union != set(range(1000)):
    raise RuntimeError("n4 segments do not cover RL1000 exactly once")
optimizer = list((checkpoint / "global_step500").glob("**/*optim_states.pt"))
if len(optimizer) != 8 or any(path.stat().st_size <= 0 for path in optimizer):
    raise RuntimeError("n4 optimizer partitions are incomplete")
result = {
    "schema_version": 1,
    "status": "passed",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "recovery_reason": "outer process ended after checkpoint save and before final audit emission",
    "global_step": 500,
    "max_steps": 500,
    "optimizer_log_rows": 500,
    "final_learning_rate": history[-1]["learning_rate"],
    "final_epoch": history[-1]["epoch"],
    "prompt_union_count": len(union),
    "pairwise_prompt_overlap": overlap,
    "rollouts_per_prompt": 4,
    "trajectory_count": 4000,
    "segments": segment_rows,
    "optimizer_partition_count": len(optimizer),
    "checkpoint": verify_model(checkpoint),
    "model_snapshot": verify_model(snapshot),
    "training_restarted": False,
}
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
PY

CURRENT_STAGE=n4_step500_evaluation
state running "$CURRENT_STAGE"
"$PYTHON" "$REPO/scripts/run_full_language_rule_rl_step100_evaluation.py" \
  --model "$N4_SNAPSHOT" --output "$N4_EVAL"

CURRENT_STAGE=n8_same_sft_step500
state running "$CURRENT_STAGE"
"$PYTHON" "$REPO/scripts/run_full_language_rule_rl_n8_step500.py"

CURRENT_STAGE=three_model_split_audit
state running "$CURRENT_STAGE"
"$PYTHON" "$REPO/scripts/run_pathmmu_split_audit_sequence.py" --output "$SPLIT_AUDIT"

CURRENT_STAGE=complete
state completed "$CURRENT_STAGE"
