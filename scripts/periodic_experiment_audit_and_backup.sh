#!/usr/bin/env bash
set -euo pipefail
umask 077

WORKSPACE_ROOT="${WJY_WORK_ROOT:-/home/dataset-assist-0/czy/wjy}"
REPO_ROOT="$WORKSPACE_ROOT/myr1"
STAGE3_RUN_ROOT="${PATHVLM_GPT4O_RUN_ROOT:-$WORKSPACE_ROOT/pathvlm_r1_v1_a100/runs/stage3_process_grpo/gpt4o_full3epoch_seed42_20260801}"
EVAL_ROOT="${PATHVLM_HOSTED_EVAL_ROOT:-$WORKSPACE_ROOT/pathvlm_revision_eval_a100/runs/hosted_baseline_full_20260801}"
QWEN_PATHVQA="$EVAL_ROOT/qwen-vl-plus/pathvqa_full_fresh_20260802"
HAIKU_PATHVQA="$EVAL_ROOT/claude-haiku-4-5-20251001/pathvqa"
REPORT_ROOT="${PATHVLM_PERIODIC_REPORT_ROOT:-$WORKSPACE_ROOT/pathvlm_revision_eval_a100/reports/periodic_experiment_audit}"
STATE_ROOT="$REPORT_ROOT/state"
AUDIT_ROOT="$REPORT_ROOT/audits"
HF_REPO_ID="${PATHVLM_EVALUATION_BACKUP_REPO:-Freddie1946/PathVLM-R1-Revision-Evaluation-Results}"
HF_REMOTE_ROOT="${PATHVLM_PERIODIC_HF_REMOTE_ROOT:-live/active_experiments}"
HF_BIN="${HF_BIN:-/usr/local/bin/hf}"
PYTHON="${PYTHON:-/usr/bin/python3}"
TIMEOUT_BIN="${TIMEOUT_BIN:-/usr/bin/timeout}"
HF_AUTH_TIMEOUT_SECONDS="${PATHVLM_HF_AUTH_TIMEOUT_SECONDS:-60}"
HF_UPLOAD_TIMEOUT_SECONDS="${PATHVLM_HF_UPLOAD_TIMEOUT_SECONDS:-600}"
export HF_HOME="${HF_HOME:-$WORKSPACE_ROOT/cache/huggingface}"

[[ -d "$REPO_ROOT" && -d "$STAGE3_RUN_ROOT" && -d "$EVAL_ROOT" ]] || {
  echo "Required repository or run root is missing" >&2
  exit 2
}
[[ -x "$PYTHON" ]] || { echo "Python is missing: $PYTHON" >&2; exit 2; }
[[ -x "$TIMEOUT_BIN" ]] || { echo "timeout is missing: $TIMEOUT_BIN" >&2; exit 2; }
[[ "$HF_AUTH_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "Invalid HF auth timeout: $HF_AUTH_TIMEOUT_SECONDS" >&2
  exit 2
}
[[ "$HF_UPLOAD_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "Invalid HF upload timeout: $HF_UPLOAD_TIMEOUT_SECONDS" >&2
  exit 2
}
mkdir -p "$STATE_ROOT" "$AUDIT_ROOT"

exec 9>"$STATE_ROOT/periodic.lock"
flock -n 9 || exit 0

STAMP="$(date '+%Y%m%d_%H%M%S')"
AUDIT_PATH="$AUDIT_ROOT/${STAMP}.json"
STAGING="$(mktemp -d "$STATE_ROOT/staging.XXXXXX")"
trap 'rm -rf -- "$STAGING"' EXIT

"$PYTHON" - "$STAGE3_RUN_ROOT" "$QWEN_PATHVQA" "$HAIKU_PATHVQA" "$AUDIT_PATH" <<'PY'
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

stage3, qwen, haiku, output = map(Path, sys.argv[1:])

def json_value(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"read_error": f"{type(exc).__name__}: {exc}"}

def jsonl_summary(path):
    result = {"path": str(path), "rows": 0, "invalid_json_rows": 0, "bytes": 0}
    if not path.is_file():
        result["missing"] = True
        return result
    result["bytes"] = path.stat().st_size
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            result["rows"] += 1
            try:
                json.loads(line)
            except Exception:
                result["invalid_json_rows"] += 1
    return result

def process_count(fragment):
    completed = subprocess.run(
        ["pgrep", "-fc", fragment], text=True, capture_output=True, check=False
    )
    try:
        return int(completed.stdout.strip() or "0")
    except ValueError:
        return 0

def latest_step():
    steps = []
    for path in stage3.glob("train_segment*.log"):
        text = path.read_text(encoding="utf-8", errors="replace")
        steps.extend(int(value) for value in re.findall(r"(\d+)/1500", text))
    return max(steps, default=0)

budget = json_value(stage3 / "judge" / "budget_ledger.json")
fallback = json_value(stage3 / "judge" / "rule_fallback_ledger.json")
checkpoints = sorted(
    (int(path.name.split("-")[-1]), path.name)
    for path in (stage3 / "output").glob("checkpoint-*")
    if path.name.split("-")[-1].isdigit()
)
reward_segments = {}
for segment in sorted((stage3 / "reward_audit").glob("segment*")):
    summaries = [jsonl_summary(path) for path in sorted(segment.glob("rank_*.jsonl"))]
    reward_segments[segment.name] = {
        "rows": sum(row["rows"] for row in summaries),
        "invalid_json_rows": sum(row["invalid_json_rows"] for row in summaries),
        "bytes": sum(row["bytes"] for row in summaries),
        "rank_files": len(summaries),
    }

def pathvqa_summary(root):
    predictions = jsonl_summary(root / "predictions.jsonl")
    judgments = jsonl_summary(root / "gpt-5-mini_semantic_judgments.jsonl")
    skipped = jsonl_summary(root / "gpt-5-mini_semantic_judgments_skipped_predictions.jsonl")
    semantic_metrics_path = root / "gpt-5-mini_semantic_metrics.json"
    semantic_metrics = (
        json_value(semantic_metrics_path) if semantic_metrics_path.is_file() else None
    )
    return {
        "predictions": predictions,
        "semantic_judgments": judgments,
        "skipped_predictions": skipped,
        "unique_judgments_plus_skips": judgments["rows"] + skipped["rows"],
        "semantic_metrics": semantic_metrics,
        "complete": bool(
            isinstance(semantic_metrics, dict)
            and semantic_metrics.get("status") == "completed"
            and semantic_metrics.get("expected_count") == predictions["rows"]
        ),
    }

gpu_query = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
    capture_output=True,
    check=False,
)
gpus = []
for line in gpu_query.stdout.splitlines():
    fields = [field.strip() for field in line.split(",")]
    if len(fields) == 4 and all(field.isdigit() for field in fields):
        gpus.append(
            {
                "index": int(fields[0]),
                "memory_used_mib": int(fields[1]),
                "memory_total_mib": int(fields[2]),
                "utilization_percent": int(fields[3]),
            }
        )

disk = shutil.disk_usage(stage3)
audit = {
    "schema_version": 1,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "formal_result": False,
    "audit_kind": "periodic_runtime_and_backup_audit",
    "stage3_gpt4o": {
        "latest_observed_step": latest_step(),
        "max_steps": 1500,
        "supervisor_processes": process_count("supervise_stage3_gpt4o_formal.sh"),
        "launcher_processes": process_count("launch_stage3_gpt4o_formal.sh"),
        "torchrun_processes": process_count("torch.distributed.run.*grpo_pathmmu.py"),
        "checkpoints": [name for _, name in checkpoints],
        "budget": {
            "committed_spend_usd": budget.get("committed_spend_usd"),
            "completed_physical_requests": budget.get("completed_unique_requests"),
            "inflight_or_unresolved_reservations": len(budget.get("reservations", {})),
            "budget_breached": budget.get("budget_breached"),
            "technical_capacity_usd": budget.get("limit_usd"),
            "maximum_physical_requests": budget.get("max_unique_requests"),
        },
        "fallback": {
            "total_used": fallback.get("total_used"),
            "consecutive_used": fallback.get("consecutive_used"),
            "total_limit": fallback.get("total_limit"),
            "consecutive_limit": fallback.get("consecutive_limit"),
        },
        "reward_audit": reward_segments,
    },
    "pathvqa_semantic_judge": {
        "qwen_vl_plus": pathvqa_summary(qwen),
        "claude_haiku_4_5": pathvqa_summary(haiku),
        "worker_processes": process_count("pathvqa_llm_judge.py"),
    },
    "gpu": {"query_status": gpu_query.returncode, "devices": gpus},
    "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
    "actions": {
        "training_restart_attempted": False,
        "reason": "periodic auditor is read-only; the classified Stage3 supervisor owns recovery",
    },
}
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_suffix(output.suffix + ".tmp")
temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY

stage_file() {
  local source="$1"
  local relative="$2"
  [[ -f "$source" ]] || return 0
  mkdir -p "$STAGING/$(dirname "$relative")"
  cp -- "$source" "$STAGING/$relative"
}

while IFS= read -r -d '' source; do
  stage_file "$source" "stage3_gpt4o/${source#"$STAGE3_RUN_ROOT"/}"
done < <(
  find "$STAGE3_RUN_ROOT" -maxdepth 1 -type f \
    \( -name '*.json' -o -name '*.jsonl' -o -name '*.log' \) \
    ! -name '*.lock' -print0
  find "$STAGE3_RUN_ROOT/reward_audit" -type f -name '*.jsonl' -print0
  find "$STAGE3_RUN_ROOT/judge" -maxdepth 1 -type f -name '*.json' -print0
)

for pair in "qwen_vl_plus:$QWEN_PATHVQA" "claude_haiku_4_5:$HAIKU_PATHVQA"; do
  label="${pair%%:*}"
  root="${pair#*:}"
  for name in \
    predictions.jsonl metrics.json run_config.json \
    gpt-5-mini_semantic_judgments.jsonl \
    gpt-5-mini_semantic_judgments_skipped_predictions.jsonl \
    gpt-5-mini_semantic_metrics.json \
    judge_retry_20260802.log; do
    stage_file "$root/$name" "pathvqa_semantic_judge/$label/$name"
  done
done

stage_file "$AUDIT_PATH" "status/latest_audit.json"

"$PYTHON" - "$STAGING" "$STAMP" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
stamp = sys.argv[2]
rows = []
for path in sorted(path for path in root.rglob("*") if path.is_file()):
    relative = path.relative_to(root).as_posix()
    if relative in {"snapshot_manifest.json", "snapshot_manifest.tsv"}:
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows.append((relative, path.stat().st_size, digest))
serialization = "".join(f"{name}\t{size}\t{digest}\n" for name, size, digest in rows)
aggregate = hashlib.sha256(serialization.encode("utf-8")).hexdigest()
(root / "snapshot_manifest.tsv").write_text(serialization, encoding="utf-8")
(root / "snapshot_manifest.json").write_text(
    json.dumps(
        {
            "schema_version": 1,
            "snapshot_timestamp_cst": stamp,
            "file_count": len(rows),
            "total_bytes": sum(size for _, size, _ in rows),
            "aggregate_manifest_sha256": aggregate,
            "contains_model_weights": False,
            "contains_credentials": False,
            "snapshot_class": "mutable_disaster_recovery_copy_not_formal_result",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
print(aggregate)
PY
AGGREGATE_SHA256="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["aggregate_manifest_sha256"])' "$STAGING/snapshot_manifest.json")"
LAST_HASH_PATH="$STATE_ROOT/last_uploaded_aggregate_sha256"
LAST_HASH="$(cat "$LAST_HASH_PATH" 2>/dev/null || true)"

if [[ "$AGGREGATE_SHA256" == "$LAST_HASH" ]]; then
  echo "$STAMP unchanged; HF upload skipped"
  exit 0
fi
if [[ "${PATHVLM_PERIODIC_BACKUP_DRY_RUN:-false}" == "true" ]]; then
  echo "$STAMP dry-run audit passed; aggregate=$AGGREGATE_SHA256"
  exit 0
fi

[[ -x "$HF_BIN" ]] || { echo "Hugging Face CLI is missing: $HF_BIN" >&2; exit 2; }
authenticated=false
for delay in 0 15 45; do
  (( delay == 0 )) || sleep "$delay"
  if "$TIMEOUT_BIN" --signal=TERM --kill-after=5s "${HF_AUTH_TIMEOUT_SECONDS}s" \
    "$HF_BIN" auth whoami >/dev/null 2>&1; then
    authenticated=true
    break
  fi
  echo "$STAMP HF authentication check failed; bounded retry follows" >&2
done
[[ "$authenticated" == "true" ]] || {
  echo "$STAMP HF authentication failed after three attempts" >&2
  exit 3
}

uploaded=false
UPLOAD_RESULT=""
for delay in 0 15 45; do
  (( delay == 0 )) || sleep "$delay"
  if UPLOAD_RESULT="$($TIMEOUT_BIN --signal=TERM --kill-after=5s \
    "${HF_UPLOAD_TIMEOUT_SECONDS}s" "$HF_BIN" upload \
    "$HF_REPO_ID" "$STAGING" "$HF_REMOTE_ROOT" \
    --repo-type dataset \
    --commit-message "Periodic active experiment backup $STAMP" \
    --commit-description "Mutable disaster-recovery copy; see snapshot_manifest.json for exact hashes. No model weights or credentials." \
    --quiet)"; then
    uploaded=true
    break
  fi
  echo "$STAMP HF upload failed; bounded retry follows" >&2
done
[[ "$uploaded" == "true" ]] || {
  echo "$STAMP HF upload failed after three attempts" >&2
  exit 4
}
printf '%s\n' "$AGGREGATE_SHA256" > "$LAST_HASH_PATH"
printf '%s\t%s\t%s\n' "$STAMP" "$AGGREGATE_SHA256" "$UPLOAD_RESULT" >> "$STATE_ROOT/upload_history.tsv"
echo "$STAMP audit and private HF backup completed: $UPLOAD_RESULT"
