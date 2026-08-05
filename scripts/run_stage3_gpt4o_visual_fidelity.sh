#!/usr/bin/env bash
set -euo pipefail
umask 077

WORKSPACE="/home/dataset-assist-0/czy/wjy"
REPO="$WORKSPACE/myr1"
INSTALL="$WORKSPACE/pathvlm_r1_v1_a100"
EVAL="$WORKSPACE/pathvlm_revision_eval_a100"
PYTHON="$INSTALL/envs/sft/bin/python"
MODEL="$INSTALL/runs/stage3_process_grpo/gpt4o_full3epoch_seed42_20260801/epoch_model_snapshots/checkpoint-1000"
PANEL="$REPO/protocol/visual_fidelity_panel_v1_20260804.json"
PANEL_SHA="0c7e09a2e870abdb07b63df9cca1a410c73c9922e996bede9771777b636280e8"
RUNNER="$REPO/scripts/run_visual_fidelity_experiment.py"
VERIFY="$REPO/scripts/verify_visual_fidelity_run.py"
SNAPSHOT_VERIFY="$REPO/scripts/verify_checkpoint_snapshot.py"
GPU="${PATHVLM_STAGE3_GPT4O_VISUAL_GPU:-3}"
SHARED_GPU_MODE="${PATHVLM_ALLOW_SHARED_GPU_WITH_STAGE3:-false}"
SHARED_OWNER="${PATHVLM_SHARED_GPU_OWNER_RUN_DIR:-}"
SHARED_MAX_USED_MIB="${PATHVLM_SHARED_GPU_MAX_INITIAL_USED_MIB:-30000}"
SHARED_MIN_FREE_MIB="${PATHVLM_SHARED_GPU_MIN_FREE_MIB:-45000}"

: "${PATHVLM_STAGE3_GPT4O_VISUAL_ROOT:?Set a dedicated GPT-4o visual-fidelity root}"
ROOT="$(readlink -m "$PATHVLM_STAGE3_GPT4O_VISUAL_ROOT")"
case "$ROOT" in
  "$EVAL/runs"/*) ;;
  *) echo "Visual-fidelity root must be a child of $EVAL/runs" >&2; exit 2 ;;
esac
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "Invalid visual-fidelity GPU: $GPU" >&2; exit 2; }
[[ "$SHARED_GPU_MODE" == "true" || "$SHARED_GPU_MODE" == "false" ]] || {
  echo "PATHVLM_ALLOW_SHARED_GPU_WITH_STAGE3 must be true or false" >&2; exit 2;
}
[[ "$SHARED_MAX_USED_MIB" =~ ^[1-9][0-9]*$ && "$SHARED_MIN_FREE_MIB" =~ ^[1-9][0-9]*$ ]] || {
  echo "Shared GPU memory thresholds must be positive integers" >&2; exit 2;
}
if [[ "$SHARED_GPU_MODE" == "true" ]]; then
  : "${SHARED_OWNER:?Shared GPU mode requires PATHVLM_SHARED_GPU_OWNER_RUN_DIR}"
  SHARED_OWNER="$(readlink -f "$SHARED_OWNER")"
  case "$SHARED_OWNER" in
    "$INSTALL/runs/stage3_process_grpo"/*) ;;
    *) echo "Shared GPU owner must be a Stage3 run" >&2; exit 2 ;;
  esac
  worker_count="$(
    { pgrep -af -- "$SHARED_OWNER/output" || true; } |
      { grep -F 'grpo_pathmmu.py' || true; } | wc -l
  )"
  (( worker_count >= 8 )) || { echo "Shared Stage3 owner has only $worker_count live workers" >&2; exit 2; }
  grep -q 'rewards/audited_process_reward' "$SHARED_OWNER/train_segment00.log" || {
    echo "Shared Stage3 owner has not completed an optimizer step" >&2; exit 2;
  }
fi
for path in "$PYTHON" "$MODEL" "$PANEL" "$RUNNER" "$VERIFY" "$SNAPSHOT_VERIFY"; do
  [[ -e "$path" ]] || { echo "Required visual-fidelity input is missing: $path" >&2; exit 2; }
done
[[ "$(sha256sum "$PANEL" | awk '{print $1}')" == "$PANEL_SHA" ]] || {
  echo "Visual-fidelity panel hash mismatch" >&2; exit 2;
}
"$PYTHON" "$SNAPSHOT_VERIFY" "$MODEL" --expected-step 1000 --expected-epoch 2 >/dev/null
IFS=',' read -r used total < <(
  nvidia-smi --id="$GPU" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
)
used="${used//[[:space:]]/}"
total="${total//[[:space:]]/}"
if [[ "$SHARED_GPU_MODE" == "true" ]]; then
  (( used <= SHARED_MAX_USED_MIB && total - used >= SHARED_MIN_FREE_MIB )) || {
    echo "Visual GPU $GPU fails shared-memory admission: used=$used total=$total MiB" >&2; exit 2;
  }
else
  (( used <= 10 )) || { echo "Visual-fidelity GPU $GPU is not idle: ${used} MiB" >&2; exit 2; }
fi

OUTPUT="$ROOT/stage3_gpt4o"
LOG="$ROOT/stage3_gpt4o.log"
VERIFICATION="$OUTPUT/verification.json"
COMPLETED="$ROOT/completed.json"
if [[ -f "$COMPLETED" ]]; then
  "$PYTHON" "$VERIFY" --metrics "$OUTPUT/metrics.json" \
    --expected-label stage3_gpt4o --expected-model "$MODEL" \
    --expected-panel-sha256 "$PANEL_SHA" --output "$VERIFICATION" >/dev/null
  echo "GPT-4o visual-fidelity arm already completed and verified: $ROOT"
  exit 0
fi
[[ ! -e "$ROOT" ]] || { echo "Incomplete visual-fidelity root exists: $ROOT" >&2; exit 3; }
mkdir -p "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
"$PYTHON" "$RUNNER" --model "$MODEL" --model-label stage3_gpt4o \
  --panel "$PANEL" --expected-panel-sha256 "$PANEL_SHA" --output-dir "$OUTPUT" \
  --grid-rows 6 --grid-columns 6 --batch-size 8 --random-permutations 5 \
  >"$LOG" 2>&1
"$PYTHON" "$VERIFY" --metrics "$OUTPUT/metrics.json" \
  --expected-label stage3_gpt4o --expected-model "$MODEL" \
  --expected-panel-sha256 "$PANEL_SHA" --output "$VERIFICATION" >/dev/null
"$PYTHON" - "$COMPLETED" "$OUTPUT/metrics.json" "$VERIFICATION" "$REPO" \
  "$SHARED_GPU_MODE" "$SHARED_OWNER" "$SHARED_MAX_USED_MIB" "$SHARED_MIN_FREE_MIB" <<'PY'
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

path = Path(sys.argv[1])
metrics = Path(sys.argv[2])
verification = Path(sys.argv[3])
value = {
    "schema_version": 1,
    "status": "completed_single_arm",
    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "model_label": "stage3_gpt4o",
    "five_arm_comparison_completed": False,
    "gpu_admission": {
        "mode": "shared_with_stage3" if sys.argv[5] == "true" else "idle_exclusive",
        "stage3_owner_run": sys.argv[6] or None,
        "maximum_initial_used_mib": int(sys.argv[7]),
        "minimum_initial_free_mib": int(sys.argv[8]),
    },
    "metrics": str(metrics.resolve()),
    "metrics_sha256": hashlib.sha256(metrics.read_bytes()).hexdigest(),
    "verification": str(verification.resolve()),
    "verification_sha256": hashlib.sha256(verification.read_bytes()).hexdigest(),
    "repository_commit": subprocess.check_output(
        ["/home/dataset-assist-0/czy/wjy/.local-git/usr/bin/git", "-C", sys.argv[4], "rev-parse", "HEAD"],
        text=True,
    ).strip(),
}
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
echo "GPT-4o visual-fidelity arm completed and verified: $ROOT"
