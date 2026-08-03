#!/usr/bin/env bash
set -euo pipefail
umask 077

WORKSPACE="/home/dataset-assist-0/czy/wjy"
REPO="$WORKSPACE/myr1"
INSTALL="$WORKSPACE/pathvlm_r1_v1_a100"
EVAL="$WORKSPACE/pathvlm_revision_eval_a100"
PYTHON="$INSTALL/envs/sft/bin/python"
KIMI_RUN="$INSTALL/runs/stage3_process_grpo/kimi26_full3epoch_seed42_fresh_20260803"
GPT_RUN="$INSTALL/runs/stage3_process_grpo/gpt4o_full3epoch_seed42_20260801"
KIMI_SELECTION="$KIMI_RUN/validation_0385_epochs/selection.json"
GPT_SELECTION="$GPT_RUN/validation_0385_epochs/selection.json"
VALIDATE="$REPO/scripts/run_stage3_formal_validation.sh"
EVALUATE="$REPO/scripts/run_stage3_selected_full_evaluations.sh"
VISUAL="$REPO/scripts/run_visual_fidelity_experiment.py"
VERIFY_VISUAL="$REPO/scripts/verify_visual_fidelity_run.py"
SUMMARIZE_VISUAL="$REPO/scripts/summarize_visual_fidelity_runs.py"
PANEL="$REPO/protocol/visual_fidelity_panel_v1_20260804.json"
PANEL_SHA="0c7e09a2e870abdb07b63df9cca1a410c73c9922e996bede9771777b636280e8"
RUN_ROOT="${PATHVLM_POST_KIMI_ROOT:-$EVAL/runs/stage3_post_kimi_closed_and_visual_20260804}"
EVAL_ROOT="$RUN_ROOT/selected_closed_evaluations"
VISUAL_ROOT="$RUN_ROOT/visual_fidelity"
EVENTS="$RUN_ROOT/events.jsonl"

case "$(readlink -m "$RUN_ROOT")" in
  "$EVAL/runs"/*) ;;
  *) echo "Post-Kimi root must be under $EVAL/runs" >&2; exit 2 ;;
esac
for path in "$PYTHON" "$VALIDATE" "$EVALUATE" "$VISUAL" "$VERIFY_VISUAL" \
  "$SUMMARIZE_VISUAL" "$PANEL" "$GPT_SELECTION"; do
  [[ -e "$path" ]] || { echo "Required post-Kimi input is missing: $path" >&2; exit 2; }
done
[[ "$(sha256sum "$PANEL" | awk '{print $1}')" == "$PANEL_SHA" ]] || {
  echo "Frozen visualization panel hash mismatch" >&2
  exit 2
}

mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/sequence.lock"
flock -n 9 || { echo "Another post-Kimi sequence owns the lock" >&2; exit 2; }
[[ ! -f "$RUN_ROOT/completed.json" ]] || {
  echo "Post-Kimi sequence is already complete: $RUN_ROOT/completed.json" >&2
  exit 2
}

record_event() {
  "$PYTHON" - "$EVENTS" "$1" "${2:-}" <<'PY'
import datetime
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
row = {
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "event": sys.argv[2],
    "detail": sys.argv[3],
}
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
}

"$PYTHON" - "$KIMI_RUN" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
audit = run / "supervisor_recovery_audit.jsonl"
try:
    events = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"cannot read Kimi supervisor audit: {exc}") from exc
if not events or events[-1].get("event") != "training_completed":
    raise SystemExit("Kimi training has not completed successfully; this script never waits or guesses")
for step in (500, 1000, 1500):
    snapshot = run / "epoch_model_snapshots" / f"checkpoint-{step}"
    state = snapshot / "trainer_state.json"
    if not snapshot.is_dir() or not state.is_file():
        raise SystemExit(f"Kimi epoch snapshot is missing: {snapshot}")
    if int(json.loads(state.read_text(encoding="utf-8")).get("global_step", -1)) != step:
        raise SystemExit(f"Kimi snapshot trainer step mismatch: {snapshot}")
ledger = json.loads((run / "judge" / "budget_ledger.json").read_text(encoding="utf-8"))
if ledger.get("budget_breached") is not False:
    raise SystemExit("Kimi budget ledger is breached or incomplete")
PY
record_event "kimi_completion_verified" "supervisor completion, three snapshots, and budget ledger passed"

if [[ ! -f "$KIMI_SELECTION" ]]; then
  record_event "kimi_validation_started" "three frozen epoch snapshots on validation385"
  PATHVLM_STAGE3_COMPLETED_RUN_DIR="$KIMI_RUN" bash "$VALIDATE"
  record_event "kimi_validation_completed" "$KIMI_SELECTION"
fi

readarray -t SELECTED_MODELS < <(
  "$PYTHON" - "$GPT_SELECTION" "$KIMI_SELECTION" <<'PY'
import json
import sys
from pathlib import Path

for path_text in sys.argv[1:]:
    path = Path(path_text)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "completed" or value.get("selection_split") != "pathmmu_validation_0385":
        raise SystemExit(f"invalid Stage3 selection: {path}")
    if value.get("test_accessed") is not False or int(value.get("selected_step", -1)) not in {500, 1000, 1500}:
        raise SystemExit(f"invalid Stage3 selection provenance: {path}")
    model = Path(value["selected_model_path"]).resolve()
    if not model.is_dir():
        raise SystemExit(f"selected model is missing: {model}")
    print(model)
PY
)
[[ "${#SELECTED_MODELS[@]}" -eq 2 ]] || exit 2
GPT_MODEL="${SELECTED_MODELS[0]}"
KIMI_MODEL="${SELECTED_MODELS[1]}"

if [[ ! -f "$EVAL_ROOT/completed.json" ]]; then
  record_event "closed_evaluations_started" "GPT-4o and Kimi PathMMU, PathVQA yes/no, and OmniMedVQA"
  PATHVLM_STAGE3_SELECTED_EVAL_ROOT="$EVAL_ROOT" bash "$EVALUATE"
  record_event "closed_evaluations_completed" "$EVAL_ROOT"
fi

SFT3000="$INSTALL/transferred_checkpoints/sft_n3000_seed42_epoch03_step1125"
SFT4000="$INSTALL/runs/stage2_control_sft4000/n1000_seed0042/sft4000_control_rl1000_seed0042_epoch02_20260729_212746/output"
STAGE2="$INSTALL/transferred_checkpoints/outcome_grpo_n1000_seed42_epoch02_step1000"
VISUAL_SPECS=(
  "sft3000|$SFT3000|0"
  "sft4000|$SFT4000|1"
  "stage2_outcome_grpo|$STAGE2|2"
  "stage3_gpt4o|$GPT_MODEL|3"
  "stage3_kimi26|$KIMI_MODEL|4"
)

mapfile -t GPU_MEMORY < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
[[ "${#GPU_MEMORY[@]}" -eq 8 ]] || { echo "Exactly eight GPUs are required" >&2; exit 2; }
for gpu in 0 1 2 3 4; do
  used="${GPU_MEMORY[$gpu]//[[:space:]]/}"
  (( used <= 10 )) || { echo "Visual-fidelity GPU $gpu is not idle: ${used} MiB" >&2; exit 2; }
done

mkdir -p "$VISUAL_ROOT/logs"
record_event "visual_fidelity_started" "five frozen arms on the outcome-blind validation panel"
pids=()
for spec in "${VISUAL_SPECS[@]}"; do
  IFS='|' read -r label model gpu <<<"$spec"
  output="$VISUAL_ROOT/$label"
  [[ -f "$model/config.json" ]] || { echo "Visual-fidelity model is missing: $model" >&2; exit 2; }
  if [[ -f "$output/metrics.json" ]]; then
    continue
  fi
  [[ ! -e "$output" ]] || { echo "Incomplete visual-fidelity output exists: $output" >&2; exit 3; }
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
    "$PYTHON" "$VISUAL" --model "$model" --model-label "$label" \
      --panel "$PANEL" --expected-panel-sha256 "$PANEL_SHA" \
      --output-dir "$output" --grid-rows 6 --grid-columns 6 \
      --batch-size 8 --random-permutations 5
  ) >"$VISUAL_ROOT/logs/$label.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { record_event "visual_fidelity_failed"; exit 4; }

ARM_ARGS=()
for spec in "${VISUAL_SPECS[@]}"; do
  IFS='|' read -r label model gpu <<<"$spec"
  output="$VISUAL_ROOT/$label"
  "$PYTHON" "$VERIFY_VISUAL" --metrics "$output/metrics.json" \
    --expected-label "$label" --expected-model "$model" \
    --expected-panel-sha256 "$PANEL_SHA" --output "$output/verification.json" \
    >"$VISUAL_ROOT/logs/${label}_verification.log"
  ARM_ARGS+=(--arm "$label" "$model" "$output/metrics.json")
done
[[ ! -e "$VISUAL_ROOT/comparison" ]] || {
  echo "Refusing to overwrite visual-fidelity comparison" >&2
  exit 4
}
"$PYTHON" "$SUMMARIZE_VISUAL" "${ARM_ARGS[@]}" \
  --expected-panel-sha256 "$PANEL_SHA" --output-dir "$VISUAL_ROOT/comparison" \
  >"$VISUAL_ROOT/logs/comparison.log"
record_event "visual_fidelity_completed" "$VISUAL_ROOT/comparison"

"$PYTHON" - "$RUN_ROOT/completed.json" "$KIMI_SELECTION" "$EVAL_ROOT" "$VISUAL_ROOT" <<'PY'
import datetime
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
value = {
    "schema_version": 1,
    "status": "completed",
    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "kimi_selection": str(Path(sys.argv[2]).resolve()),
    "closed_evaluations": str(Path(sys.argv[3]).resolve()),
    "pathvqa_free_form_status": "postponed_and_excluded",
    "visual_fidelity": str(Path(sys.argv[4]).resolve()),
}
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
record_event "sequence_completed" "$RUN_ROOT/completed.json"
echo "Post-Kimi closed-question evaluations and visual-fidelity experiment completed: $RUN_ROOT"
