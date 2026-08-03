#!/usr/bin/env bash
set -euo pipefail
umask 077

WORKSPACE="/home/dataset-assist-0/czy/wjy"
REPO="$WORKSPACE/myr1"
INSTALL="$WORKSPACE/pathvlm_r1_v1_a100"
EVAL="$WORKSPACE/pathvlm_revision_eval_a100"
PYTHON="$INSTALL/envs/sft/bin/python"
PATHMMU_RUNNER="$REPO/scripts/run_pathmmu_qwen_diagnostic.py"
EXTERNAL_RUNNER="$REPO/scripts/run_external_vqa_qwen.py"
SMOKE_VERIFY="$REPO/scripts/verify_local_baseline_smoke.py"
FULL_VERIFY="$REPO/scripts/verify_local_baseline_full.py"
SNAPSHOT_VERIFY="$REPO/scripts/verify_checkpoint_snapshot.py"

GPT_RUN="$INSTALL/runs/stage3_process_grpo/gpt4o_full3epoch_seed42_20260801"
KIMI_RUN="$INSTALL/runs/stage3_process_grpo/kimi26_full3epoch_seed42_fresh_20260803"
GPT_SELECTION="$GPT_RUN/validation_0385_epochs/selection.json"
KIMI_SELECTION="$KIMI_RUN/validation_0385_epochs/selection.json"

PATHMMU_VALID="$INSTALL/data/pathmmu_image_disjoint_v2/rewritten_records/validation_0385.json"
PATHMMU_TEST="$INSTALL/data/pathmmu_image_disjoint_v2/rewritten_records/test_0999.json"
PATHVQA="$EVAL/datasets/external_vqa_contract_v1_20260729/pathvqa_test_6719.json"
OMNI="$EVAL/datasets/external_vqa_contract_v1_20260729/omnimedvqa_four_sources_8518.json"
PATHMMU_VALID_SHA="f52c9a412da579db8fb8bf52f6fa320e7c4ae9cb42b6cf1bdf62f77ab158dcf0"
PATHMMU_TEST_SHA="3420c8ff2f1642801a5ecebd21271f4e7de5877d6459814473ddbcdda996b549"
PATHVQA_SHA="5fa7319784dd47c81027c2b99cbb0d0d472638d0dc5c5169dcbd5d43987784be"
OMNI_SHA="04ba0790410cd5de4c95689d349574b102034a153afa00d952e351d3c13991a7"

: "${PATHVLM_STAGE3_SELECTED_EVAL_ROOT:?Set a dedicated Stage3 selected-model evaluation root}"
RUN_ROOT="$(readlink -m "$PATHVLM_STAGE3_SELECTED_EVAL_ROOT")"
case "$RUN_ROOT" in
  "$EVAL/runs"/*) ;;
  *) echo "Evaluation root must be a child of $EVAL/runs" >&2; exit 2 ;;
esac

for path in "$PYTHON" "$PATHMMU_RUNNER" "$EXTERNAL_RUNNER" "$SMOKE_VERIFY" \
  "$FULL_VERIFY" "$SNAPSHOT_VERIFY" "$GPT_SELECTION" "$KIMI_SELECTION" \
  "$PATHMMU_VALID" "$PATHMMU_TEST" "$PATHVQA" "$OMNI"; do
  [[ -e "$path" ]] || { echo "Required evaluation input is missing: $path" >&2; exit 2; }
done
[[ "$(sha256sum "$PATHMMU_VALID" | awk '{print $1}')" == "$PATHMMU_VALID_SHA" ]] || exit 2
[[ "$(sha256sum "$PATHMMU_TEST" | awk '{print $1}')" == "$PATHMMU_TEST_SHA" ]] || exit 2
[[ "$(sha256sum "$PATHVQA" | awk '{print $1}')" == "$PATHVQA_SHA" ]] || exit 2
[[ "$(sha256sum "$OMNI" | awk '{print $1}')" == "$OMNI_SHA" ]] || exit 2

readarray -t SELECTION_VALUES < <(
  "$PYTHON" - "$GPT_SELECTION" "$KIMI_SELECTION" "$GPT_RUN" "$KIMI_RUN" <<'PY'
import json
import sys
from pathlib import Path

for selection_name, path_text, run_text in (
    ("gpt4o", sys.argv[1], sys.argv[3]),
    ("kimi26", sys.argv[2], sys.argv[4]),
):
    path = Path(path_text)
    run = Path(run_text).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "completed" or value.get("selection_split") != "pathmmu_validation_0385":
        raise SystemExit(f"{selection_name} selection contract is incomplete")
    if value.get("test_accessed") is not False or int(value.get("selected_step", -1)) not in {500, 1000, 1500}:
        raise SystemExit(f"{selection_name} selection provenance is invalid")
    model = Path(value["selected_model_path"]).resolve()
    expected = run / "epoch_model_snapshots" / f"checkpoint-{value['selected_step']}"
    if model != expected or not model.is_dir():
        raise SystemExit(f"{selection_name} selected model path mismatch")
    print(model)
    print(int(value["selected_step"]))
PY
)
[[ "${#SELECTION_VALUES[@]}" -eq 4 ]] || exit 2
GPT_MODEL="${SELECTION_VALUES[0]}"
GPT_STEP="${SELECTION_VALUES[1]}"
KIMI_MODEL="${SELECTION_VALUES[2]}"
KIMI_STEP="${SELECTION_VALUES[3]}"

"$PYTHON" "$SNAPSHOT_VERIFY" "$GPT_MODEL" \
  --expected-step "$GPT_STEP" --expected-epoch "$((GPT_STEP / 500))" >/dev/null
"$PYTHON" "$SNAPSHOT_VERIFY" "$KIMI_MODEL" \
  --expected-step "$KIMI_STEP" --expected-epoch "$((KIMI_STEP / 500))" >/dev/null

mapfile -t GPU_MEMORY < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
[[ "${#GPU_MEMORY[@]}" -eq 8 ]] || exit 2
for index in "${!GPU_MEMORY[@]}"; do
  used="${GPU_MEMORY[$index]//[[:space:]]/}"
  (( used <= 10 )) || { echo "GPU $index is not idle: ${used} MiB" >&2; exit 2; }
done

mkdir -p "$RUN_ROOT/logs"
EVENTS="$RUN_ROOT/events.jsonl"
CONTRACT="$RUN_ROOT/run_contract.json"
GIT_COMMIT="$(/home/dataset-assist-0/czy/wjy/.local-git/usr/bin/git -C "$REPO" rev-parse HEAD)"

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

"$PYTHON" - "$CONTRACT" "$GPT_MODEL" "$GPT_STEP" "$KIMI_MODEL" "$KIMI_STEP" \
  "$GIT_COMMIT" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
value = {
    "schema_version": 1,
    "status": "frozen",
    "models": {
        "stage3_gpt4o": {"path": sys.argv[2], "selected_step": int(sys.argv[3])},
        "stage3_kimi26": {"path": sys.argv[4], "selected_step": int(sys.argv[5])},
    },
    "tasks": {
        "pathmmu": {"count": 999, "split_role": "test999_development"},
        "pathvqa": {"count": 6719, "split_role": "external_test"},
        "omnimedvqa": {"count": 8518, "split_role": "external_test"},
    },
    "smoke_count_per_model_task": 16,
    "accuracy_used_as_smoke_gate": False,
    "repository_commit": sys.argv[6],
}
if path.exists():
    prior = json.loads(path.read_text(encoding="utf-8"))
    if prior != value:
        raise SystemExit("existing Stage3 full-evaluation contract differs")
else:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
PY

model_config_sha() {
  sha256sum "$1/config.json" | awk '{print $1}'
}

run_smoke() {
  local label="$1" model="$2" task="$3" data="$4" gpu="$5"
  local output="$RUN_ROOT/$label/${task}_smoke16"
  local log="$RUN_ROOT/logs/${label}_${task}_smoke16.log"
  mkdir -p "$RUN_ROOT/$label"
  if [[ -f "$output/metrics.json" ]]; then
    return
  fi
  [[ ! -e "$output" ]] || { echo "Incomplete smoke output exists: $output" >&2; return 3; }
  export CUDA_VISIBLE_DEVICES="$gpu"
  if [[ "$task" == "pathmmu" ]]; then
    "$PYTHON" "$PATHMMU_RUNNER" --model "$model" --backend qwen2_5_vl \
      --data "$data" --output-dir "$output" --split-role validation_smoke \
      --batch-size 1 --limit 16 >"$log" 2>&1
  else
    "$PYTHON" "$EXTERNAL_RUNNER" --task "$task" --model "$model" \
      --backend qwen2_5_vl --data "$data" --output-dir "$output" \
      --split-role adapter_smoke --batch-size 8 --limit 16 >"$log" 2>&1
  fi
}

verify_smoke() {
  local label="$1" model="$2" task="$3" data_sha="$4"
  local output="$RUN_ROOT/$label/${task}_smoke16"
  "$PYTHON" "$SMOKE_VERIFY" --task "$task" --metrics "$output/metrics.json" \
    --predictions "$output/predictions.jsonl" --expected-count 16 \
    --expected-data-sha256 "$data_sha" \
    --expected-model-config-sha256 "$(model_config_sha "$model")" \
    --minimum-nonempty-rate 0.80 --minimum-parseable-rate 0.80 \
    --maximum-cap-hit-rate 0.20 --output "$output/smoke_gate.json" >/dev/null
}

run_full() {
  local label="$1" model="$2" task="$3" data="$4" gpu="$5"
  local suffix count split output log
  if [[ "$task" == "pathmmu" ]]; then
    suffix="pathmmu_test999"; count=999; split="test999_development"
  elif [[ "$task" == "pathvqa" ]]; then
    suffix="pathvqa_full6719"; count=6719; split="external_test"
  else
    suffix="omnimedvqa_full8518"; count=8518; split="external_test"
  fi
  output="$RUN_ROOT/$label/$suffix"
  log="$RUN_ROOT/logs/${label}_${suffix}.log"
  if [[ -f "$output/metrics.json" ]]; then
    return
  fi
  local resume=()
  [[ ! -e "$output" ]] || resume=(--resume)
  export CUDA_VISIBLE_DEVICES="$gpu"
  if [[ "$task" == "pathmmu" ]]; then
    "$PYTHON" "$PATHMMU_RUNNER" --model "$model" --backend qwen2_5_vl \
      --data "$data" --output-dir "$output" --split-role "$split" \
      --batch-size 1 "${resume[@]}" >"$log" 2>&1
  else
    "$PYTHON" "$EXTERNAL_RUNNER" --task "$task" --model "$model" \
      --backend qwen2_5_vl --data "$data" --output-dir "$output" \
      --split-role "$split" --batch-size 8 "${resume[@]}" >"$log" 2>&1
  fi
}

verify_full() {
  local label="$1" model="$2" task="$3" data_sha="$4"
  local suffix count split output
  if [[ "$task" == "pathmmu" ]]; then
    suffix="pathmmu_test999"; count=999; split="test999_development"
  elif [[ "$task" == "pathvqa" ]]; then
    suffix="pathvqa_full6719"; count=6719; split="external_test"
  else
    suffix="omnimedvqa_full8518"; count=8518; split="external_test"
  fi
  output="$RUN_ROOT/$label/$suffix"
  "$PYTHON" "$FULL_VERIFY" --task "$task" --metrics "$output/metrics.json" \
    --predictions "$output/predictions.jsonl" --run-config "$output/run_config.json" \
    --expected-count "$count" --expected-split-role "$split" \
    --expected-data-sha256 "$data_sha" \
    --expected-model-config-sha256 "$(model_config_sha "$model")" \
    --output "$output/full_integrity_verified.json" >/dev/null
}

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false PYTHONPATH="$REPO/scripts"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

SPECS=(
  "stage3_gpt4o|$GPT_MODEL|pathmmu|$PATHMMU_VALID|0|$PATHMMU_VALID_SHA"
  "stage3_gpt4o|$GPT_MODEL|pathvqa|$PATHVQA|1|$PATHVQA_SHA"
  "stage3_gpt4o|$GPT_MODEL|omnimedvqa|$OMNI|2|$OMNI_SHA"
  "stage3_kimi26|$KIMI_MODEL|pathmmu|$PATHMMU_VALID|3|$PATHMMU_VALID_SHA"
  "stage3_kimi26|$KIMI_MODEL|pathvqa|$PATHVQA|4|$PATHVQA_SHA"
  "stage3_kimi26|$KIMI_MODEL|omnimedvqa|$OMNI|5|$OMNI_SHA"
)

record_event "smoke_phase_started" "six native Stage3 model-task checks"
pids=()
for spec in "${SPECS[@]}"; do
  IFS='|' read -r label model task data gpu data_sha <<<"$spec"
  run_smoke "$label" "$model" "$task" "$data" "$gpu" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { record_event "smoke_phase_failed"; exit 3; }
for spec in "${SPECS[@]}"; do
  IFS='|' read -r label model task data gpu data_sha <<<"$spec"
  verify_smoke "$label" "$model" "$task" "$data_sha"
done
record_event "smoke_phase_passed" "accuracy was not a gate"

FULL_SPECS=(
  "stage3_gpt4o|$GPT_MODEL|pathmmu|$PATHMMU_TEST|0|$PATHMMU_TEST_SHA"
  "stage3_gpt4o|$GPT_MODEL|pathvqa|$PATHVQA|1|$PATHVQA_SHA"
  "stage3_gpt4o|$GPT_MODEL|omnimedvqa|$OMNI|2|$OMNI_SHA"
  "stage3_kimi26|$KIMI_MODEL|pathmmu|$PATHMMU_TEST|3|$PATHMMU_TEST_SHA"
  "stage3_kimi26|$KIMI_MODEL|pathvqa|$PATHVQA|4|$PATHVQA_SHA"
  "stage3_kimi26|$KIMI_MODEL|omnimedvqa|$OMNI|5|$OMNI_SHA"
)
record_event "full_phase_started" "six full Stage3 model-task evaluations"
pids=()
for spec in "${FULL_SPECS[@]}"; do
  IFS='|' read -r label model task data gpu data_sha <<<"$spec"
  run_full "$label" "$model" "$task" "$data" "$gpu" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || { record_event "full_phase_failed"; exit 4; }
for spec in "${FULL_SPECS[@]}"; do
  IFS='|' read -r label model task data gpu data_sha <<<"$spec"
  verify_full "$label" "$model" "$task" "$data_sha"
done
record_event "full_phase_verified" "all six count/hash/source gates passed"
echo "Stage3 selected-model full evaluations completed and verified: $RUN_ROOT"
