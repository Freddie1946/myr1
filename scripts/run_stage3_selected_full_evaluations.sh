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

EVAL_ARMS="${PATHVLM_STAGE3_EVAL_ARMS:-gpt4o,kimi26}"
RUN_GPT=0
RUN_KIMI=0
IFS=',' read -r -a REQUESTED_ARMS <<<"$EVAL_ARMS"
for arm in "${REQUESTED_ARMS[@]}"; do
  case "$arm" in
    gpt4o)
      (( RUN_GPT == 0 )) || { echo "Duplicate Stage3 evaluation arm: $arm" >&2; exit 2; }
      RUN_GPT=1
      ;;
    kimi26)
      (( RUN_KIMI == 0 )) || { echo "Duplicate Stage3 evaluation arm: $arm" >&2; exit 2; }
      RUN_KIMI=1
      ;;
    *) echo "Unsupported Stage3 evaluation arm: $arm" >&2; exit 2 ;;
  esac
done
(( RUN_GPT == 1 || RUN_KIMI == 1 )) || { echo "No Stage3 evaluation arm selected" >&2; exit 2; }

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

SHARED_GPU_MODE="${PATHVLM_ALLOW_SHARED_GPU_WITH_STAGE3:-false}"
SHARED_OWNER="${PATHVLM_SHARED_GPU_OWNER_RUN_DIR:-}"
SHARED_MAX_USED_MIB="${PATHVLM_SHARED_GPU_MAX_INITIAL_USED_MIB:-30000}"
SHARED_MIN_FREE_MIB="${PATHVLM_SHARED_GPU_MIN_FREE_MIB:-45000}"
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
fi

verify_shared_owner() {
  [[ "$SHARED_GPU_MODE" == "true" ]] || return 0
  local worker_count
  worker_count="$(
    { pgrep -af -- "$SHARED_OWNER/output" || true; } |
      { grep -F 'grpo_pathmmu.py' || true; } | wc -l
  )"
  (( worker_count >= 8 )) || {
    echo "Shared Stage3 owner has only $worker_count live workers" >&2; return 1;
  }
  grep -q 'rewards/audited_process_reward' "$SHARED_OWNER/train_segment00.log" || {
    echo "Shared Stage3 owner has not completed an optimizer step" >&2; return 1;
  }
}

for path in "$PYTHON" "$PATHMMU_RUNNER" "$EXTERNAL_RUNNER" "$SMOKE_VERIFY" \
  "$FULL_VERIFY" "$SNAPSHOT_VERIFY" \
  "$PATHMMU_VALID" "$PATHMMU_TEST" "$PATHVQA" "$OMNI"; do
  [[ -e "$path" ]] || { echo "Required evaluation input is missing: $path" >&2; exit 2; }
done
(( RUN_GPT == 0 )) || [[ -e "$GPT_SELECTION" ]] || {
  echo "Required GPT-4o selection is missing: $GPT_SELECTION" >&2; exit 2;
}
(( RUN_KIMI == 0 )) || [[ -e "$KIMI_SELECTION" ]] || {
  echo "Required Kimi selection is missing: $KIMI_SELECTION" >&2; exit 2;
}
[[ "$(sha256sum "$PATHMMU_VALID" | awk '{print $1}')" == "$PATHMMU_VALID_SHA" ]] || exit 2
[[ "$(sha256sum "$PATHMMU_TEST" | awk '{print $1}')" == "$PATHMMU_TEST_SHA" ]] || exit 2
[[ "$(sha256sum "$PATHVQA" | awk '{print $1}')" == "$PATHVQA_SHA" ]] || exit 2
[[ "$(sha256sum "$OMNI" | awk '{print $1}')" == "$OMNI_SHA" ]] || exit 2

resolve_selection() {
  local selection_name="$1" selection_path="$2" run_path="$3"
  "$PYTHON" - "$selection_name" "$selection_path" "$run_path" <<'PY'
import json
import sys
from pathlib import Path

selection_name = sys.argv[1]
path = Path(sys.argv[2])
run = Path(sys.argv[3]).resolve()
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
}

GPT_MODEL="" GPT_STEP="" KIMI_MODEL="" KIMI_STEP=""
if (( RUN_GPT == 1 )); then
  mapfile -t SELECTION_VALUES < <(resolve_selection gpt4o "$GPT_SELECTION" "$GPT_RUN")
  [[ "${#SELECTION_VALUES[@]}" -eq 2 ]] || exit 2
  GPT_MODEL="${SELECTION_VALUES[0]}"
  GPT_STEP="${SELECTION_VALUES[1]}"
  "$PYTHON" "$SNAPSHOT_VERIFY" "$GPT_MODEL" \
    --expected-step "$GPT_STEP" --expected-epoch "$((GPT_STEP / 500))" >/dev/null
fi
if (( RUN_KIMI == 1 )); then
  mapfile -t SELECTION_VALUES < <(resolve_selection kimi26 "$KIMI_SELECTION" "$KIMI_RUN")
  [[ "${#SELECTION_VALUES[@]}" -eq 2 ]] || exit 2
  KIMI_MODEL="${SELECTION_VALUES[0]}"
  KIMI_STEP="${SELECTION_VALUES[1]}"
  "$PYTHON" "$SNAPSHOT_VERIFY" "$KIMI_MODEL" \
    --expected-step "$KIMI_STEP" --expected-epoch "$((KIMI_STEP / 500))" >/dev/null
fi

parse_gpu_triplet() {
  local name="$1" value="$2"
  local -a ids
  IFS=',' read -r -a ids <<<"$value"
  [[ "${#ids[@]}" -eq 3 ]] || { echo "$name must contain exactly three GPU IDs" >&2; exit 2; }
  for id in "${ids[@]}"; do
    [[ "$id" =~ ^[0-7]$ ]] || { echo "Invalid GPU ID in $name: $id" >&2; exit 2; }
  done
  [[ "${ids[0]}" != "${ids[1]}" && "${ids[0]}" != "${ids[2]}" && "${ids[1]}" != "${ids[2]}" ]] || {
    echo "$name contains duplicate GPU IDs" >&2; exit 2;
  }
  printf '%s\n' "${ids[@]}"
}

GPT_GPUS=() KIMI_GPUS=() REQUIRED_GPUS=()
if (( RUN_GPT == 1 )); then
  mapfile -t GPT_GPUS < <(parse_gpu_triplet PATHVLM_STAGE3_GPT4O_EVAL_GPUS "${PATHVLM_STAGE3_GPT4O_EVAL_GPUS:-0,1,2}")
  [[ "${#GPT_GPUS[@]}" -eq 3 ]] || exit 2
  REQUIRED_GPUS+=("${GPT_GPUS[@]}")
fi
if (( RUN_KIMI == 1 )); then
  mapfile -t KIMI_GPUS < <(parse_gpu_triplet PATHVLM_STAGE3_KIMI_EVAL_GPUS "${PATHVLM_STAGE3_KIMI_EVAL_GPUS:-3,4,5}")
  [[ "${#KIMI_GPUS[@]}" -eq 3 ]] || exit 2
  REQUIRED_GPUS+=("${KIMI_GPUS[@]}")
fi
declare -A SEEN_GPUS=()
verify_shared_owner
for gpu in "${REQUIRED_GPUS[@]}"; do
  [[ -z "${SEEN_GPUS[$gpu]:-}" ]] || { echo "GPU $gpu is assigned to multiple evaluations" >&2; exit 2; }
  SEEN_GPUS[$gpu]=1
  IFS=',' read -r used total < <(
    nvidia-smi --id="$gpu" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
  )
  used="${used//[[:space:]]/}"
  total="${total//[[:space:]]/}"
  if [[ "$SHARED_GPU_MODE" == "true" ]]; then
    (( used <= SHARED_MAX_USED_MIB && total - used >= SHARED_MIN_FREE_MIB )) || {
      echo "GPU $gpu fails shared-memory admission: used=$used total=$total MiB" >&2; exit 2;
    }
  else
    (( used <= 10 )) || { echo "GPU $gpu is not idle: ${used} MiB" >&2; exit 2; }
  fi
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

"$PYTHON" - "$CONTRACT" "$EVAL_ARMS" "$GPT_MODEL" "$GPT_STEP" "$KIMI_MODEL" "$KIMI_STEP" \
  "$GIT_COMMIT" "$SHARED_GPU_MODE" "$SHARED_OWNER" "$SHARED_MAX_USED_MIB" "$SHARED_MIN_FREE_MIB" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
arms = sys.argv[2].split(",")
models = {}
if "gpt4o" in arms:
    models["stage3_gpt4o"] = {"path": sys.argv[3], "selected_step": int(sys.argv[4])}
if "kimi26" in arms:
    models["stage3_kimi26"] = {"path": sys.argv[5], "selected_step": int(sys.argv[6])}
value = {
    "schema_version": 1,
    "status": "frozen",
    "models": models,
    "tasks": {
        "pathmmu": {"count": 999, "split_role": "test999_development"},
        "pathvqa_yes_no": {"count": 3362, "split_role": "external_test"},
        "omnimedvqa": {"count": 8518, "split_role": "external_test"},
    },
    "pathvqa_free_form_status": "postponed_by_user_and_excluded_from_this_run",
    "smoke_count_per_model_task": 16,
    "accuracy_used_as_smoke_gate": False,
    "repository_commit": sys.argv[7],
    "gpu_admission": {
        "mode": "shared_with_stage3" if sys.argv[8] == "true" else "idle_exclusive",
        "stage3_owner_run": sys.argv[9] or None,
        "maximum_initial_used_mib": int(sys.argv[10]),
        "minimum_initial_free_mib": int(sys.argv[11]),
    },
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
    local scope=()
    [[ "$task" != "pathvqa" ]] || scope=(--pathvqa-answer-scope yes_no_only)
    "$PYTHON" "$EXTERNAL_RUNNER" --task "$task" --model "$model" \
      --backend qwen2_5_vl --data "$data" --output-dir "$output" \
      --split-role adapter_smoke --batch-size 8 --limit 16 "${scope[@]}" >"$log" 2>&1
  fi
}

verify_smoke() {
  local label="$1" model="$2" task="$3" data_sha="$4"
  local output="$RUN_ROOT/$label/${task}_smoke16"
  local scope=()
  [[ "$task" != "pathvqa" ]] || scope=(--expected-pathvqa-answer-scope yes_no_only)
  "$PYTHON" "$SMOKE_VERIFY" --task "$task" --metrics "$output/metrics.json" \
    --predictions "$output/predictions.jsonl" --expected-count 16 \
    --expected-data-sha256 "$data_sha" \
    --expected-model-config-sha256 "$(model_config_sha "$model")" \
    --minimum-nonempty-rate 0.80 --minimum-parseable-rate 0.80 \
    --maximum-cap-hit-rate 0.20 "${scope[@]}" \
    --output "$output/smoke_gate.json" >/dev/null
}

run_full() {
  local label="$1" model="$2" task="$3" data="$4" gpu="$5"
  local suffix count split output log
  if [[ "$task" == "pathmmu" ]]; then
    suffix="pathmmu_test999"; count=999; split="test999_development"
  elif [[ "$task" == "pathvqa" ]]; then
    suffix="pathvqa_yesno3362"; count=3362; split="external_test"
  elif [[ "$task" == "omnimedvqa" ]]; then
    suffix="omnimedvqa_full8518"; count=8518; split="external_test"
  else
    echo "Unsupported Stage3 evaluation task: $task" >&2
    return 2
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
    local scope=()
    [[ "$task" != "pathvqa" ]] || scope=(--pathvqa-answer-scope yes_no_only)
    "$PYTHON" "$EXTERNAL_RUNNER" --task "$task" --model "$model" \
      --backend qwen2_5_vl --data "$data" --output-dir "$output" \
      --split-role "$split" --batch-size 8 "${scope[@]}" "${resume[@]}" >"$log" 2>&1
  fi
}

verify_full() {
  local label="$1" model="$2" task="$3" data_sha="$4"
  local suffix count split output
  if [[ "$task" == "pathmmu" ]]; then
    suffix="pathmmu_test999"; count=999; split="test999_development"
  elif [[ "$task" == "pathvqa" ]]; then
    suffix="pathvqa_yesno3362"; count=3362; split="external_test"
  elif [[ "$task" == "omnimedvqa" ]]; then
    suffix="omnimedvqa_full8518"; count=8518; split="external_test"
  else
    echo "Unsupported Stage3 verification task: $task" >&2
    return 2
  fi
  output="$RUN_ROOT/$label/$suffix"
  local scope=()
  [[ "$task" != "pathvqa" ]] || scope=(--expected-pathvqa-answer-scope yes_no_only)
  "$PYTHON" "$FULL_VERIFY" --task "$task" --metrics "$output/metrics.json" \
    --predictions "$output/predictions.jsonl" --run-config "$output/run_config.json" \
    --expected-count "$count" --expected-split-role "$split" \
    --expected-data-sha256 "$data_sha" \
    --expected-model-config-sha256 "$(model_config_sha "$model")" "${scope[@]}" \
    --output "$output/full_integrity_verified.json" >/dev/null
}

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false PYTHONPATH="$REPO/scripts"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

SPECS=()
if (( RUN_GPT == 1 )); then
  SPECS+=(
    "stage3_gpt4o|$GPT_MODEL|pathmmu|$PATHMMU_VALID|${GPT_GPUS[0]}|$PATHMMU_VALID_SHA"
    "stage3_gpt4o|$GPT_MODEL|pathvqa|$PATHVQA|${GPT_GPUS[1]}|$PATHVQA_SHA"
    "stage3_gpt4o|$GPT_MODEL|omnimedvqa|$OMNI|${GPT_GPUS[2]}|$OMNI_SHA"
  )
fi
if (( RUN_KIMI == 1 )); then
  SPECS+=(
    "stage3_kimi26|$KIMI_MODEL|pathmmu|$PATHMMU_VALID|${KIMI_GPUS[0]}|$PATHMMU_VALID_SHA"
    "stage3_kimi26|$KIMI_MODEL|pathvqa|$PATHVQA|${KIMI_GPUS[1]}|$PATHVQA_SHA"
    "stage3_kimi26|$KIMI_MODEL|omnimedvqa|$OMNI|${KIMI_GPUS[2]}|$OMNI_SHA"
  )
fi

record_event "smoke_phase_started" "${#SPECS[@]} closed-question Stage3 model-task checks: $EVAL_ARMS"
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
verify_shared_owner || { record_event "shared_stage3_owner_failed_after_smoke"; exit 3; }

FULL_SPECS=()
for spec in "${SPECS[@]}"; do
  IFS='|' read -r label model task _ gpu _ <<<"$spec"
  if [[ "$task" == "pathmmu" ]]; then
    FULL_SPECS+=("$label|$model|$task|$PATHMMU_TEST|$gpu|$PATHMMU_TEST_SHA")
  elif [[ "$task" == "pathvqa" ]]; then
    FULL_SPECS+=("$label|$model|$task|$PATHVQA|$gpu|$PATHVQA_SHA")
  else
    FULL_SPECS+=("$label|$model|$task|$OMNI|$gpu|$OMNI_SHA")
  fi
done
record_event "full_phase_started" "${#FULL_SPECS[@]} full closed-question Stage3 model-task evaluations: $EVAL_ARMS"
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
record_event "full_phase_verified" "all ${#FULL_SPECS[@]} count/hash/source gates passed: $EVAL_ARMS"
"$PYTHON" - "$RUN_ROOT/completed.json" "$EVENTS" "$EVAL_ARMS" <<'PY'
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
root = path.parent
artifacts = []
labels = {"gpt4o": "stage3_gpt4o", "kimi26": "stage3_kimi26"}
for arm in sys.argv[3].split(","):
    label = labels[arm]
    for suffix in ("pathmmu_test999", "pathvqa_yesno3362", "omnimedvqa_full8518"):
        value = root / label / suffix / "full_integrity_verified.json"
        if not value.is_file():
            raise SystemExit(f"missing full-integrity artifact: {value}")
        artifacts.append({
            "path": str(value.relative_to(root)),
            "sha256": hashlib.sha256(value.read_bytes()).hexdigest(),
        })
result = {
    "schema_version": 1,
    "status": "completed",
    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "pathvqa_scope": "yes_no_only",
    "pathvqa_free_form_inference": False,
    "artifacts": artifacts,
    "events": str(Path(sys.argv[2]).resolve()),
}
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
echo "Stage3 selected-model full evaluations completed and verified: $RUN_ROOT"
