#!/usr/bin/env bash
set -euo pipefail
umask 077

WORKSPACE_ROOT="/home/dataset-assist-0/czy/wjy"
REPO_ROOT="$WORKSPACE_ROOT/myr1"
INSTALL_ROOT="$WORKSPACE_ROOT/pathvlm_r1_v1_a100"
EVAL_ROOT="$WORKSPACE_ROOT/pathvlm_revision_eval_a100"
RUN_ROOT="$EVAL_ROOT/runs/local_gpu_baselines_20260803"
SEQUENCE_ROOT="$RUN_ROOT/sequence"
EVENTS="$SEQUENCE_ROOT/events.jsonl"
LOCK="$SEQUENCE_ROOT/sequence.lock"
FULL_VERIFY="$REPO_ROOT/scripts/verify_local_baseline_full.py"
SMOKE_VERIFY="$REPO_ROOT/scripts/verify_local_baseline_smoke.py"

LLAMA_PY="$EVAL_ROOT/envs/llama32_vision/bin/python"
LLAMA90="$INSTALL_ROOT/models/Llama-3.2-90B-Vision-Instruct-modelscope-master"
LLAMA90_CONFIG_SHA="367178a632da6fe82d83b659ac9cfad70a858e38b4aa5e6883acf34e81df8d80"
LLAMA11_CONFIG_SHA="db708a72a246253fdf8aa3837a5f0b80ac1960692495ce05cb3101833128e845"
DEEPSEEK_CONFIG_SHA="d5dadaf4d41af00d569ac376bf09de4e28324c853b3ae0c7f6fed66f19fdcf8c"

PATHMMU_VALID="$INSTALL_ROOT/data/pathmmu_image_disjoint_v2/rewritten_records/validation_0385.json"
PATHMMU_VALID_SHA="f52c9a412da579db8fb8bf52f6fa320e7c4ae9cb42b6cf1bdf62f77ab158dcf0"
PATHMMU_TEST="$INSTALL_ROOT/data/pathmmu_image_disjoint_v2/rewritten_records/test_0999.json"
PATHMMU_TEST_SHA="3420c8ff2f1642801a5ecebd21271f4e7de5877d6459814473ddbcdda996b549"
PATHVQA="$EVAL_ROOT/datasets/external_vqa_contract_v1_20260729/pathvqa_test_6719.json"
PATHVQA_SHA="5fa7319784dd47c81027c2b99cbb0d0d472638d0dc5c5169dcbd5d43987784be"
OMNI="$EVAL_ROOT/datasets/external_vqa_contract_v1_20260729/omnimedvqa_four_sources_8518.json"
OMNI_SHA="04ba0790410cd5de4c95689d349574b102034a153afa00d952e351d3c13991a7"

KIMI_RUN="$INSTALL_ROOT/runs/stage3_process_grpo/kimi26_full3epoch_seed42_fresh_20260803"
KIMI_BUDGET="26.71782984"
KIMI_SESSION="stage3_kimi26_fresh_20260803"
KIMI_SMOKE="$REPO_ROOT/scripts/run_stage3_kimi26_formal_smoke.py"
KIMI_SUPERVISOR="$REPO_ROOT/scripts/supervise_stage3_kimi26_formal.sh"
SECRET_FILE="$WORKSPACE_ROOT/.secrets/openrouter.env"

mkdir -p "$SEQUENCE_ROOT"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Another local-baseline/Kimi sequence is already running" >&2
  exit 2
fi

record_event() {
  local event="$1"
  local detail="${2:-}"
  python3 - "$EVENTS" "$event" "$detail" <<'PY'
import datetime
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "event": sys.argv[2],
    "detail": sys.argv[3],
}
line = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
try:
    os.write(fd, line)
    os.fsync(fd)
finally:
    os.close(fd)
PY
}

verify_full() {
  local task="$1"
  local output_dir="$2"
  local count="$3"
  local split_role="$4"
  local data_sha="$5"
  local model_sha="$6"
  python3 "$FULL_VERIFY" \
    --task "$task" \
    --metrics "$output_dir/metrics.json" \
    --predictions "$output_dir/predictions.jsonl" \
    --run-config "$output_dir/run_config.json" \
    --expected-count "$count" \
    --expected-split-role "$split_role" \
    --expected-data-sha256 "$data_sha" \
    --expected-model-config-sha256 "$model_sha" \
    --output "$output_dir/full_integrity_verified.json" >/dev/null
}

wait_for_current_full_runs() {
  local all_complete output_dir task count split_role data_sha model_sha
  while true; do
    all_complete=true
    while IFS='|' read -r task output_dir count split_role data_sha model_sha; do
      if [[ -f "$output_dir/metrics.json" ]]; then
        verify_full "$task" "$output_dir" "$count" "$split_role" "$data_sha" "$model_sha"
      else
        all_complete=false
        if ! pgrep -f "python.*${output_dir}" >/dev/null; then
          record_event "current_full_run_missing" "$output_dir"
          echo "Current full run stopped without metrics: $output_dir" >&2
          exit 3
        fi
      fi
    done <<EOF
pathmmu|$RUN_ROOT/llama32_11b/pathmmu_test999|999|test999_development|$PATHMMU_TEST_SHA|$LLAMA11_CONFIG_SHA
omnimedvqa|$RUN_ROOT/llama32_11b/omnimedvqa_full8518|8518|external_test|$OMNI_SHA|$LLAMA11_CONFIG_SHA
pathvqa|$RUN_ROOT/deepseek_vl2/pathvqa_full6719|6719|external_test|$PATHVQA_SHA|$DEEPSEEK_CONFIG_SHA
omnimedvqa|$RUN_ROOT/deepseek_vl2/omnimedvqa_full8518|8518|external_test|$OMNI_SHA|$DEEPSEEK_CONFIG_SHA
EOF
    if [[ "$all_complete" == true ]]; then
      record_event "current_full_runs_verified" "four completed artifacts"
      return
    fi
    sleep 60
  done
}

wait_for_idle_gpus() {
  local used busy
  while true; do
    busy=false
    while IFS= read -r used; do
      used="${used//[[:space:]]/}"
      if (( used > 10 )); then
        busy=true
      fi
    done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [[ "$busy" == false ]]; then
      return
    fi
    sleep 30
  done
}

verify_smoke() {
  local task="$1"
  local output_dir="$2"
  local data_sha="$3"
  python3 "$SMOKE_VERIFY" \
    --task "$task" \
    --metrics "$output_dir/metrics.json" \
    --predictions "$output_dir/predictions.jsonl" \
    --expected-count 16 \
    --expected-data-sha256 "$data_sha" \
    --expected-model-config-sha256 "$LLAMA90_CONFIG_SHA" \
    --minimum-nonempty-rate 0.80 \
    --minimum-parseable-rate 0.80 \
    --maximum-cap-hit-rate 0.20 \
    --output "$output_dir/smoke_gate.json" >/dev/null
}

run_llama90_smoke() {
  local task="$1"
  local data="$2"
  local data_sha="$3"
  local output_dir="$RUN_ROOT/llama32_90b/${task}_smoke16"
  local log="$RUN_ROOT/llama32_90b/${task}_smoke16.log"
  mkdir -p "$RUN_ROOT/llama32_90b"
  if [[ -f "$output_dir/metrics.json" ]]; then
    verify_smoke "$task" "$output_dir" "$data_sha"
    return 0
  fi
  if [[ -e "$output_dir" ]]; then
    record_event "llama90_smoke_incomplete" "$output_dir"
    return 1
  fi
  wait_for_idle_gpus
  record_event "llama90_smoke_started" "$task"
  export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
  export TOKENIZERS_PARALLELISM=false PYTHONPATH="$REPO_ROOT/scripts"
  if [[ "$task" == "pathmmu" ]]; then
    "$LLAMA_PY" "$REPO_ROOT/scripts/run_pathmmu_qwen_diagnostic.py" \
      --model "$LLAMA90" --backend mllama --data "$data" \
      --output-dir "$output_dir" --split-role validation_smoke \
      --batch-size 8 --limit 16 >"$log" 2>&1 || return 1
  else
    "$LLAMA_PY" "$REPO_ROOT/scripts/run_external_vqa_qwen.py" \
      --task "$task" --model "$LLAMA90" --backend mllama --data "$data" \
      --output-dir "$output_dir" --split-role adapter_smoke \
      --batch-size 8 --limit 16 >"$log" 2>&1 || return 1
  fi
  verify_smoke "$task" "$output_dir" "$data_sha"
}

run_llama90_full() {
  local task="$1"
  local data="$2"
  local data_sha="$3"
  local count="$4"
  local split_role="$5"
  local output_dir="$RUN_ROOT/llama32_90b/${task}_full${count}"
  local log="$RUN_ROOT/llama32_90b/${task}_full${count}.log"
  if [[ "$task" == "pathmmu" ]]; then
    output_dir="$RUN_ROOT/llama32_90b/pathmmu_test999"
    log="$RUN_ROOT/llama32_90b/pathmmu_test999.log"
  fi
  if [[ -f "$output_dir/metrics.json" ]]; then
    verify_full "$task" "$output_dir" "$count" "$split_role" "$data_sha" "$LLAMA90_CONFIG_SHA"
    return
  fi
  if [[ -e "$output_dir" ]]; then
    record_event "llama90_full_incomplete" "$output_dir"
    echo "Refusing to overwrite or blindly resume incomplete 90B run: $output_dir" >&2
    exit 3
  fi
  wait_for_idle_gpus
  record_event "llama90_full_started" "$task"
  export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
  export TOKENIZERS_PARALLELISM=false PYTHONPATH="$REPO_ROOT/scripts"
  if [[ "$task" == "pathmmu" ]]; then
    "$LLAMA_PY" "$REPO_ROOT/scripts/run_pathmmu_qwen_diagnostic.py" \
      --model "$LLAMA90" --backend mllama --data "$data" \
      --output-dir "$output_dir" --split-role "$split_role" \
      --batch-size 8 >"$log" 2>&1
  else
    "$LLAMA_PY" "$REPO_ROOT/scripts/run_external_vqa_qwen.py" \
      --task "$task" --model "$LLAMA90" --backend mllama --data "$data" \
      --output-dir "$output_dir" --split-role "$split_role" \
      --batch-size 8 >"$log" 2>&1
  fi
  verify_full "$task" "$output_dir" "$count" "$split_role" "$data_sha" "$LLAMA90_CONFIG_SHA"
  record_event "llama90_full_verified" "$task"
}

launch_kimi() {
  wait_for_idle_gpus
  if [[ ! -f "$SECRET_FILE" || "$(stat -c '%a' "$SECRET_FILE")" != "600" ]]; then
    echo "OpenRouter secret file is missing or not mode 0600" >&2
    exit 4
  fi
  if [[ ! -f "$KIMI_RUN/kimi26_formal_contract_smoke_passed.json" ]]; then
    if [[ -e "$KIMI_RUN" ]]; then
      echo "Fresh Kimi run directory exists without a smoke marker: $KIMI_RUN" >&2
      exit 4
    fi
    set -a
    source "$SECRET_FILE"
    set +a
    "$INSTALL_ROOT/envs/grpo/bin/python" "$KIMI_SMOKE" \
      --run-dir "$KIMI_RUN" --budget-limit-usd "$KIMI_BUDGET" \
      >"$SEQUENCE_ROOT/kimi_formal_smoke.log" 2>&1
    record_event "kimi_formal_smoke_passed" "$KIMI_RUN"
  fi
  if tmux has-session -t "$KIMI_SESSION" 2>/dev/null; then
    echo "Kimi tmux session already exists: $KIMI_SESSION" >&2
    exit 4
  fi
  tmux new-session -d -s "$KIMI_SESSION" \
    "env PATHVLM_STAGE3_RUN_DIR='$KIMI_RUN' PATHVLM_STAGE3_FORMAL_BUDGET_USD='$KIMI_BUDGET' PATHVLM_STAGE3_SAVE_STEPS=100 PATHVLM_STAGE3_MAX_RECOVERIES=3 PATHVLM_STAGE3_MASTER_PORT=29641 bash '$KIMI_SUPERVISOR' >'$KIMI_RUN/supervisor.log' 2>&1"
  local attempts=0
  while (( attempts < 60 )); do
    if [[ -f "$KIMI_RUN/supervisor_recovery_audit.jsonl" ]] && \
       grep -qE '"event"[[:space:]]*:[[:space:]]*"segment_start"' "$KIMI_RUN/supervisor_recovery_audit.jsonl" && \
       tmux has-session -t "$KIMI_SESSION" 2>/dev/null; then
      record_event "kimi_stage3_launched" "$KIMI_RUN"
      python3 - "$SEQUENCE_ROOT/completed.json" "$KIMI_RUN" "$KIMI_SESSION" <<'PY'
import datetime
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "completed_after_kimi_launch_verified",
    "kimi_run_dir": sys.argv[2],
    "kimi_tmux_session": sys.argv[3],
    "continued_monitoring_required": False,
}
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.chmod(temporary, 0o600)
os.replace(temporary, path)
PY
      return
    fi
    if ! tmux has-session -t "$KIMI_SESSION" 2>/dev/null; then
      echo "Kimi supervisor exited before launch verification" >&2
      exit 4
    fi
    attempts=$((attempts + 1))
    sleep 5
  done
  echo "Kimi supervisor did not reach a verified segment start within five minutes" >&2
  exit 4
}

record_event "sequence_started" "wait current full runs, Llama90 gates/full, Kimi last"
wait_for_current_full_runs
wait_for_idle_gpus

declare -A LLAMA90_GATE=()
if run_llama90_smoke pathmmu "$PATHMMU_VALID" "$PATHMMU_VALID_SHA"; then
  LLAMA90_GATE[pathmmu]=pass
  record_event "llama90_smoke_passed" "pathmmu"
else
  LLAMA90_GATE[pathmmu]=fail
  record_event "llama90_smoke_failed" "pathmmu"
fi
wait_for_idle_gpus
if run_llama90_smoke pathvqa "$PATHVQA" "$PATHVQA_SHA"; then
  LLAMA90_GATE[pathvqa]=pass
  record_event "llama90_smoke_passed" "pathvqa"
else
  LLAMA90_GATE[pathvqa]=fail
  record_event "llama90_smoke_failed" "pathvqa"
fi
wait_for_idle_gpus
if run_llama90_smoke omnimedvqa "$OMNI" "$OMNI_SHA"; then
  LLAMA90_GATE[omnimedvqa]=pass
  record_event "llama90_smoke_passed" "omnimedvqa"
else
  LLAMA90_GATE[omnimedvqa]=fail
  record_event "llama90_smoke_failed" "omnimedvqa"
fi

if [[ "${LLAMA90_GATE[pathmmu]}" == pass ]]; then
  run_llama90_full pathmmu "$PATHMMU_TEST" "$PATHMMU_TEST_SHA" 999 test999_development
fi
if [[ "${LLAMA90_GATE[pathvqa]}" == pass ]]; then
  run_llama90_full pathvqa "$PATHVQA" "$PATHVQA_SHA" 6719 external_test
fi
if [[ "${LLAMA90_GATE[omnimedvqa]}" == pass ]]; then
  run_llama90_full omnimedvqa "$OMNI" "$OMNI_SHA" 8518 external_test
fi

record_event "llama90_eligible_full_runs_finished" "all passed tasks verified"
launch_kimi
record_event "sequence_completed" "Kimi launched; no continued monitoring"
