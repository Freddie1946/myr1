#!/usr/bin/env bash
set -euo pipefail
umask 077

WORK_ROOT=/home/dataset-assist-0/czy/wjy
REPO=$WORK_ROOT/myr1
INSTALL=$WORK_ROOT/pathvlm_r1_v1_a100
EVAL=$WORK_ROOT/pathvlm_revision_eval_a100
PYTHON=$INSTALL/envs/sft/bin/python
RUNNER=$REPO/scripts/run_external_vqa_qwen.py
VERIFY=$REPO/scripts/verify_omnimed_corrective_v3_smoke.py
DATA=$EVAL/datasets/external_vqa_contract_v1_20260729/omnimedvqa_four_sources_8518.json
DATA_SHA=04ba0790410cd5de4c95689d349574b102034a153afa00d952e351d3c13991a7
RUN_ROOT=$EVAL/runs/omnimedvqa_corrective_v3_192_20260810
GPU=${OMNIMED_CORRECTIVE_GPU:-7}
MIN_FREE_MIB=${OMNIMED_CORRECTIVE_MIN_FREE_MIB:-34000}
MODE=${1:-smoke}

[[ "$MODE" == smoke || "$MODE" == full ]] || {
  echo "Usage: $0 [smoke|full]" >&2
  exit 2
}
[[ "$GPU" =~ ^[0-7]$ ]] || { echo "invalid GPU: $GPU" >&2; exit 2; }
[[ "$MIN_FREE_MIB" =~ ^[1-9][0-9]*$ ]] || { echo "invalid memory threshold" >&2; exit 2; }

GPT_MODEL=$INSTALL/runs/stage3_process_grpo/gpt4o_full3epoch_seed42_20260801/epoch_model_snapshots/checkpoint-1000
GPT_SHA=677754b792532590407a989718dd68eed61b46d0846db56ad67a57a67838d189
GROK_MODEL=$INSTALL/runs/stage3_process_grpo/grok43_full3epoch_seed42_20260805_attempt02/epoch_model_snapshots/checkpoint-1500
GROK_SHA=dc3e7c414bb7eddb19ce06a71b3819dd355f57bad1fb7a19f86fa09cebb8f9e0

for path in "$PYTHON" "$RUNNER" "$VERIFY" "$DATA" "$GPT_MODEL/config.json" "$GROK_MODEL/config.json"; do
  [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 2; }
done
[[ "$(sha256sum "$DATA" | awk '{print $1}')" == "$DATA_SHA" ]] || exit 2
[[ "$(sha256sum "$GPT_MODEL/config.json" | awk '{print $1}')" == "$GPT_SHA" ]] || exit 2
[[ "$(sha256sum "$GROK_MODEL/config.json" | awk '{print $1}')" == "$GROK_SHA" ]] || exit 2

mkdir -p "$RUN_ROOT/logs"
EVENTS=$RUN_ROOT/events.jsonl

record_event() {
  "$PYTHON" - "$EVENTS" "$1" "$2" <<'PY'
import datetime, json, os, sys
from pathlib import Path
path = Path(sys.argv[1])
row = {"recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "event": sys.argv[2], "detail": sys.argv[3]}
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
PY
}

admit_gpu() {
  local free_mib
  free_mib=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
  free_mib=${free_mib//[[:space:]]/}
  (( free_mib >= MIN_FREE_MIB )) || {
    record_event admission_failed "gpu=$GPU free_mib=$free_mib required=$MIN_FREE_MIB"
    echo "GPU $GPU has only $free_mib MiB free; refusing evaluator launch" >&2
    return 1
  }
  local controller workers
  controller=$(pgrep -fc 'run_data_ratio_ablation_optimized_tail.py --mode formal' || true)
  workers=$(pgrep -fc 'grpo_pathmmu.py.*data_ratio_rule_rl_ablation_v1' || true)
  (( controller >= 1 && workers >= 8 )) || {
    record_event training_owner_missing "controller=$controller workers=$workers"
    echo "data-ratio training owner is not healthy" >&2
    return 1
  }
  record_event admission_passed "gpu=$GPU free_mib=$free_mib controller=$controller workers=$workers"
}

run_model() {
  local name=$1 model=$2 model_sha=$3 output log
  if [[ "$MODE" == smoke ]]; then
    output=$RUN_ROOT/$name/smoke16
    log=$RUN_ROOT/logs/${name}_smoke16.log
    [[ ! -e "$output" ]] || { echo "smoke output already exists: $output" >&2; return 2; }
  else
    output=$RUN_ROOT/$name/full8518
    log=$RUN_ROOT/logs/${name}_full8518.log
    [[ -f "$RUN_ROOT/$name/smoke16/smoke_verification.json" ]] || {
      echo "missing passed smoke for $name" >&2; return 2;
    }
  fi
  admit_gpu
  record_event "${MODE}_started" "model=$name output=$output gpu=$GPU"
  local -a resume=() extra=()
  if [[ "$MODE" == smoke ]]; then
    extra=(--limit 16)
  elif [[ -e "$output" ]]; then
    resume=(--resume)
  fi
  CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false PYTHONPATH=$REPO/scripts \
    "$PYTHON" "$RUNNER" \
      --task omnimedvqa \
      --model "$model" \
      --backend qwen2_5_vl \
      --data "$DATA" \
      --output-dir "$output" \
      --split-role "$([[ "$MODE" == smoke ]] && echo adapter_smoke || echo external_test)" \
      --max-new-tokens 192 \
      --generation-contract omnimed_corrective_v3_192 \
      --batch-size 1 \
      "${extra[@]}" "${resume[@]}" >"$log" 2>&1
  if [[ "$MODE" == smoke ]]; then
    "$PYTHON" "$VERIFY" \
      --metrics "$output/metrics.json" \
      --predictions "$output/predictions.jsonl" \
      --expected-data-sha256 "$DATA_SHA" \
      --expected-model-config-sha256 "$model_sha" \
      --output "$output/smoke_verification.json" >>"$log" 2>&1
  fi
  record_event "${MODE}_completed" "model=$name output=$output"
}

record_event sequence_started "mode=$MODE gpu=$GPU"
run_model stage3_gpt4o_checkpoint1000 "$GPT_MODEL" "$GPT_SHA"
run_model stage3_grok43_checkpoint1500 "$GROK_MODEL" "$GROK_SHA"
record_event sequence_completed "mode=$MODE gpu=$GPU"
