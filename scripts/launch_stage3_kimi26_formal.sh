#!/usr/bin/env bash
set -euo pipefail
umask 077

WORKSPACE_ROOT="/home/dataset-assist-0/czy/wjy"
REPO_ROOT="$WORKSPACE_ROOT/myr1"
INSTALL_ROOT="$WORKSPACE_ROOT/pathvlm_r1_v1_a100"
PYTHON="$INSTALL_ROOT/envs/grpo/bin/python"
PARENT="$INSTALL_ROOT/transferred_checkpoints/outcome_grpo_n1000_seed42_epoch02_step1000"
DATASET="$INSTALL_ROOT/data/pathmmu_image_disjoint_v2/grpo/pathvlm_rl_n1000.yaml"
IMAGE_HASH_MANIFEST="$REPO_ROOT/data/pathmmu_image_disjoint_v2/image_content_sha256.json"
SECRET_FILE="$WORKSPACE_ROOT/.secrets/openrouter.env"
RUN_DIR="${PATHVLM_STAGE3_RUN_DIR:-$INSTALL_ROOT/runs/stage3_process_grpo/kimi26_full3epoch_seed42_20260730}"
PARENT_ALIAS="$RUN_DIR/parent_Qwen2.5-VL-Stage2-epoch02-step1000"
OUTPUT_DIR="$RUN_DIR/output"
JUDGE_ROOT="$RUN_DIR/judge"
EPOCH_SNAPSHOT_DIR="$RUN_DIR/epoch_model_snapshots"
SMOKE_MARKER="$RUN_DIR/kimi26_formal_contract_smoke_passed.json"
MASTER_PORT="${PATHVLM_STAGE3_MASTER_PORT:-29641}"
SAVE_STEPS="${PATHVLM_STAGE3_SAVE_STEPS:-100}"
SEGMENT_ID="${PATHVLM_STAGE3_SEGMENT_ID:-segment00}"
RESUME_FROM="${PATHVLM_STAGE3_RESUME_FROM_CHECKPOINT:-}"
REWARD_LOG_DIR="$RUN_DIR/reward_audit/$SEGMENT_ID"
TRAIN_LOG="$RUN_DIR/train_${SEGMENT_ID}.log"
# The paid smoke marker retains the original contract. Runtime caps may only be
# raised through an audited on-disk ledger amendment before this launcher runs.
MAX_UNIQUE_REQUESTS="${PATHVLM_STAGE3_MAX_UNIQUE_REQUESTS:-12976}"
FALLBACK_TOTAL_LIMIT="${PATHVLM_STAGE3_RULE_FALLBACK_TOTAL_LIMIT:-24}"
FALLBACK_CONSECUTIVE_LIMIT="${PATHVLM_STAGE3_RULE_FALLBACK_CONSECUTIVE_LIMIT:-4}"
ALLOW_SHARED_GPUS="${PATHVLM_STAGE3_ALLOW_SHARED_GPUS:-false}"
SHARED_GPU_MIN_FREE_MIB="${PATHVLM_STAGE3_SHARED_GPU_MIN_FREE_MIB:-30720}"
MIN_REQUEST_INTERVAL_SECONDS="${PATHVLM_STAGE3_MIN_REQUEST_INTERVAL_SECONDS:-4}"

: "${PATHVLM_STAGE3_FORMAL_BUDGET_USD:?Set the separately approved Kimi formal-arm budget}"

if [[ ! "$PATHVLM_STAGE3_FORMAL_BUDGET_USD" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "PATHVLM_STAGE3_FORMAL_BUDGET_USD must be a positive number" >&2
  exit 2
fi
if [[ ! "$MAX_UNIQUE_REQUESTS" =~ ^[1-9][0-9]*$ ]] || (( MAX_UNIQUE_REQUESTS < 12976 )); then
  echo "PATHVLM_STAGE3_MAX_UNIQUE_REQUESTS must be at least 12976" >&2
  exit 2
fi
if [[ ! "$FALLBACK_TOTAL_LIMIT" =~ ^[1-9][0-9]*$ ]] || \
   [[ ! "$FALLBACK_CONSECUTIVE_LIMIT" =~ ^[1-9][0-9]*$ ]] || \
   (( FALLBACK_CONSECUTIVE_LIMIT > FALLBACK_TOTAL_LIMIT )); then
  echo "Invalid Stage3 fallback limits" >&2
  exit 2
fi
if [[ "$ALLOW_SHARED_GPUS" != "true" && "$ALLOW_SHARED_GPUS" != "false" ]]; then
  echo "PATHVLM_STAGE3_ALLOW_SHARED_GPUS must be true or false" >&2
  exit 2
fi
if [[ ! "$SHARED_GPU_MIN_FREE_MIB" =~ ^[1-9][0-9]*$ ]]; then
  echo "PATHVLM_STAGE3_SHARED_GPU_MIN_FREE_MIB must be a positive integer" >&2
  exit 2
fi
if [[ ! "$MIN_REQUEST_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
   (( MIN_REQUEST_INTERVAL_SECONDS < 4 )); then
  echo "PATHVLM_STAGE3_MIN_REQUEST_INTERVAL_SECONDS must be an integer of at least 4" >&2
  exit 2
fi
if [[ ! "$SAVE_STEPS" =~ ^[1-9][0-9]*$ ]] || (( 500 % SAVE_STEPS != 0 )); then
  echo "PATHVLM_STAGE3_SAVE_STEPS must be a positive divisor of 500" >&2
  exit 2
fi
if [[ ! "$SEGMENT_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "PATHVLM_STAGE3_SEGMENT_ID contains unsafe characters: $SEGMENT_ID" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "GRPO Python is missing: $PYTHON" >&2
  exit 2
fi
if [[ ! -d "$PARENT" || ! -f "$PARENT/model.safetensors.index.json" ]]; then
  echo "Stage2 parent snapshot is incomplete: $PARENT" >&2
  exit 2
fi
if [[ ! -f "$DATASET" || ! -f "$IMAGE_HASH_MANIFEST" ]]; then
  echo "Frozen RL1000 dataset or image hash manifest is missing" >&2
  exit 2
fi
if [[ ! -f "$SECRET_FILE" || "$(stat -c '%a' "$SECRET_FILE")" != "600" ]]; then
  echo "OpenRouter secret file must exist with mode 0600: $SECRET_FILE" >&2
  exit 2
fi
if [[ ! -f "$SMOKE_MARKER" ]]; then
  echo "Formal Kimi contract smoke marker is missing: $SMOKE_MARKER" >&2
  exit 2
fi
if [[ -n "$RESUME_FROM" ]]; then
  if [[ ! -d "$OUTPUT_DIR" || ! -d "$RESUME_FROM" ]]; then
    echo "Resume requires an existing output directory and checkpoint" >&2
    exit 2
  fi
  resolved_output="$(readlink -f "$OUTPUT_DIR")"
  resolved_resume="$(readlink -f "$RESUME_FROM")"
  if [[ "$(dirname "$resolved_resume")" != "$resolved_output" ]]; then
    echo "Resume checkpoint must be an immediate child of $resolved_output" >&2
    exit 2
  fi
  "$PYTHON" "$REPO_ROOT/scripts/stage3_checkpoint_recovery.py" \
    validate "$resolved_resume" --world-size 8 >/dev/null
  RESUME_FROM="$resolved_resume"
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1"
elif [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite existing Stage3 output without a validated resume: $OUTPUT_DIR" >&2
  exit 2
else
  unset TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD || true
fi

"$PYTHON" - "$DATASET" "$IMAGE_HASH_MANIFEST" "$SMOKE_MARKER" \
  "$MASTER_PORT" "$PATHVLM_STAGE3_FORMAL_BUDGET_USD" "$MAX_UNIQUE_REQUESTS" \
  "$JUDGE_ROOT" "$FALLBACK_TOTAL_LIMIT" "$FALLBACK_CONSECUTIVE_LIMIT" <<'PY'
import hashlib
import json
import socket
import sys
from pathlib import Path

dataset_yaml = Path(sys.argv[1])
hash_manifest_path = Path(sys.argv[2])
smoke_marker_path = Path(sys.argv[3])
port = int(sys.argv[4])
budget_limit = float(sys.argv[5])
max_unique_requests = int(sys.argv[6])
judge_root = Path(sys.argv[7])
fallback_total_limit = int(sys.argv[8])
fallback_consecutive_limit = int(sys.argv[9])

if max_unique_requests < 12976:
    raise SystemExit(
        f"Kimi recovery request cap is below the approved prior cap: {max_unique_requests}"
    )

yaml_text = dataset_yaml.read_text(encoding="utf-8")
json_lines = [
    line.split(":", 1)[1].strip()
    for line in yaml_text.splitlines()
    if line.strip().startswith("- json_path:")
]
if len(json_lines) != 1:
    raise SystemExit("RL1000 YAML must identify exactly one JSON dataset")
records_path = Path(json_lines[0])
records = json.loads(records_path.read_text(encoding="utf-8"))
hashes = json.loads(hash_manifest_path.read_text(encoding="utf-8")).get("images")
if not isinstance(records, list) or len(records) != 1000:
    raise SystemExit(f"RL1000 adapter count mismatch: {len(records)}")
if not isinstance(hashes, dict) or not hashes:
    raise SystemExit("image hash manifest is empty")
for index, row in enumerate(records):
    image = Path(row["image"])
    expected = hashes.get(image.name)
    if not image.is_file() or not expected:
        raise SystemExit(f"missing image or frozen hash at RL row {index}: {image}")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(f"image hash mismatch at RL row {index}: {image}")

marker = json.loads(smoke_marker_path.read_text(encoding="utf-8"))
expected_marker = {
    "status": "passed",
    "run_class": "complete_contemporary_kimi_stage3_arm",
    "formal_result": False,
    "model": "moonshotai/kimi-k2.6",
    "provider": "inceptron",
    "zdr": True,
    "data_collection": "deny",
    "distillable_enforced": True,
    "non_training": True,
    "max_judge_tokens": 1024,
    "evidence_target_characters": 160,
    "evidence_max_characters": 512,
    "reasoning_enabled": False,
    "minimum_request_interval_seconds": 4,
    "retry_delays_seconds": [15, 45, 90],
    "cache_reuse_across_training_segments": False,
    "rule_fallback_enabled": True,
    "rule_fallback_total_limit": 24,
    "rule_fallback_consecutive_limit": 4,
    "rule_fallback_max_reward": 0.5,
    "budget_limit_usd": 26.71782984,
    "max_unique_requests": 12361,
    "maximum_physical_attempts_per_logical_judgment": 4,
    "unresolved_reservations": 0,
}
mismatch = {
    key: {"expected": expected, "actual": marker.get(key)}
    for key, expected in expected_marker.items()
    if marker.get(key) != expected
}
if mismatch:
    raise SystemExit(f"Kimi formal smoke marker mismatch: {mismatch}")

budget_ledger_path = judge_root / "budget_ledger.json"
if budget_ledger_path.is_file():
    ledger = json.loads(budget_ledger_path.read_text(encoding="utf-8"))
    runtime_contract = {
        "limit_usd": budget_limit,
        "reserve_usd": 0.05,
        "max_unique_requests": max_unique_requests,
    }
    ledger_mismatch = {
        key: {"expected": expected, "actual": ledger.get(key)}
        for key, expected in runtime_contract.items()
        if ledger.get(key) != expected
    }
    if ledger_mismatch:
        raise SystemExit(f"Kimi runtime budget ledger mismatch: {ledger_mismatch}")
    if ledger.get("reservations"):
        raise SystemExit("Kimi runtime budget ledger has unresolved reservations")
    if float(ledger.get("committed_spend_usd", -1)) > budget_limit:
        raise SystemExit("Kimi runtime committed spend exceeds its budget limit")

fallback_ledger_path = judge_root / "rule_fallback_ledger.json"
if fallback_ledger_path.is_file():
    fallback = json.loads(fallback_ledger_path.read_text(encoding="utf-8"))
    fallback_contract = {
        "total_limit": fallback_total_limit,
        "consecutive_limit": fallback_consecutive_limit,
    }
    fallback_mismatch = {
        key: {"expected": expected, "actual": fallback.get(key)}
        for key, expected in fallback_contract.items()
        if fallback.get(key) != expected
    }
    if fallback_mismatch:
        raise SystemExit(f"Kimi fallback ledger mismatch: {fallback_mismatch}")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as exc:
        raise SystemExit(f"master port {port} is unavailable: {exc}") from exc
print("Full Kimi Stage3 static preflight passed")
PY

mapfile -t GPU_MEMORY < <(
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits
)
if [[ "${#GPU_MEMORY[@]}" -ne 8 ]]; then
  echo "Exactly eight GPUs are required; found ${#GPU_MEMORY[@]}" >&2
  exit 2
fi
for index in "${!GPU_MEMORY[@]}"; do
  IFS=',' read -r used free <<<"${GPU_MEMORY[$index]}"
  used="${used//[[:space:]]/}"
  free="${free//[[:space:]]/}"
  if [[ "$ALLOW_SHARED_GPUS" == "true" ]]; then
    if (( free < SHARED_GPU_MIN_FREE_MIB )); then
      echo "GPU $index has only ${free} MiB free; shared launch requires ${SHARED_GPU_MIN_FREE_MIB} MiB" >&2
      exit 2
    fi
  elif (( used > 10 )); then
    echo "GPU $index is not idle: ${used} MiB used" >&2
    exit 2
  fi
done

available_bytes="$(df --output=avail -B1 "$INSTALL_ROOT" | tail -n 1 | tr -d ' ')"
if (( available_bytes < 500 * 1024 * 1024 * 1024 )); then
  echo "Less than 500 GiB free under $INSTALL_ROOT" >&2
  exit 2
fi

set -a
source "$SECRET_FILE"
set +a
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY is missing from the secret file}"

mkdir -p "$RUN_DIR" "$JUDGE_ROOT" "$REWARD_LOG_DIR" "$EPOCH_SNAPSHOT_DIR"
if [[ -L "$PARENT_ALIAS" ]]; then
  if [[ "$(readlink -f "$PARENT_ALIAS")" != "$(readlink -f "$PARENT")" ]]; then
    echo "Existing parent alias resolves to the wrong checkpoint: $PARENT_ALIAS" >&2
    exit 2
  fi
elif [[ -e "$PARENT_ALIAS" ]]; then
  echo "Parent alias path exists but is not a symlink: $PARENT_ALIAS" >&2
  exit 2
else
  ln -s "$PARENT" "$PARENT_ALIAS"
fi

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE="disabled"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export TOKENIZERS_PARALLELISM="false"
export DEBUG_MODE="false"
export PATHVLM_REQUIRE_SOURCE_AUDIT="true"
export PATHVLM_IMAGE_HASH_MANIFEST="$IMAGE_HASH_MANIFEST"
export PATHVLM_REWARD_LOG_DIR="$REWARD_LOG_DIR"
export PATHVLM_TRAINING_SEGMENT="kimi26_stage3_full3epoch_seed42_${SEGMENT_ID}"
export PATHVLM_STAGE3_JUDGE_ROOT="$JUDGE_ROOT"
export PATHVLM_OPENROUTER_MODEL_ID="moonshotai/kimi-k2.6"
export PATHVLM_OPENROUTER_PROVIDER_ONLY="inceptron"
export PATHVLM_OPENROUTER_MAX_TOKEN_PARAMETER="max_tokens"
export PATHVLM_OPENROUTER_MAX_JUDGE_TOKENS="1024"
export PATHVLM_OPENROUTER_ZDR_REQUIRED="true"
export PATHVLM_OPENROUTER_DATA_COLLECTION="deny"
export PATHVLM_OPENROUTER_ENFORCE_DISTILLABLE_TEXT="true"
export PATHVLM_OPENROUTER_REASONING_ENABLED="false"
export PATHVLM_OPENROUTER_MIN_REQUEST_INTERVAL_SECONDS="$MIN_REQUEST_INTERVAL_SECONDS"
export PATHVLM_OPENROUTER_RETRY_DELAYS_SECONDS="15,45,90"
export PATHVLM_OPENROUTER_CACHE_NAMESPACE="$SEGMENT_ID"
export PATHVLM_OPENROUTER_LIMIT_USD="$PATHVLM_STAGE3_FORMAL_BUDGET_USD"
export PATHVLM_OPENROUTER_RESERVE_USD="0.05"
export PATHVLM_OPENROUTER_MAX_UNIQUE_REQUESTS="$MAX_UNIQUE_REQUESTS"
export PATHVLM_STAGE3_RULE_FALLBACK_ENABLED="true"
export PATHVLM_STAGE3_RULE_FALLBACK_TOTAL_LIMIT="$FALLBACK_TOTAL_LIMIT"
export PATHVLM_STAGE3_RULE_FALLBACK_CONSECUTIVE_LIMIT="$FALLBACK_CONSECUTIVE_LIMIT"
export PATHVLM_TRAIN_STATE_AUDIT_NAME="kimi26_full3epoch_train_state_audit.json"
export PATHVLM_EPOCH_SNAPSHOT_STEPS="500,1000,1500"
export PATHVLM_EPOCH_SNAPSHOT_DIR="$EPOCH_SNAPSHOT_DIR"
if [[ -n "$RESUME_FROM" ]]; then
  export PATHVLM_RESUME_FROM_CHECKPOINT="$RESUME_FROM"
else
  unset PATHVLM_RESUME_FROM_CHECKPOINT || true
fi

cd "$REPO_ROOT/vendor/open-r1-multimodal"
cmd=(
  "$PYTHON" -m torch.distributed.run
  --nproc_per_node=8
  "--master_port=$MASTER_PORT"
  "$REPO_ROOT/scripts/grpo_pathmmu.py"
  --deepspeed "$REPO_ROOT/configs/deepspeed/ds_z3_optimizer_offload_torch_adamw.json"
  --output_dir "$OUTPUT_DIR"
  --model_name_or_path "$PARENT_ALIAS"
  --dataset_name "$DATASET"
  --image_root /
  --reward_funcs accuracy format process
  --freeze_vision_modules true
  --max_pixels 65536
  --min_pixels 3136
  --num_generations 4
  --max_completion_length 192
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 1
  --learning_rate 1.0e-6
  --logging_steps 1
  --bf16 true
  --torch_dtype bfloat16
  --gradient_checkpointing true
  --attn_implementation sdpa
  --beta 0.04
  --num_iterations 1
  --max_steps 1500
  --save_strategy steps
  --save_steps "$SAVE_STEPS"
  --save_total_limit 2
  --save_only_model false
  --report_to none
  --seed 42
  --data_seed 42
  --remove_unused_columns false
)

printf 'Starting full Kimi 2.6 Stage3 arm; segment=%s resume=%s log=%s\n' \
  "$SEGMENT_ID" "${RESUME_FROM:-none}" "$TRAIN_LOG"
"${cmd[@]}" 2>&1 | tee "$TRAIN_LOG"
