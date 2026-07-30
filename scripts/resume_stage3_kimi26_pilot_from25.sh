#!/usr/bin/env bash
set -euo pipefail
umask 077

WORKSPACE_ROOT="/home/dataset-assist-0/czy/wjy"
REPO_ROOT="$WORKSPACE_ROOT/myr1"
INSTALL_ROOT="$WORKSPACE_ROOT/pathvlm_r1_v1_a100"
PYTHON="$INSTALL_ROOT/envs/grpo/bin/python"
RUN_DIR="$INSTALL_ROOT/runs/stage3_process_grpo/kimi26_50step_seed42_20260730"
OUTPUT_DIR="$RUN_DIR/output_attempt03"
CHECKPOINT="$OUTPUT_DIR/checkpoint-25"
PARENT_ALIAS="$RUN_DIR/parent_Qwen2.5-VL-Stage2-epoch02-step1000"
DATASET="$INSTALL_ROOT/data/pathmmu_image_disjoint_v2/grpo/pathvlm_rl_n1000.yaml"
IMAGE_HASH_MANIFEST="$REPO_ROOT/data/pathmmu_image_disjoint_v2/image_content_sha256.json"
SECRET_FILE="$WORKSPACE_ROOT/.secrets/openrouter.env"
JUDGE_ROOT="$RUN_DIR/judge"
REWARD_LOG_DIR="$RUN_DIR/reward_audit_attempt07_resume25"
TRAIN_LOG="$RUN_DIR/train_attempt07_resume25.log"
MASTER_PORT="${PATHVLM_STAGE3_MASTER_PORT:-29631}"

if [[ ! -x "$PYTHON" ]]; then
  echo "GRPO Python is missing: $PYTHON" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT/trainer_state.json" ||
      ! -f "$CHECKPOINT/model.safetensors.index.json" ||
      ! -f "$CHECKPOINT/latest" ]]; then
  echo "Step-25 recovery checkpoint is incomplete: $CHECKPOINT" >&2
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
if [[ ! -L "$PARENT_ALIAS" ]]; then
  echo "The audited Stage2 parent alias is missing: $PARENT_ALIAS" >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR/checkpoint-50" ]]; then
  echo "Refusing to resume because checkpoint-50 already exists" >&2
  exit 2
fi
if [[ -e "$REWARD_LOG_DIR" || -e "$TRAIN_LOG" ]]; then
  echo "Refusing to overwrite an existing recovery audit or log" >&2
  exit 2
fi

"$PYTHON" - "$CHECKPOINT" "$JUDGE_ROOT" "$MASTER_PORT" <<'PY'
import json
import socket
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
judge_root = Path(sys.argv[2])
port = int(sys.argv[3])

state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
if state.get("global_step") != 25:
    raise SystemExit(f"recovery checkpoint global_step is not 25: {state.get('global_step')}")

index = json.loads(
    (checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8")
)
shards = sorted(set(index.get("weight_map", {}).values()))
if len(shards) != 4 or any(not (checkpoint / name).is_file() for name in shards):
    raise SystemExit(f"recovery checkpoint model shards are incomplete: {shards}")

ledger = json.loads((judge_root / "budget_ledger.json").read_text(encoding="utf-8"))
expected = {
    "limit_usd": 15.0,
    "reserve_usd": 0.05,
    "max_unique_requests": 550,
    "completed_unique_requests": 332,
    "budget_breached": False,
}
for key, value in expected.items():
    if ledger.get(key) != value:
        raise SystemExit(
            f"Judge recovery ledger mismatch for {key}: "
            f"expected {value!r}, found {ledger.get(key)!r}"
        )
if ledger.get("reservations"):
    raise SystemExit("Judge recovery ledger has unresolved reservations")
if abs(float(ledger.get("committed_spend_usd", -1)) - 0.68216033) > 1e-9:
    raise SystemExit("Judge recovery ledger spend does not match the reconciled value")

smoke_events = []
for audit_path in sorted((judge_root / "audit").glob("rank_*.jsonl")):
    smoke_events.extend(
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
smoke = next(
    (
        event
        for event in reversed(smoke_events)
        if event.get("training_segment") == "kimi26_stage3_recovery_contract_smoke"
        and event.get("record_index") == 271
        and event.get("status") == "completed"
    ),
    None,
)
if smoke is None:
    raise SystemExit("the exact failed-item recovery smoke audit is missing")
cache = json.loads(Path(smoke["cache_path"]).read_text(encoding="utf-8"))
request = cache.get("request", {})
schema = (
    request.get("response_format", {})
    .get("json_schema", {})
    .get("schema", {})
)
evidence_properties = (
    schema.get("properties", {})
    .get("evidence", {})
    .get("properties", {})
)
if request.get("max_tokens") != 1024:
    raise SystemExit("recovery smoke did not use max_tokens=1024")
if not evidence_properties or {
    value.get("maxLength") for value in evidence_properties.values()
} != {512}:
    raise SystemExit("recovery smoke did not use the 512-character absolute ceiling")
if cache.get("response", {}).get("choices", [{}])[0].get("finish_reason") != "stop":
    raise SystemExit("recovery smoke did not finish cleanly")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as exc:
        raise SystemExit(f"master port {port} is unavailable: {exc}") from exc

print("Stage3 step-25 recovery preflight passed")
PY

mapfile -t GPU_MEMORY < <(
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
)
if [[ "${#GPU_MEMORY[@]}" -ne 8 ]]; then
  echo "Exactly eight GPUs are required; found ${#GPU_MEMORY[@]}" >&2
  exit 2
fi
for index in "${!GPU_MEMORY[@]}"; do
  used="${GPU_MEMORY[$index]//[[:space:]]/}"
  if (( used > 10 )); then
    echo "GPU $index is not idle: ${used} MiB used" >&2
    exit 2
  fi
done

set -a
source "$SECRET_FILE"
set +a
: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY is missing from the secret file}"

mkdir -p "$REWARD_LOG_DIR"
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
export PATHVLM_TRAINING_SEGMENT="kimi26_stage3_pilot_50step_seed42_attempt07_resume25"
export PATHVLM_RESUME_FROM_CHECKPOINT="$CHECKPOINT"
export PATHVLM_STAGE3_JUDGE_ROOT="$JUDGE_ROOT"
export PATHVLM_OPENROUTER_MODEL_ID="moonshotai/kimi-k2.6"
export PATHVLM_OPENROUTER_PROVIDER_ONLY="inceptron"
export PATHVLM_OPENROUTER_MAX_TOKEN_PARAMETER="max_tokens"
export PATHVLM_OPENROUTER_MAX_JUDGE_TOKENS="1024"
export PATHVLM_OPENROUTER_ZDR_REQUIRED="true"
export PATHVLM_OPENROUTER_DATA_COLLECTION="deny"
export PATHVLM_OPENROUTER_ENFORCE_DISTILLABLE_TEXT="true"
export PATHVLM_OPENROUTER_REASONING_ENABLED="false"
export PATHVLM_OPENROUTER_MIN_REQUEST_INTERVAL_SECONDS="4"
export PATHVLM_OPENROUTER_RETRY_DELAYS_SECONDS="15,45,90"
export PATHVLM_OPENROUTER_LIMIT_USD="15"
export PATHVLM_OPENROUTER_RESERVE_USD="0.05"
export PATHVLM_OPENROUTER_MAX_UNIQUE_REQUESTS="550"
export PATHVLM_TRAIN_STATE_AUDIT_NAME="kimi26_stage3_train_state_audit_attempt07.json"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1"

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
  --max_steps 50
  --save_strategy steps
  --save_steps 25
  --save_total_limit 2
  --save_only_model false
  --report_to none
  --seed 42
  --data_seed 42
  --remove_unused_columns false
)

printf 'Resuming Kimi 2.6 Stage3 pilot from checkpoint-25; log=%s\n' "$TRAIN_LOG"
"${cmd[@]}" 2>&1 | tee "$TRAIN_LOG"
