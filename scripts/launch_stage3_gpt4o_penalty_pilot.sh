#!/usr/bin/env bash
set -euo pipefail
umask 077

WORKSPACE_ROOT="/home/dataset-assist-0/czy/wjy"
REPO_ROOT="$WORKSPACE_ROOT/myr1"
INSTALL_ROOT="$WORKSPACE_ROOT/pathvlm_r1_v1_a100"
PYTHON="$INSTALL_ROOT/envs/grpo/bin/python"
PARENT="$INSTALL_ROOT/transferred_checkpoints/outcome_grpo_n1000_seed42_epoch02_step1000"
PARENT_MANIFEST="$PARENT/snapshot_manifest.json"
PARENT_MANIFEST_SHA256="83df2570a33bd760bebb6ef8afca175b71c33bb585e107cdb2f3889edebc04e0"
DATASET="$INSTALL_ROOT/data/pathmmu_image_disjoint_v2/grpo/pathvlm_rl_n1000.yaml"
DATASET_SHA256="0d443486bebf27a8611a670f65af19f961f76093fe7f9f3b838e7127485f7260"
IMAGE_HASH_MANIFEST="$REPO_ROOT/data/pathmmu_image_disjoint_v2/image_content_sha256.json"
IMAGE_HASH_MANIFEST_SHA256="e200edf3659e5af860d824a539bb0c5a29ee0c790de8b7a4ec528989c45b51ed"
SECRET_FILE="$WORKSPACE_ROOT/.secrets/aigcbest.env"
SMOKE_RESULT="$WORKSPACE_ROOT/pathvlm_revision_eval_a100/reports/aigcbest_gpt4o_stage3_smoke_20260801/pathmmu_validation_result.json"
SMOKE_RESULT_SHA256="d6d1d148b712044be874e950a462a0d87ce6a30e15ca541addbebd175adfc148"
MODEL_ID="gpt-4o-2024-08-06"
BUDGET_USD="6"
MAX_HTTP_ATTEMPTS="800"
RESERVE_USD="0.02"
MAX_JUDGE_TOKENS="320"

PENALTY="${1:-}"
case "$PENALTY" in
  0.3) TAG="0p3"; DEFAULT_PORT="29703" ;;
  0.4) TAG="0p4"; DEFAULT_PORT="29704" ;;
  0.5) TAG="0p5"; DEFAULT_PORT="29705" ;;
  *) echo "Usage: $0 {0.3|0.4|0.5}" >&2; exit 2 ;;
esac

MASTER_PORT="${PATHVLM_STAGE3_MASTER_PORT:-$DEFAULT_PORT}"
RUN_DIR="$INSTALL_ROOT/runs/stage3_process_grpo/gpt4o_penalty_${TAG}_100step_seed42_20260801"
PARENT_ALIAS="$RUN_DIR/parent_Qwen2.5-VL-Stage2-epoch02-step1000"
OUTPUT_DIR="$RUN_DIR/output"
JUDGE_ROOT="$RUN_DIR/judge"
REWARD_LOG_DIR="$RUN_DIR/reward_audit"
TRAIN_LOG="$RUN_DIR/train.log"
PREFLIGHT="$RUN_DIR/launch_preflight.json"
SEGMENT="gpt4o_stage3_penalty_${TAG}_100step_seed42"

[[ -x "$PYTHON" ]] || { echo "GRPO Python is missing: $PYTHON" >&2; exit 2; }
[[ -f "$PARENT_MANIFEST" && -f "$PARENT/model.safetensors.index.json" ]] || {
  echo "Stage2 parent snapshot is incomplete: $PARENT" >&2
  exit 2
}
[[ -f "$DATASET" && -f "$IMAGE_HASH_MANIFEST" ]] || {
  echo "Frozen RL1000 dataset or image manifest is missing" >&2
  exit 2
}
[[ -f "$SECRET_FILE" && "$(stat -c '%a' "$SECRET_FILE")" == "600" ]] || {
  echo "AIGCBest secret file must exist with mode 0600: $SECRET_FILE" >&2
  exit 2
}
[[ -f "$SMOKE_RESULT" ]] || { echo "GPT-4o smoke result is missing" >&2; exit 2; }
[[ ! -e "$RUN_DIR" ]] || {
  echo "Refusing to overwrite or reuse a coefficient pilot: $RUN_DIR" >&2
  exit 2
}
[[ -z "$("$WORKSPACE_ROOT/.local-git/usr/bin/git" -C "$REPO_ROOT" status --porcelain)" ]] || {
  echo "Repository must be clean before a paid pilot" >&2
  exit 2
}

set -a
source "$SECRET_FILE"
set +a
: "${AIGCBEST_API_KEY:?AIGCBEST_API_KEY is missing from the secret file}"

"$PYTHON" - "$PARENT_MANIFEST" "$PARENT_MANIFEST_SHA256" \
  "$DATASET" "$DATASET_SHA256" "$IMAGE_HASH_MANIFEST" "$IMAGE_HASH_MANIFEST_SHA256" \
  "$SMOKE_RESULT" "$SMOKE_RESULT_SHA256" "$MASTER_PORT" "$MODEL_ID" "$PENALTY" <<'PY'
import hashlib
import json
import os
import socket
import sys
import urllib.request
from pathlib import Path

(
    parent_manifest_path, parent_manifest_sha, dataset_yaml, dataset_sha,
    image_manifest_path, image_manifest_sha, smoke_path, smoke_sha,
    port_text, model_id, penalty_text,
) = sys.argv[1:]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

expected_hashes = {
    parent_manifest_path: parent_manifest_sha,
    dataset_yaml: dataset_sha,
    image_manifest_path: image_manifest_sha,
    smoke_path: smoke_sha,
}
for path, expected in expected_hashes.items():
    actual = digest(path)
    if actual != expected:
        raise SystemExit(f"preflight SHA-256 mismatch: {path}: {actual}")

manifest_path = Path(parent_manifest_path)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("global_step") != 1000 or manifest.get("epoch") != 2.0:
    raise SystemExit("Stage2 parent manifest identity mismatch")
for row in manifest.get("files", []):
    path = manifest_path.parent / row["name"]
    if not path.is_file() or path.stat().st_size != row["size_bytes"] or digest(path) != row["sha256"]:
        raise SystemExit(f"Stage2 parent file integrity mismatch: {path}")

yaml_text = Path(dataset_yaml).read_text(encoding="utf-8")
json_paths = [
    line.split(":", 1)[1].strip()
    for line in yaml_text.splitlines()
    if line.strip().startswith("- json_path:")
]
if len(json_paths) != 1:
    raise SystemExit("RL1000 YAML must identify exactly one JSON dataset")
records = json.loads(Path(json_paths[0]).read_text(encoding="utf-8"))
image_hashes = json.loads(Path(image_manifest_path).read_text(encoding="utf-8")).get("images")
if not isinstance(records, list) or len(records) != 1000 or not isinstance(image_hashes, dict):
    raise SystemExit("RL1000 count or image manifest mismatch")
for index, row in enumerate(records):
    image = Path(row["image"])
    expected = image_hashes.get(image.name)
    if not image.is_file() or not expected or digest(image) != expected:
        raise SystemExit(f"RL1000 image integrity mismatch at row {index}: {image}")

smoke = json.loads(Path(smoke_path).read_text(encoding="utf-8"))
expected_smoke = {
    "status": "passed",
    "training_call": False,
    "requested_model": model_id,
    "served_model": model_id,
    "response_schema": "pathvlm_stage3_process_events_v1",
    "max_tokens": 320,
    "retry_count": 0,
}
mismatch = {
    name: {"expected": expected, "actual": smoke.get(name)}
    for name, expected in expected_smoke.items()
    if smoke.get(name) != expected
}
if mismatch:
    raise SystemExit(f"GPT-4o smoke contract mismatch: {mismatch}")

headers = {
    "Authorization": "Bearer " + os.environ["AIGCBEST_API_KEY"],
    "User-Agent": "PathVLM-R1 Stage3 preflight",
}
with urllib.request.urlopen(
    urllib.request.Request("https://api2.aigcbest.top/v1/models", headers=headers), timeout=30
) as response:
    catalog = json.load(response).get("data", [])
if sum(isinstance(row, dict) and row.get("id") == model_id for row in catalog) != 1:
    raise SystemExit("exact GPT-4o model is absent from authenticated catalog")
with urllib.request.urlopen("https://api2.aigcbest.top/api/pricing", timeout=30) as response:
    pricing = json.load(response).get("data", [])
rows = [row for row in pricing if isinstance(row, dict) and row.get("model_name") == model_id]
if len(rows) != 1 or rows[0].get("model_ratio") != 1.25 or rows[0].get("completion_ratio") != 4:
    raise SystemExit(f"GPT-4o public price contract changed: {rows}")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    try:
        sock.bind(("127.0.0.1", int(port_text)))
    except OSError as exc:
        raise SystemExit(f"master port {port_text} is unavailable: {exc}") from exc
print(json.dumps({
    "status": "static_preflight_passed",
    "model": model_id,
    "penalty": float(penalty_text),
    "records": len(records),
}, sort_keys=True))
PY

mapfile -t GPU_MEMORY < <(
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
)
[[ "${#GPU_MEMORY[@]}" -eq 8 ]] || { echo "Exactly eight GPUs are required" >&2; exit 2; }
for index in "${!GPU_MEMORY[@]}"; do
  used="${GPU_MEMORY[$index]//[[:space:]]/}"
  (( used <= 10 )) || { echo "GPU $index is not idle: ${used} MiB" >&2; exit 2; }
done
available_bytes="$(df --output=avail -B1 "$INSTALL_ROOT" | tail -n 1 | tr -d ' ')"
(( available_bytes >= 500 * 1024 * 1024 * 1024 )) || {
  echo "Less than 500 GiB free under $INSTALL_ROOT" >&2
  exit 2
}

mkdir -p "$RUN_DIR" "$JUDGE_ROOT" "$REWARD_LOG_DIR"
ln -s "$PARENT" "$PARENT_ALIAS"
"$PYTHON" - "$PREFLIGHT" "$REPO_ROOT" "$PENALTY" "$MASTER_PORT" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

path, repo, penalty, port = sys.argv[1:]
commit = subprocess.check_output(
    ["/home/dataset-assist-0/czy/wjy/.local-git/usr/bin/git", "-C", repo, "rev-parse", "HEAD"],
    text=True,
).strip()
value = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "status": "passed",
    "formal_result": False,
    "run_class": "matched_100step_coefficient_pilot",
    "repository_commit": commit,
    "parent_manifest_sha256": "83df2570a33bd760bebb6ef8afca175b71c33bb585e107cdb2f3889edebc04e0",
    "judge_gateway": "aigcbest",
    "judge_model": "gpt-4o-2024-08-06",
    "penalty": float(penalty),
    "budget_limit_usd": 6.0,
    "maximum_http_attempts": 800,
    "per_attempt_reserve_usd": 0.02,
    "retry_delays_seconds": [15, 45, 90],
    "rule_fallback_total_limit": 24,
    "rule_fallback_consecutive_limit": 4,
    "automatic_training_restarts": 0,
    "max_steps": 100,
    "save_steps": 100,
    "master_port": int(port),
    "test_accessed": False,
}
Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

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
export PATHVLM_TRAINING_SEGMENT="$SEGMENT"
export PATHVLM_STAGE3_JUDGE_BACKEND="aigcbest"
export PATHVLM_STAGE3_JUDGE_ROOT="$JUDGE_ROOT"
export PATHVLM_STAGE3_PENALTY="$PENALTY"
export PATHVLM_AIGCBEST_CACHE_NAMESPACE="$SEGMENT"
export PATHVLM_AIGCBEST_LIMIT_USD="$BUDGET_USD"
export PATHVLM_AIGCBEST_RESERVE_USD="$RESERVE_USD"
export PATHVLM_AIGCBEST_MAX_HTTP_ATTEMPTS="$MAX_HTTP_ATTEMPTS"
export PATHVLM_AIGCBEST_MAX_JUDGE_TOKENS="$MAX_JUDGE_TOKENS"
export PATHVLM_AIGCBEST_MIN_REQUEST_INTERVAL_SECONDS="1"
export PATHVLM_AIGCBEST_RETRY_DELAYS_SECONDS="15,45,90"
export PATHVLM_STAGE3_RULE_FALLBACK_ENABLED="true"
export PATHVLM_STAGE3_RULE_FALLBACK_TOTAL_LIMIT="24"
export PATHVLM_STAGE3_RULE_FALLBACK_CONSECUTIVE_LIMIT="4"
export PATHVLM_TRAIN_STATE_AUDIT_NAME="gpt4o_penalty_${TAG}_train_state_audit.json"

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
  --max_steps 100
  --save_strategy steps
  --save_steps 100
  --save_total_limit 2
  --save_only_model false
  --report_to none
  --seed 42
  --data_seed 42
  --remove_unused_columns false
)

printf 'Starting GPT-4o Stage3 coefficient pilot; penalty=%s budget=%s max_http_attempts=%s log=%s\n' \
  "$PENALTY" "$BUDGET_USD" "$MAX_HTTP_ATTEMPTS" "$TRAIN_LOG"
"${cmd[@]}" 2>&1 | tee "$TRAIN_LOG"
