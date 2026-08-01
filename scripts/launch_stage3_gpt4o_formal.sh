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
PENALTY="0.4"
MAX_HTTP_ATTEMPTS=12360
RESERVE_USD="0.02"
# User requested no USD budget ceiling.  This is a technical ledger capacity
# derived from the physical-attempt cap, not a spending target or selection gate.
ACCOUNTING_CAPACITY_USD="247.20"
MAX_JUDGE_TOKENS=320

: "${PATHVLM_STAGE3_RUN_DIR:?Set a dedicated GPT-4o formal run directory}"
RUN_DIR="$(readlink -m "$PATHVLM_STAGE3_RUN_DIR")"
expected_parent="$INSTALL_ROOT/runs/stage3_process_grpo"
case "$RUN_DIR" in
  "$expected_parent"/*) ;;
  *) echo "Run directory must be a child of $expected_parent" >&2; exit 2 ;;
esac

PARENT_ALIAS="$RUN_DIR/parent_Qwen2.5-VL-Stage2-epoch02-step1000"
OUTPUT_DIR="$RUN_DIR/output"
JUDGE_ROOT="$RUN_DIR/judge"
EPOCH_SNAPSHOT_DIR="$RUN_DIR/epoch_model_snapshots"
MASTER_PORT="${PATHVLM_STAGE3_MASTER_PORT:-29740}"
SAVE_STEPS="${PATHVLM_STAGE3_SAVE_STEPS:-100}"
SEGMENT_ID="${PATHVLM_STAGE3_SEGMENT_ID:-segment00}"
RESUME_FROM="${PATHVLM_STAGE3_RESUME_FROM_CHECKPOINT:-}"
REWARD_LOG_DIR="$RUN_DIR/reward_audit/$SEGMENT_ID"
TRAIN_LOG="$RUN_DIR/train_${SEGMENT_ID}.log"
PREFLIGHT="$RUN_DIR/launch_preflight_${SEGMENT_ID}.json"

[[ "$SAVE_STEPS" == "100" ]] || { echo "Formal save interval must be 100" >&2; exit 2; }
[[ "$SEGMENT_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Unsafe segment id" >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "GRPO Python is missing: $PYTHON" >&2; exit 2; }
[[ -f "$PARENT_MANIFEST" && -f "$PARENT/model.safetensors.index.json" ]] || {
  echo "Stage2 parent snapshot is incomplete" >&2; exit 2;
}
[[ -f "$DATASET" && -f "$IMAGE_HASH_MANIFEST" ]] || {
  echo "Frozen data or image manifest is missing" >&2; exit 2;
}
[[ -f "$SECRET_FILE" && "$(stat -c '%a' "$SECRET_FILE")" == "600" ]] || {
  echo "AIGCBest secret must exist with mode 0600" >&2; exit 2;
}
[[ -f "$SMOKE_RESULT" ]] || { echo "Frozen GPT-4o smoke result is missing" >&2; exit 2; }
[[ -z "$("$WORKSPACE_ROOT/.local-git/usr/bin/git" -C "$REPO_ROOT" status --porcelain)" ]] || {
  echo "Repository must be clean before formal launch" >&2; exit 2;
}

if [[ -n "$RESUME_FROM" ]]; then
  [[ -d "$OUTPUT_DIR" && -d "$RESUME_FROM" ]] || { echo "Resume path is missing" >&2; exit 2; }
  resolved_output="$(readlink -f "$OUTPUT_DIR")"
  resolved_resume="$(readlink -f "$RESUME_FROM")"
  [[ "$(dirname "$resolved_resume")" == "$resolved_output" ]] || {
    echo "Resume checkpoint must be an immediate output child" >&2; exit 2;
  }
  "$PYTHON" "$REPO_ROOT/scripts/stage3_checkpoint_recovery.py" \
    validate "$resolved_resume" --world-size 8 >/dev/null
  RESUME_FROM="$resolved_resume"
elif [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing existing output without a validated resume" >&2
  exit 2
fi

set -a
source "$SECRET_FILE"
set +a
: "${AIGCBEST_API_KEY:?AIGCBEST_API_KEY is missing}"

"$PYTHON" - "$PARENT_MANIFEST" "$PARENT_MANIFEST_SHA256" \
  "$DATASET" "$DATASET_SHA256" "$IMAGE_HASH_MANIFEST" "$IMAGE_HASH_MANIFEST_SHA256" \
  "$SMOKE_RESULT" "$SMOKE_RESULT_SHA256" "$MASTER_PORT" "$MODEL_ID" <<'PY'
import hashlib, json, os, socket, sys, urllib.request
from pathlib import Path
(parent, parent_sha, dataset, dataset_sha, images, images_sha,
 smoke, smoke_sha, port, model) = sys.argv[1:]
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
for path, expected in ((parent,parent_sha),(dataset,dataset_sha),(images,images_sha),(smoke,smoke_sha)):
    actual=digest(path)
    if actual != expected: raise SystemExit(f"preflight hash mismatch: {path}: {actual}")
manifest=json.loads(Path(parent).read_text())
if manifest.get("global_step") != 1000 or manifest.get("epoch") != 2.0:
    raise SystemExit("Stage2 parent identity mismatch")
for row in manifest.get("files", []):
    path=Path(parent).parent / row["name"]
    if not path.is_file() or path.stat().st_size != row["size_bytes"] or digest(path) != row["sha256"]:
        raise SystemExit(f"Stage2 parent file mismatch: {path}")
yaml=Path(dataset).read_text()
paths=[line.split(":",1)[1].strip() for line in yaml.splitlines() if line.strip().startswith("- json_path:")]
if len(paths) != 1: raise SystemExit("RL1000 YAML path mismatch")
records=json.loads(Path(paths[0]).read_text())
frozen=json.loads(Path(images).read_text()).get("images")
if len(records) != 1000 or not isinstance(frozen, dict): raise SystemExit("RL1000 count mismatch")
for index,row in enumerate(records):
    image=Path(row["image"]); expected=frozen.get(image.name)
    if not image.is_file() or not expected or digest(image) != expected:
        raise SystemExit(f"RL1000 image mismatch at {index}: {image}")
prior=json.loads(Path(smoke).read_text())
expected={"status":"passed","training_call":False,"requested_model":model,
          "served_model":model,"response_schema":"pathvlm_stage3_process_events_v1",
          "max_tokens":320,"retry_count":0}
bad={k:{"expected":v,"actual":prior.get(k)} for k,v in expected.items() if prior.get(k)!=v}
if bad: raise SystemExit(f"GPT-4o smoke mismatch: {bad}")
headers={"Authorization":"Bearer "+os.environ["AIGCBEST_API_KEY"],"User-Agent":"PathVLM-R1 formal preflight"}
with urllib.request.urlopen(urllib.request.Request("https://api2.aigcbest.top/v1/models",headers=headers),timeout=30) as r:
    catalog=json.load(r).get("data",[])
if sum(isinstance(x,dict) and x.get("id")==model for x in catalog)!=1:
    raise SystemExit("exact GPT-4o model absent")
with urllib.request.urlopen("https://api2.aigcbest.top/api/pricing",timeout=30) as r:
    pricing=json.load(r).get("data",[])
rows=[x for x in pricing if isinstance(x,dict) and x.get("model_name")==model]
if len(rows)!=1 or rows[0].get("model_ratio")!=1.25 or rows[0].get("completion_ratio")!=4:
    raise SystemExit(f"GPT-4o price contract changed: {rows}")
with socket.socket() as sock:
    try: sock.bind(("127.0.0.1",int(port)))
    except OSError as exc: raise SystemExit(f"master port unavailable: {exc}")
PY

mapfile -t GPU_MEMORY < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
[[ "${#GPU_MEMORY[@]}" -eq 8 ]] || { echo "Exactly eight GPUs are required" >&2; exit 2; }
for index in "${!GPU_MEMORY[@]}"; do
  used="${GPU_MEMORY[$index]//[[:space:]]/}"
  (( used <= 10 )) || { echo "GPU $index is not idle: ${used} MiB" >&2; exit 2; }
done
available_bytes="$(df --output=avail -B1 "$INSTALL_ROOT" | tail -n 1 | tr -d ' ')"
(( available_bytes >= 500 * 1024 * 1024 * 1024 )) || { echo "Less than 500 GiB free" >&2; exit 2; }
if [[ "${PATHVLM_STAGE3_PREFLIGHT_ONLY:-false}" == "true" ]]; then
  echo "Formal GPT-4o Stage3 static/read-only preflight passed"
  exit 0
fi

mkdir -p "$RUN_DIR" "$JUDGE_ROOT" "$REWARD_LOG_DIR" "$EPOCH_SNAPSHOT_DIR"
if [[ -L "$PARENT_ALIAS" ]]; then
  [[ "$(readlink -f "$PARENT_ALIAS")" == "$(readlink -f "$PARENT")" ]] || exit 2
elif [[ -e "$PARENT_ALIAS" ]]; then
  echo "Parent alias exists but is not a symlink" >&2; exit 2
else
  ln -s "$PARENT" "$PARENT_ALIAS"
fi

"$PYTHON" - "$PREFLIGHT" "$REPO_ROOT" "$SEGMENT_ID" "$RESUME_FROM" <<'PY'
import json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
path,repo,segment,resume=sys.argv[1:]
commit=subprocess.check_output(["/home/dataset-assist-0/czy/wjy/.local-git/usr/bin/git","-C",repo,"rev-parse","HEAD"],text=True).strip()
value={"schema_version":1,"created_at":datetime.now(timezone.utc).isoformat(),"status":"passed",
"formal_result":False,"run_class":"formal_stage3_gpt4o_seed42","repository_commit":commit,
"judge_gateway":"aigcbest","judge_model":"gpt-4o-2024-08-06","penalty":0.4,
"user_usd_budget_limit":None,"technical_accounting_capacity_usd":247.2,
"maximum_physical_http_attempts":12360,"maximum_logical_judgments":12000,
"retry_delays_seconds":[15,45,90],"rule_fallback_total_limit":24,
"rule_fallback_consecutive_limit":4,"max_steps":1500,"save_steps":100,
"segment":segment,"resume_from":resume or None,"test_accessed":False}
Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
PY

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE="disabled" HF_HUB_OFFLINE="1" TRANSFORMERS_OFFLINE="1"
export TOKENIZERS_PARALLELISM="false" DEBUG_MODE="false"
export PATHVLM_REQUIRE_SOURCE_AUDIT="true"
export PATHVLM_IMAGE_HASH_MANIFEST="$IMAGE_HASH_MANIFEST"
export PATHVLM_REWARD_LOG_DIR="$REWARD_LOG_DIR"
export PATHVLM_TRAINING_SEGMENT="gpt4o_stage3_full_seed42_${SEGMENT_ID}"
export PATHVLM_STAGE3_JUDGE_BACKEND="aigcbest"
export PATHVLM_STAGE3_JUDGE_ROOT="$JUDGE_ROOT"
export PATHVLM_STAGE3_PENALTY="$PENALTY"
export PATHVLM_AIGCBEST_CACHE_NAMESPACE="$SEGMENT_ID"
export PATHVLM_AIGCBEST_LIMIT_USD="$ACCOUNTING_CAPACITY_USD"
export PATHVLM_AIGCBEST_RESERVE_USD="$RESERVE_USD"
export PATHVLM_AIGCBEST_MAX_HTTP_ATTEMPTS="$MAX_HTTP_ATTEMPTS"
export PATHVLM_AIGCBEST_MAX_JUDGE_TOKENS="$MAX_JUDGE_TOKENS"
export PATHVLM_AIGCBEST_MIN_REQUEST_INTERVAL_SECONDS="1"
export PATHVLM_AIGCBEST_RETRY_DELAYS_SECONDS="15,45,90"
export PATHVLM_STAGE3_RULE_FALLBACK_ENABLED="true"
export PATHVLM_STAGE3_RULE_FALLBACK_TOTAL_LIMIT="24"
export PATHVLM_STAGE3_RULE_FALLBACK_CONSECUTIVE_LIMIT="4"
export PATHVLM_TRAIN_STATE_AUDIT_NAME="gpt4o_full_train_state_audit.json"
export PATHVLM_EPOCH_SNAPSHOT_STEPS="500,1000,1500"
export PATHVLM_EPOCH_SNAPSHOT_DIR="$EPOCH_SNAPSHOT_DIR"
if [[ -n "$RESUME_FROM" ]]; then export PATHVLM_RESUME_FROM_CHECKPOINT="$RESUME_FROM"; else unset PATHVLM_RESUME_FROM_CHECKPOINT || true; fi

cd "$REPO_ROOT/vendor/open-r1-multimodal"
cmd=(
  "$PYTHON" -m torch.distributed.run --nproc_per_node=8 "--master_port=$MASTER_PORT"
  "$REPO_ROOT/scripts/grpo_pathmmu.py"
  --deepspeed "$REPO_ROOT/configs/deepspeed/ds_z3_optimizer_offload_torch_adamw.json"
  --output_dir "$OUTPUT_DIR" --model_name_or_path "$PARENT_ALIAS"
  --dataset_name "$DATASET" --image_root / --reward_funcs accuracy format process
  --freeze_vision_modules true --max_pixels 65536 --min_pixels 3136
  --num_generations 4 --max_completion_length 192 --per_device_train_batch_size 1
  --gradient_accumulation_steps 1 --learning_rate 1.0e-6 --logging_steps 1
  --bf16 true --torch_dtype bfloat16 --gradient_checkpointing true
  --attn_implementation sdpa --beta 0.04 --num_iterations 1 --max_steps 1500
  --save_strategy steps --save_steps 100 --save_total_limit 2 --save_only_model false
  --report_to none --seed 42 --data_seed 42 --remove_unused_columns false
)
printf 'Starting formal GPT-4o Stage3; segment=%s resume=%s log=%s\n' "$SEGMENT_ID" "${RESUME_FROM:-none}" "$TRAIN_LOG"
"${cmd[@]}" 2>&1 | tee "$TRAIN_LOG"
