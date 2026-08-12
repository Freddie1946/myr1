#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH to merged selected SFT checkpoint}"
: "${RUN_DIR:?set RUN_DIR}"
MODE=${MODE:-smoke}
[[ "$MODE" == smoke || "$MODE" == formal ]] || exit 2
RUN_ROLE=${RUN_ROLE:-$MODE}
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-1}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-false}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-512}
LEARNING_RATE=${LEARNING_RATE:-1.0e-6}
REWARD_MODE=${REWARD_MODE:-accuracy_format}
FORMAT_REWARD_WEIGHT=${FORMAT_REWARD_WEIGHT:-0.1}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
GENERATION_TEMPERATURE=${GENERATION_TEMPERATURE:-0.9}
GENERATION_TOP_P=${GENERATION_TOP_P:-1.0}
GENERATION_TOP_K=${GENERATION_TOP_K:-0}
GENERATION_TYPICAL_P=${GENERATION_TYPICAL_P:-1.0}
GENERATION_REPETITION_PENALTY=${GENERATION_REPETITION_PENALTY:-1.0}
EXPLICIT_SAVE_STEPS=${EXPLICIT_SAVE_STEPS:-}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
REWARD_LOG_SUBDIR=${REWARD_LOG_SUBDIR:-online_reward_events}
LAUNCH_CONTRACT_NAME=${LAUNCH_CONTRACT_NAME:-launch_contract.json}
TRAIN_LOG_NAME=${TRAIN_LOG_NAME:-train.log}
ALIGNMENT_OUTPUT_NAME=${ALIGNMENT_OUTPUT_NAME:-reward_alignment_verification.json}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-/home/dataset-assist-0/czy/wjy/myr1/configs/deepspeed/ds_z2_gpu_torch_adamw.json}
MASTER_PORT=${MASTER_PORT:-33642}
case "$PER_DEVICE_BATCH" in
  1) STEPS_PER_EPOCH=500 ;;
  2) STEPS_PER_EPOCH=250 ;;
  4) STEPS_PER_EPOCH=125 ;;
  5) STEPS_PER_EPOCH=100 ;;
  10) STEPS_PER_EPOCH=50 ;;
  *) echo "PER_DEVICE_BATCH must be one of 1, 2, 4, 5, 10" >&2; exit 2 ;;
esac
[[ "$GRADIENT_CHECKPOINTING" == true || "$GRADIENT_CHECKPOINTING" == false ]] || exit 2
[[ "$MAX_COMPLETION_LENGTH" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$LORA_R" =~ ^[1-9][0-9]*$ && "$LORA_ALPHA" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ -f "$DEEPSPEED_CONFIG" ]] || exit 2
case "$REWARD_MODE" in
  accuracy_format)
    REWARD_ARGS=(accuracy format)
    REWARD_TYPES=(accuracy format)
    ;;
  accuracy_only)
    REWARD_ARGS=(accuracy)
    REWARD_TYPES=(accuracy)
    ;;
  accuracy_scaled_format)
    REWARD_ARGS=(accuracy format_scaled)
    REWARD_TYPES=(accuracy format_scaled)
    ;;
  *) echo "invalid REWARD_MODE: $REWARD_MODE" >&2; exit 2 ;;
esac
REPO=/home/dataset-assist-0/czy/wjy/myr1
GRPO_PYTHON=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/grpo/bin/python
DATASET_YAML=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/pathmmu_image_disjoint_v2/grpo/pathvlm_rl_n1000.yaml
OUTPUT=$RUN_DIR/output
MAX_STEPS=2
SAVE_STEPS=1
SAVE_LIMIT=1
SAVE_STRATEGY=no
if [[ "$MODE" == formal ]]; then
  # The frozen RL split contains 1,000 prompts.  With four generations and
  # eight ranks, this preserves exactly three prompt exposures while allowing
  # the throughput smoke to select a larger safe micro-batch.
  MAX_STEPS=$((3 * STEPS_PER_EPOCH))
  SAVE_STEPS=100
  SAVE_LIMIT=3
  SAVE_STRATEGY=steps
fi
MAX_STEPS=${MAX_STEPS_OVERRIDE:-$MAX_STEPS}
SAVE_STEPS=${SAVE_STEPS_OVERRIDE:-$SAVE_STEPS}
SAVE_LIMIT=${SAVE_LIMIT_OVERRIDE:-$SAVE_LIMIT}
SAVE_STRATEGY=${SAVE_STRATEGY_OVERRIDE:-$SAVE_STRATEGY}
STOP_AFTER_SAVED_STEP=${STOP_AFTER_SAVED_STEP:-0}
[[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$SAVE_STEPS" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$SAVE_LIMIT" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$STOP_AFTER_SAVED_STEP" =~ ^[0-9]+$ ]] || exit 2
if [[ -n "$EXPLICIT_SAVE_STEPS" ]]; then
  [[ "$EXPLICIT_SAVE_STEPS" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || exit 2
  export PATHVLM_EXPLICIT_SAVE_STEPS=$EXPLICIT_SAVE_STEPS
  SAVE_STRATEGY=no
fi
EXPECTED_COMPLETED_STEPS=$MAX_STEPS
if (( STOP_AFTER_SAVED_STEP > 0 )); then
  (( STOP_AFTER_SAVED_STEP <= MAX_STEPS )) || exit 2
  if [[ -n "$EXPLICIT_SAVE_STEPS" ]]; then
    [[ ",$EXPLICIT_SAVE_STEPS," == *",$STOP_AFTER_SAVED_STEP,"* ]] || exit 2
  else
    [[ "$SAVE_STRATEGY" == steps ]] || exit 2
    (( STOP_AFTER_SAVED_STEP % SAVE_STEPS == 0 )) || exit 2
  fi
  export PATHVLM_STOP_AFTER_SAVED_STEP=$STOP_AFTER_SAVED_STEP
  EXPECTED_COMPLETED_STEPS=$STOP_AFTER_SAVED_STEP
fi
VERIFY_EXPECTED_STEPS=${VERIFY_EXPECTED_STEPS:-$EXPECTED_COMPLETED_STEPS}
[[ "$VERIFY_EXPECTED_STEPS" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ -f "$MODEL_PATH/merge_manifest.json" && -f "$DATASET_YAML" ]] || exit 3
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  [[ -d "$RESUME_FROM_CHECKPOINT" && "$RESUME_FROM_CHECKPOINT" == "$OUTPUT"/checkpoint-* ]] || exit 3
  export PATHVLM_RESUME_FROM_CHECKPOINT=$RESUME_FROM_CHECKPOINT
  # PyTorch 2.6 changed torch.load's default to weights_only=True. DeepSpeed's
  # locally generated optimizer state legitimately contains its LossScaler and
  # ZeRO metadata classes, so a resumable checkpoint cannot be read under that
  # default. This override is scoped only to a checkpoint already constrained
  # above to this run's own output directory; no external checkpoint is trusted.
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
else
  [[ ! -e "$OUTPUT" ]] || exit 3
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU compute process is active" >&2
  exit 4
fi
mkdir -p "$RUN_DIR"
export PATHVLM_ALLOW_LANGUAGE_LORA=true
export PATHVLM_REWARD_LOG_DIR=$RUN_DIR/$REWARD_LOG_SUBDIR
export PATHVLM_TRAINING_SEGMENT=${TRAINING_SEGMENT:-formal_selected_sft3000_rule_rl1000_${MODE}}
export PATHVLM_TRAIN_STATE_AUDIT_NAME=pathvlm_train_state_audit.json
export PATHVLM_REQUIRE_SOURCE_AUDIT=true
export PATHVLM_IMAGE_HASH_MANIFEST=$REPO/data/pathmmu_image_disjoint_v2/image_content_sha256.json
export DEBUG_MODE=false
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATHVLM_FORMAT_REWARD_WEIGHT=$FORMAT_REWARD_WEIGHT
export PATHVLM_GENERATION_TEMPERATURE=$GENERATION_TEMPERATURE
export PATHVLM_GENERATION_TOP_P=$GENERATION_TOP_P
export PATHVLM_GENERATION_TOP_K=$GENERATION_TOP_K
export PATHVLM_GENERATION_TYPICAL_P=$GENERATION_TYPICAL_P
export PATHVLM_GENERATION_REPETITION_PENALTY=$GENERATION_REPETITION_PENALTY
if [[ "$MODE" == smoke ]]; then
  export PATHVLM_SKIP_FINAL_MODEL_SAVE=true
fi

"$GRPO_PYTHON" - "$RUN_DIR/$LAUNCH_CONTRACT_NAME" <<PY
import hashlib, json, pathlib
path = pathlib.Path("$RUN_DIR/$LAUNCH_CONTRACT_NAME")
def digest(value):
    h = hashlib.sha256()
    with open(value, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
payload = {
    "schema_version": 1,
    "mode": "$MODE",
    "run_role": "$RUN_ROLE",
    "model_path": str(pathlib.Path("$MODEL_PATH").resolve()),
    "merge_manifest_sha256": digest("$MODEL_PATH/merge_manifest.json"),
    "dataset_yaml": str(pathlib.Path("$DATASET_YAML").resolve()),
    "dataset_yaml_sha256": digest("$DATASET_YAML"),
    "deepspeed_config": str(pathlib.Path("$DEEPSPEED_CONFIG").resolve()),
    "deepspeed_config_sha256": digest("$DEEPSPEED_CONFIG"),
    "per_device_train_batch_size": int("$PER_DEVICE_BATCH"),
    "world_size": 8,
    "num_generations": 4,
    "max_completion_length": int("$MAX_COMPLETION_LENGTH"),
    "unique_prompts_per_optimizer_step": 2 * int("$PER_DEVICE_BATCH"),
    "steps_per_epoch": int("$STEPS_PER_EPOCH"),
    "equivalent_epochs": (int("$MAX_STEPS") / int("$STEPS_PER_EPOCH")),
    "max_steps": int("$MAX_STEPS"),
    "expected_completed_steps": int("$EXPECTED_COMPLETED_STEPS"),
    "stop_after_saved_step": int("$STOP_AFTER_SAVED_STEP") or None,
    "learning_rate": float("$LEARNING_RATE"),
    "gradient_checkpointing": "$GRADIENT_CHECKPOINTING" == "true",
    "reward_mode": "$REWARD_MODE",
    "reward_functions": "${REWARD_TYPES[*]}".split(),
    "format_reward_weight": float("$FORMAT_REWARD_WEIGHT") if "$REWARD_MODE" == "accuracy_scaled_format" else None,
    "sampling_contract": {
        "do_sample": True,
        "temperature": float("$GENERATION_TEMPERATURE"),
        "top_p": float("$GENERATION_TOP_P"),
        "top_k": int("$GENERATION_TOP_K"),
        "top_k_disabled": int("$GENERATION_TOP_K") == 0,
        "typical_p": float("$GENERATION_TYPICAL_P"),
        "repetition_penalty": float("$GENERATION_REPETITION_PENALTY"),
        "max_new_tokens": int("$MAX_COMPLETION_LENGTH"),
        "stop_contract": "first_eos_or_max_new_tokens",
    },
    "explicit_save_steps": [int(value) for value in "$EXPLICIT_SAVE_STEPS".split(",") if value],
    "resume_from_checkpoint": "$RESUME_FROM_CHECKPOINT" or None,
    "trusted_local_resume_torch_weights_only_override": bool("$RESUME_FROM_CHECKPOINT"),
    "judge_llm_used": False,
    "vision_encoder": "frozen",
    "multimodal_projector": "frozen",
    "trainable_parameters": "language LoRA only",
    "lora_r": int("$LORA_R"),
    "lora_alpha": int("$LORA_ALPHA"),
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

"$GRPO_PYTHON" -m torch.distributed.run --nproc_per_node=8 --master_port="$MASTER_PORT" \
  "$REPO/scripts/grpo_pathmmu.py" \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --output_dir "$OUTPUT" --model_name_or_path "$MODEL_PATH" \
  --dataset_name "$DATASET_YAML" --image_root / \
  --reward_funcs "${REWARD_ARGS[@]}" --freeze_vision_modules true \
  --max_pixels 65536 --min_pixels 3136 \
  --num_generations 4 --max_completion_length "$MAX_COMPLETION_LENGTH" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH" --gradient_accumulation_steps 1 \
  --learning_rate "$LEARNING_RATE" --logging_steps 1 \
  --bf16 true --torch_dtype bfloat16 --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
  --attn_implementation sdpa --beta 0.04 --num_iterations 1 --temperature "$GENERATION_TEMPERATURE" \
  --use_peft true --lora_r "$LORA_R" --lora_alpha "$LORA_ALPHA" --lora_dropout 0.05 \
  --save_strategy "$SAVE_STRATEGY" --save_steps "$SAVE_STEPS" --save_total_limit "$SAVE_LIMIT" \
  --save_only_model false --report_to none --seed 42 --data_seed 42 \
  --remove_unused_columns false --max_steps "$MAX_STEPS" \
  2>&1 | tee "$RUN_DIR/$TRAIN_LOG_NAME"

"$GRPO_PYTHON" "$REPO/scripts/verify_grpo_reward_alignment.py" \
  --events-dir "$PATHVLM_REWARD_LOG_DIR" \
  --dataset-json "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/pathmmu_image_disjoint_v2/grpo/pathvlm_rl_n1000.json" \
  --expected-steps "$VERIFY_EXPECTED_STEPS" \
  --world-size 8 \
  --per-device-batch "$PER_DEVICE_BATCH" \
  --num-generations 4 \
  --reward-types "${REWARD_TYPES[@]}" \
  --format-reward-weight "$FORMAT_REWARD_WEIGHT" \
  --output "$RUN_DIR/$ALIGNMENT_OUTPUT_NAME"
