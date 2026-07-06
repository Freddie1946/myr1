#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:?set PROJECT_ROOT to pathvlm_revision_core}"
: "${OPEN_R1_ROOT:?set OPEN_R1_ROOT to VLM-R1-main/src/open-r1-multimodal}"
: "${MODEL_PATH:?set MODEL_PATH}"
: "${PATHVLM_RL_TRAIN_JSON:?set PATHVLM_RL_TRAIN_JSON}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

export PYTHONPATH="$PROJECT_ROOT/src:$OPEN_R1_ROOT/src:${PYTHONPATH:-}"
export PATHVLM_ENABLE_GPT_REWARD="${PATHVLM_ENABLE_GPT_REWARD:-false}"

python "$PROJECT_ROOT/src/pathvlm_core/grpo_pathmmu.py" \
  --dataset_name "$PROJECT_ROOT/configs/pathmmu_grpo.yaml" \
  --model_name_or_path "$MODEL_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --reward_funcs ${REWARD_FUNCS:-accuracy format} \
  --num_generations ${NUM_GENERATIONS:-4} \
  --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE:-1} \
  --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS:-1} \
  --logging_steps ${LOGGING_STEPS:-1} \
  --bf16 \
  --torch_dtype bfloat16 \
  --report_to none \
  --save_steps ${SAVE_STEPS:-100} \
  --save_only_model true
