#!/usr/bin/env bash
set -euo pipefail

: "${NPROC_PER_NODE:?set NPROC_PER_NODE}"
: "${MASTER_PORT:?set MASTER_PORT}"
: "${MODEL_PATH:?set MODEL_PATH to the newly trained formal SFT checkpoint}"
: "${DATASET_YAML:?set DATASET_YAML to the frozen RL split YAML}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${RUN_DIR:?set RUN_DIR}"
: "${EXPERIMENT_ROOT:?set EXPERIMENT_ROOT}"
: "${GRPO_PYTHON:?set GRPO_PYTHON to the pinned GRPO venv python}"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export DEBUG_MODE=true
export LOG_PATH="$RUN_DIR/online_rewards.jsonl"

"$GRPO_PYTHON" -m torch.distributed.run \
  --nproc_per_node="$NPROC_PER_NODE" \
  --master_port="$MASTER_PORT" \
  "$EXPERIMENT_ROOT/scripts/grpo_pathmmu.py" \
  --deepspeed "$EXPERIMENT_ROOT/configs/deepspeed/ds_z3_optimizer_offload_torch_adamw.json" \
  --output_dir "$OUTPUT_DIR" \
  --model_name_or_path "$MODEL_PATH" \
  --dataset_name "$DATASET_YAML" \
  --image_root / \
  --reward_funcs accuracy format \
  --freeze_vision_modules true \
  --max_pixels 65536 \
  --min_pixels 3136 \
  --num_generations 4 \
  --max_completion_length 192 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1.0e-6 \
  --bf16 true \
  --torch_dtype bfloat16 \
  --gradient_checkpointing true \
  --attn_implementation sdpa \
  --beta 0.04 \
  --num_iterations 1 \
  --logging_steps 1 \
  --save_steps 100 \
  --save_only_model false \
  --report_to none \
  --seed "${SEED:-42}" \
  --data_seed "${SEED:-42}" \
  2>&1 | tee "$RUN_DIR/train.log"

