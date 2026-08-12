#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH to the merged SFT parent}"
: "${RUN_DIR:?set RUN_DIR}"
MODE=${MODE:-smoke}
[[ "$MODE" == smoke || "$MODE" == formal ]] || exit 2
RUN_ROLE=${RUN_ROLE:-full_language_rule_rl_capacity_${MODE}}
MAX_STEPS=${MAX_STEPS:-500}
EXPLICIT_SAVE_STEPS=${EXPLICIT_SAVE_STEPS:-2}
STOP_AFTER_SAVED_STEP=${STOP_AFTER_SAVED_STEP:-2}
LEARNING_RATE=${LEARNING_RATE:-1.0e-6}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-384}
MASTER_PORT=${MASTER_PORT:-34201}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-/home/dataset-assist-0/czy/wjy/myr1/configs/deepspeed/ds_z3_gpu_torch_adamw.json}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}
SAVE_LIMIT=${SAVE_LIMIT:-2}

[[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$EXPLICIT_SAVE_STEPS" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || exit 2
[[ "$STOP_AFTER_SAVED_STEP" =~ ^[1-9][0-9]*$ ]] || exit 2
(( STOP_AFTER_SAVED_STEP <= MAX_STEPS )) || exit 2
[[ ",$EXPLICIT_SAVE_STEPS," == *",$STOP_AFTER_SAVED_STEP,"* ]] || exit 2
[[ "$MAX_COMPLETION_LENGTH" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ "$GRADIENT_CHECKPOINTING" == true || "$GRADIENT_CHECKPOINTING" == false ]] || exit 2
[[ -f "$DEEPSPEED_CONFIG" ]] || exit 3
[[ -f "$MODEL_PATH/merge_manifest.json" ]] || exit 3

REPO=/home/dataset-assist-0/czy/wjy/myr1
PYTHON=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/grpo/bin/python
DATASET_YAML=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/pathmmu_image_disjoint_v2/grpo/pathvlm_rl_n1000.yaml
DATASET_JSON=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/pathmmu_image_disjoint_v2/grpo/pathvlm_rl_n1000.json
OUTPUT=$RUN_DIR/output
[[ ! -e "$OUTPUT" ]] || { echo "existing output: $OUTPUT" >&2; exit 4; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU compute process is active" >&2
  exit 5
fi
mkdir -p "$RUN_DIR"

unset PATHVLM_ALLOW_LANGUAGE_LORA
export PATHVLM_EXPLICIT_SAVE_STEPS=$EXPLICIT_SAVE_STEPS
export PATHVLM_STOP_AFTER_SAVED_STEP=$STOP_AFTER_SAVED_STEP
export PATHVLM_SKIP_FINAL_MODEL_SAVE=true
export PATHVLM_REWARD_LOG_DIR=$RUN_DIR/online_reward_events
export PATHVLM_TRAINING_SEGMENT=$RUN_ROLE
export PATHVLM_TRAIN_STATE_AUDIT_NAME=pathvlm_train_state_audit.json
export PATHVLM_REQUIRE_SOURCE_AUDIT=true
export PATHVLM_IMAGE_HASH_MANIFEST=$REPO/data/pathmmu_image_disjoint_v2/image_content_sha256.json
export PATHVLM_GENERATION_TEMPERATURE=0.9
export PATHVLM_GENERATION_TOP_P=1.0
export PATHVLM_GENERATION_TOP_K=0
export PATHVLM_GENERATION_TYPICAL_P=1.0
export PATHVLM_GENERATION_REPETITION_PENALTY=1.0
export DEBUG_MODE=false
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON" - "$RUN_DIR/launch_contract.json" <<PY
import hashlib,json,pathlib
def digest(value):
    h=hashlib.sha256()
    with open(value,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
payload={
  'schema_version':1,'mode':'$MODE','run_role':'$RUN_ROLE',
  'model_path':str(pathlib.Path('$MODEL_PATH').resolve()),
  'merge_manifest_sha256':digest('$MODEL_PATH/merge_manifest.json'),
  'dataset_yaml':str(pathlib.Path('$DATASET_YAML').resolve()),
  'dataset_yaml_sha256':digest('$DATASET_YAML'),
  'deepspeed_config':str(pathlib.Path('$DEEPSPEED_CONFIG').resolve()),
  'deepspeed_config_sha256':digest('$DEEPSPEED_CONFIG'),
  'world_size':8,'per_device_train_batch_size':1,'gradient_accumulation_steps':1,
  'num_generations':4,'unique_prompts_per_optimizer_step':2,
  'planned_scheduler_steps':int('$MAX_STEPS'),'expected_completed_steps':int('$STOP_AFTER_SAVED_STEP'),
  'prompt_exposure_at_stop':2*int('$STOP_AFTER_SAVED_STEP'),
  'explicit_save_steps':[int(x) for x in '$EXPLICIT_SAVE_STEPS'.split(',')],
  'learning_rate':float('$LEARNING_RATE'),'beta':0.04,'num_iterations':1,
  'gradient_checkpointing':'$GRADIENT_CHECKPOINTING'=='true',
  'reward_functions':['accuracy'],'reward_mode':'accuracy_only','judge_llm_used':False,
  'sampling_contract':{'do_sample':True,'temperature':0.9,'top_p':1.0,'top_k':0,
    'top_k_disabled':True,'typical_p':1.0,'repetition_penalty':1.0,
    'max_new_tokens':int('$MAX_COMPLETION_LENGTH'),'stop_contract':'first_eos_or_max_new_tokens'},
  'vision_encoder':'frozen','multimodal_projector':'frozen',
  'trainable_parameters':'full language model','use_peft':False,
  'test_accessed':False,
}
path=pathlib.Path('$RUN_DIR/launch_contract.json')
path.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY

"$PYTHON" -m torch.distributed.run --nproc_per_node=8 --master_port="$MASTER_PORT" \
  "$REPO/scripts/grpo_pathmmu.py" \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --output_dir "$OUTPUT" --model_name_or_path "$MODEL_PATH" \
  --dataset_name "$DATASET_YAML" --image_root / \
  --reward_funcs accuracy --freeze_vision_modules true \
  --max_pixels 65536 --min_pixels 3136 \
  --num_generations 4 --max_completion_length "$MAX_COMPLETION_LENGTH" \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --learning_rate "$LEARNING_RATE" --logging_steps 1 \
  --bf16 true --torch_dtype bfloat16 --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
  --attn_implementation sdpa --beta 0.04 --num_iterations 1 --temperature 0.9 \
  --save_strategy no --save_steps 100 --save_total_limit "$SAVE_LIMIT" \
  --save_only_model false --report_to none --seed 42 --data_seed 42 \
  --remove_unused_columns false --max_steps "$MAX_STEPS" \
  2>&1 | tee "$RUN_DIR/train.log"

"$PYTHON" "$REPO/scripts/verify_grpo_reward_alignment.py" \
  --events-dir "$PATHVLM_REWARD_LOG_DIR" --dataset-json "$DATASET_JSON" \
  --expected-steps "$STOP_AFTER_SAVED_STEP" --world-size 8 --per-device-batch 1 \
  --num-generations 4 --reward-types accuracy --format-reward-weight 0.1 \
  --output "$RUN_DIR/reward_alignment_verification.json"
