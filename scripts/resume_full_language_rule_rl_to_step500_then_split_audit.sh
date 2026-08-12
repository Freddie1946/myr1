#!/usr/bin/env bash
set -euo pipefail

REPO=/home/dataset-assist-0/czy/wjy/myr1
PYTHON=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/grpo/bin/python
RUN=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/full_language_rule_rl_capacity_20260811/formal_g2_lr1_step100
OUTPUT=$RUN/output
CHECKPOINT250=$OUTPUT/checkpoint-250
CHECKPOINT500=$OUTPUT/checkpoint-500
SNAPSHOTS=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/full_language_rule_rl_capacity_20260811/model_snapshots
SNAPSHOT250=$SNAPSHOTS/checkpoint-250
SNAPSHOT500=$SNAPSHOTS/checkpoint-500
DATASET_YAML=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/pathmmu_image_disjoint_v2/grpo/pathvlm_rl_n1000.yaml
DATASET_JSON=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/pathmmu_image_disjoint_v2/grpo/pathvlm_rl_n1000.json
DEEPSPEED=$REPO/configs/deepspeed/ds_z3_gpu_torch_adamw.json
ATTEMPT=attempt01
REWARD_EVENTS=$RUN/online_reward_events_resume250_500_${ATTEMPT}
AUDIT_NAME=pathvlm_train_state_audit_resume250_500_${ATTEMPT}.json
STATE=$RUN/continuation_to500_${ATTEMPT}_state.json
EVAL=/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/full_language_rule_rl_capacity_20260811/step500_gate
SPLIT_AUDIT=/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/pathmmu_split_passk_audit_20260812

[[ ! -e "$STATE" ]] || { echo "continuation state already exists: $STATE" >&2; exit 3; }
[[ -f "$CHECKPOINT250/trainer_state.json" && -d "$CHECKPOINT250/global_step250" ]] || exit 4
[[ -f "$SNAPSHOT250/model.safetensors.index.json" ]] || exit 5
[[ ! -e "$CHECKPOINT500" && ! -e "$SNAPSHOT500" && ! -e "$EVAL" ]] || exit 6
[[ ! -e "$REWARD_EVENTS" && ! -e "$SPLIT_AUDIT" ]] || exit 7
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU compute process is active" >&2
  exit 8
fi

state() {
  local status=$1 stage=$2
  "$PYTHON" - "$STATE" "$status" "$stage" <<'PY'
import json,os,sys
from datetime import datetime,timezone
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
  "schema_version":1,"status":sys.argv[2],"stage":sys.argv[3],
  "updated_at":datetime.now(timezone.utc).isoformat(),"pid":os.getppid(),
  "automatic_training_retry":False,"split_audit_retry_limit":2,
  "resume_optimizer_and_scheduler":True,"test_used_for_continuation":False,
},indent=2,sort_keys=True)+"\n")
PY
}
CURRENT_STAGE=preflight
trap 'rc=$?; state failed "${CURRENT_STAGE:-unknown}"; exit "$rc"' ERR
state running "$CURRENT_STAGE"

"$PYTHON" - "$RUN/resume250_500_${ATTEMPT}_contract.json" "$CHECKPOINT250" "$DEEPSPEED" <<'PY'
import hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
p,checkpoint,deepspeed=map(Path,sys.argv[1:])
p.write_text(json.dumps({
 "schema_version":1,"created_at":datetime.now(timezone.utc).isoformat(),
 "resume_from":str(checkpoint.resolve()),"start_step":250,"stop_step":500,
 "scheduler_horizon_steps":500,"new_prompt_exposure":500,"new_trajectories":2000,
 "cumulative_prompt_exposure":1000,"cumulative_trajectories":4000,
 "world_size":8,"per_device_batch":1,"prompts_per_update":2,"num_generations":4,
 "learning_rate":1e-6,"beta":0.04,"num_iterations":1,"reward":"accuracy_only",
 "temperature":0.9,"top_p":1.0,"top_k":0,"max_completion_length":384,
 "gradient_checkpointing":True,"vision_encoder":"frozen","projector":"frozen",
 "language_model":"full_parameter","deepspeed_sha256":sha(deepspeed),
 "checkpoint_trust":"locally_created_same_run",
 "torch_load_compatibility":"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1",
 "test_accessed_for_continuation":False,
 "post_step500_action":"fixed evaluation then predeclared three-model split pass@8 audit",
},indent=2,sort_keys=True)+"\n")
PY

CURRENT_STAGE=train_resume250_to500
state running "$CURRENT_STAGE"
unset PATHVLM_ALLOW_LANGUAGE_LORA
export PATHVLM_EXPLICIT_SAVE_STEPS=500
export PATHVLM_STOP_AFTER_SAVED_STEP=500
export PATHVLM_SKIP_FINAL_MODEL_SAVE=true
export PATHVLM_REWARD_LOG_DIR=$REWARD_EVENTS
export PATHVLM_TRAINING_SEGMENT=full_language_rule_rl_capacity_resume250_500_attempt01
export PATHVLM_TRAIN_STATE_AUDIT_NAME=$AUDIT_NAME
export PATHVLM_REQUIRE_SOURCE_AUDIT=true
export PATHVLM_IMAGE_HASH_MANIFEST=$REPO/data/pathmmu_image_disjoint_v2/image_content_sha256.json
export PATHVLM_GENERATION_TEMPERATURE=0.9
export PATHVLM_GENERATION_TOP_P=1.0
export PATHVLM_GENERATION_TOP_K=0
export PATHVLM_GENERATION_TYPICAL_P=1.0
export PATHVLM_GENERATION_REPETITION_PENALTY=1.0
export PATHVLM_RESUME_FROM_CHECKPOINT=$CHECKPOINT250
export PATHVLM_EPOCH_SNAPSHOT_STEPS=500
export PATHVLM_EPOCH_SNAPSHOT_DIR=$SNAPSHOTS
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false DEBUG_MODE=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON" -m torch.distributed.run --nproc_per_node=8 --master_port=34213 \
  "$REPO/scripts/grpo_pathmmu.py" \
  --deepspeed "$DEEPSPEED" --output_dir "$OUTPUT" \
  --model_name_or_path "$SNAPSHOT250" --dataset_name "$DATASET_YAML" --image_root / \
  --reward_funcs accuracy --freeze_vision_modules true \
  --max_pixels 65536 --min_pixels 3136 --num_generations 4 --max_completion_length 384 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --learning_rate 1.0e-6 --logging_steps 1 --bf16 true --torch_dtype bfloat16 \
  --gradient_checkpointing true --attn_implementation sdpa \
  --beta 0.04 --num_iterations 1 --temperature 0.9 \
  --save_strategy no --save_steps 100 --save_total_limit 1 --save_only_model false \
  --report_to none --seed 42 --data_seed 42 --remove_unused_columns false --max_steps 500 \
  2>&1 | tee "$RUN/train_resume250_500_${ATTEMPT}.log"

CURRENT_STAGE=verify_full_epoch_coverage
state running "$CURRENT_STAGE"
"$PYTHON" "$REPO/scripts/verify_grpo_reward_alignment.py" \
  --events-dir "$REWARD_EVENTS" --dataset-json "$DATASET_JSON" \
  --expected-steps 250 --world-size 8 --per-device-batch 1 --num-generations 4 \
  --reward-types accuracy --format-reward-weight 0.1 \
  --output "$RUN/reward_alignment_verification_resume250_500_${ATTEMPT}.json"

"$PYTHON" - "$OUTPUT/$AUDIT_NAME" "$SNAPSHOT500" "$RUN" <<'PY'
import json,math,sys
from pathlib import Path
audit_path,snapshot,run=map(Path,sys.argv[1:])
audit=json.loads(audit_path.read_text())
if audit['global_step'] != 500 or audit['resume_from_checkpoint'] is None:
    raise RuntimeError('resume audit did not reach step500 from a checkpoint')
t=audit['trainability']; p=t['parameters']
if not (t['passed'] and t['language_mode']=='full'
        and p['language']['trainable']==p['language']['total']
        and p['visual']['trainable']==0 and p['multimodal_projector']['trainable']==0):
    raise RuntimeError('trainability/freeze gate failed')
rows=[r for r in audit['log_history'] if 251 <= r.get('step',-1) <= 500 and 'grad_norm' in r]
if len(rows)!=250 or not all(math.isfinite(float(r['grad_norm'])) and float(r['grad_norm'])>0 for r in rows):
    raise RuntimeError('resume gradients are incomplete or invalid')
if not (snapshot/'model.safetensors.index.json').is_file():
    raise RuntimeError('step500 model-only snapshot is missing')
event_dirs=[
 run/'online_reward_events',
 run/'online_reward_events_resume100_250_attempt02',
 run/'online_reward_events_resume250_500_attempt01',
]
sets=[]
expected=[200,300,500]
for directory,want in zip(event_dirs,expected,strict=True):
    events=[]
    for path in directory.glob('rank_*.jsonl'):
        events.extend(json.loads(line) for line in path.open() if line.strip())
    indices={int(event['record_index']) for event in events}
    if len(indices)!=want or len(events)!=want*4:
        raise RuntimeError(f'coverage mismatch in {directory}: {len(indices)} records/{len(events)} events')
    sets.append(indices)
if any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)):
    raise RuntimeError('prompt exposure segments overlap before completing one epoch')
if set.union(*sets) != set(range(1000)):
    raise RuntimeError('step500 does not cover all frozen RL1000 records exactly once')
(run/'full_epoch_prompt_coverage_verification.json').write_text(json.dumps({
    'schema_version':1,'status':'passed','unique_prompt_count':1000,
    'segment_unique_counts':expected,'pairwise_overlap_count':0,
    'record_index_min':0,'record_index_max':999,
},indent=2,sort_keys=True)+'\n')
PY

CURRENT_STAGE=evaluate_step500
state running "$CURRENT_STAGE"
"$PYTHON" "$REPO/scripts/run_full_language_rule_rl_step100_evaluation.py" \
  --model "$SNAPSHOT500" --output "$EVAL"

CURRENT_STAGE=three_model_split_audit
state running "$CURRENT_STAGE"
"$PYTHON" "$REPO/scripts/run_pathmmu_split_audit_sequence.py" --output "$SPLIT_AUDIT"

CURRENT_STAGE=complete
state completed "$CURRENT_STAGE"
