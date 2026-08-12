#!/usr/bin/env bash
set -euo pipefail

TRAIN_ROOT=${1:-/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/visual_adaptation_followup_n1500_e3_2seed_20260811}
EVAL_ROOT=${2:-/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/visual_adaptation_followup_n1500_e3_2seed_20260811}
REPO=/home/dataset-assist-0/czy/wjy/myr1
PYTHON=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/sft/bin/python
MODEL=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/models/Qwen2.5-VL-7B-Instruct-cc594898137f460bfe9f0759e9844b3ce807cfb5
TRAIN_PROBE=/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/visual_adaptation_followup_train_probe500_20260811/records_n0500.json
PATHVQA=/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/pathvqa_architecture_validation_v1_20260811/pathvqa_validation_balanced_image_unique_512.json
MMMU=/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/mmmu_nonmedical_dev_retention_v1_20260811/panel.json

[[ -f "$EVAL_ROOT/pathmmu_funnel_summary.json" ]] || {
  echo "PathMMU funnel is not complete" >&2
  exit 2
}
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
  echo "GPU compute process is active" >&2
  exit 3
fi
mkdir -p "$EVAL_ROOT/train_probe" "$EVAL_ROOT/pathvqa" "$EVAL_ROOT/mmmu" "$EVAL_ROOT/logs"

adapter_for() {
  local id=$1 arm seed_text step_text step
  IFS=_ read -r arm seed_text step_text <<<"$id"
  step=$((10#${step_text#step}))
  printf '%s\n' "$TRAIN_ROOT/${arm}_${seed_text}/output/checkpoint-$step"
}

run_job() {
  local spec=$1 gpu=$2 kind=${1%%|*} id=${1#*|} adapter output
  adapter=$(adapter_for "$id")
  case "$kind" in
    train)
      output=$EVAL_ROOT/train_probe/$id
      [[ -f "$output/metrics.json" ]] && return 0
      [[ ! -e "$output" ]] || return 10
      CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO/scripts "$PYTHON" \
        "$REPO/scripts/run_pathmmu_qwen_diagnostic.py" \
        --model "$MODEL" --adapter "$adapter" --backend qwen2_5_vl \
        --data "$TRAIN_PROBE" --output-dir "$output" --batch-size 8 \
        --split-role train_memorization_probe --max-new-tokens 1024 \
        >"$EVAL_ROOT/logs/${id}_train_probe.log" 2>&1
      ;;
    original|cyclic_mismatch|global_mean_blank)
      output=$EVAL_ROOT/pathvqa/$id/$kind
      [[ -f "$output/metrics.json" ]] && return 0
      [[ ! -e "$output" ]] || return 11
      CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO/scripts "$PYTHON" \
        "$REPO/scripts/run_pathvqa_forced_binary_logits.py" \
        --model "$MODEL" --adapter "$adapter" --data "$PATHVQA" \
        --output-dir "$output" --batch-size 32 --split-role validation_diagnostic \
        --image-mode "$kind" >"$EVAL_ROOT/logs/${id}_pathvqa_${kind}.log" 2>&1
      ;;
    mmmu)
      output=$EVAL_ROOT/mmmu/$id
      [[ -f "$output/metrics.json" ]] && return 0
      [[ ! -e "$output" ]] || return 12
      CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$REPO/scripts "$PYTHON" \
        "$REPO/scripts/run_mmmu_retention_diagnostic.py" \
        --model "$MODEL" --adapter "$adapter" --panel "$MMMU" \
        --output-dir "$output" --batch-size 4 --max-new-tokens 4096 \
        >"$EVAL_ROOT/logs/${id}_mmmu.log" 2>&1
      ;;
    *) return 13 ;;
  esac
}

dispatch() {
  local -n jobs_ref=$1
  local start offset failed
  for ((start=0; start<${#jobs_ref[@]}; start+=8)); do
    pids=()
    for ((offset=0; offset<8 && start+offset<${#jobs_ref[@]}; offset++)); do
      run_job "${jobs_ref[$((start+offset))]}" "$offset" &
      pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    ((failed == 0)) || return 1
  done
}

mapfile -t shortlist < <("$PYTHON" - "$EVAL_ROOT/pathmmu_funnel_summary.json" <<'PY'
import json,sys
for value in json.load(open(sys.argv[1]))["shortlist"]:
    print(value)
PY
)
phase2a=()
for id in "${shortlist[@]}"; do
  phase2a+=("train|$id" "original|$id")
done
dispatch phase2a

"$PYTHON" - "$EVAL_ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
phase1=json.loads((root/'pathmmu_funnel_summary.json').read_text())
by_id={row['id']:row for row in phase1['rows']}
rows=[]
for name in phase1['shortlist']:
    val=by_id[name]
    train=json.loads((root/'train_probe'/name/'metrics.json').read_text())
    pqa=json.loads((root/'pathvqa'/name/'original'/'metrics.json').read_text())
    row={**val,'train_accuracy':train['accuracy'],'train_val_gap':train['accuracy']-val['accuracy'],
         'pathvqa_accuracy':pqa['accuracy']}
    row['preliminary_eligible']=(row['train_val_gap'] <= .20+1e-12 and
                                 row['pathvqa_accuracy'] >= .6402-1e-12)
    rows.append(row)
payload={'schema_version':1,'status':'retention_phase_a_completed','rows':rows,
         'preliminary_shortlist':[row['id'] for row in rows if row['preliminary_eligible']]}
(root/'retention_phase_a_summary.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY

mapfile -t preliminary < <("$PYTHON" - "$EVAL_ROOT/retention_phase_a_summary.json" <<'PY'
import json,sys
for value in json.load(open(sys.argv[1]))["preliminary_shortlist"]:
    print(value)
PY
)
if [[ ${#preliminary[@]} -eq 0 ]]; then
  echo "no checkpoint passed preliminary retention filters" >&2
  exit 6
fi
phase2b=()
for id in "${preliminary[@]}"; do
  phase2b+=("cyclic_mismatch|$id" "global_mean_blank|$id" "mmmu|$id")
done
dispatch phase2b

"$PYTHON" - "$EVAL_ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
phase_a=json.loads((root/'retention_phase_a_summary.json').read_text())
rows=[]
for source in phase_a['rows']:
    row=dict(source)
    if not row['preliminary_eligible']:
        row.update({'delta_vision':None,'delta_blank':None,'mmmu_accuracy':None,'eligible':False})
    else:
        original=json.loads((root/'pathvqa'/row['id']/'original'/'metrics.json').read_text())
        shuffle=json.loads((root/'pathvqa'/row['id']/'cyclic_mismatch'/'metrics.json').read_text())
        blank=json.loads((root/'pathvqa'/row['id']/'global_mean_blank'/'metrics.json').read_text())
        mmmu=json.loads((root/'mmmu'/row['id']/'metrics.json').read_text())
        row['delta_vision']=original['accuracy']-shuffle['accuracy']
        row['delta_blank']=original['accuracy']-blank['accuracy']
        row['mmmu_accuracy']=mmmu['accuracy']
        row['mmmu_generation_cap_hit_count']=mmmu['generation_cap_hit_count']
        row['eligible']=(row['delta_vision'] >= .13-1e-12 and row['mmmu_accuracy'] >= .499-1e-12)
    rows.append(row)

best={}
for arm in ('l','a'):
    for seed in (42,):
        group=[row for row in rows if row['arm']==arm and row['seed']==seed and row['eligible']]
        if group:
            best[(arm,seed)]=max(group,key=lambda row:(row['accuracy'],-row['step']))

choose_a=False
reasons=[]
if all((arm,42) in best for arm in ('l','a')):
    path_diff=best[('a',42)]['accuracy']-best[('l',42)]['accuracy']
    pqa_diff=best[('a',42)]['pathvqa_accuracy']-best[('l',42)]['pathvqa_accuracy']
    mmmu_diff=best[('a',42)]['mmmu_accuracy']-best[('l',42)]['mmmu_accuracy']
    vision_diff=best[('a',42)]['delta_vision']-best[('l',42)]['delta_vision']
    choose_a=(path_diff >= .02-1e-12 and
              pqa_diff >= -.01-1e-12 and mmmu_diff >= -.01-1e-12 and vision_diff >= -.01-1e-12)
    reasons=[f'pathmmu_diff={path_diff:.6f}',
             f'pathvqa_diff={pqa_diff:.6f}',f'mmmu_diff={mmmu_diff:.6f}',f'delta_vision_diff={vision_diff:.6f}']
else:
    reasons=['A requires eligible seed42 checkpoints for both arms; condition not met']
selected='a' if choose_a else 'l'
if (selected,42) not in best:
    raise SystemExit(f'selected fallback arm {selected} lacks an eligible seed42 checkpoint')
payload={'schema_version':1,'status':'architecture_decision_completed','rows':rows,
         'best_eligible_by_arm_seed':{f'{a}_seed{s}':r for (a,s),r in best.items()},
         'selected_architecture':selected,'decision_reasons':reasons}
(root/'architecture_decision.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
