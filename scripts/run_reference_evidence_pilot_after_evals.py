#!/usr/bin/env python3
"""Queue decision-score validation and reference-deletion pilot after n8 evals."""
import json, os, subprocess, time
from pathlib import Path

REPO=Path('/home/dataset-assist-0/czy/wjy/myr1')
PY=Path('/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/grpo/bin/python')
MODEL=Path('/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/full_language_rule_rl_clean_n4_n8_step1000_20260812/n4_fresh_step1000/model_snapshots/checkpoint-1000')
PANEL=REPO/'protocol/predefined_evidence_selected_panel_v1_20260812.json'
EVAL_STATE=Path('/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/full_language_rule_rl_clean_n4_n8_step1000_20260812/candidate_ood_eval/complete_summary.json')
OUT=Path('/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/reference_evidence_causal_pilot_n4_step1000_20260812')

def gpu_pids():
 return subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
def main():
 while not EVAL_STATE.is_file() or gpu_pids(): time.sleep(300)
 if OUT.exists(): raise FileExistsError(OUT)
 OUT.mkdir(parents=True); env=os.environ.copy(); env.update({'CUDA_VISIBLE_DEVICES':'0','PYTHONPATH':str(REPO/'scripts'),'TOKENIZERS_PARALLELISM':'false','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'})
 jobs=[
  ('decision_score',[str(PY),str(REPO/'scripts/run_decision_score_agreement.py'),'--model',str(MODEL),'--panel',str(PANEL),'--output-dir',str(OUT/'decision_score')]),
  ('reference_deletion',[str(PY),str(REPO/'scripts/run_reference_evidence_behavior.py'),'--model',str(MODEL),'--panel',str(PANEL),'--output-dir',str(OUT/'reference_deletion')]),
 ]
 state={'schema_version':1,'status':'running','jobs':[]}; (OUT/'state.json').write_text(json.dumps(state,indent=2)+'\n')
 for name,cmd in jobs:
  with (OUT/f'{name}.log').open('w') as log: code=subprocess.run(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
  ok=code==0 and (OUT/name/'metrics.json').is_file(); state['jobs'].append({'name':name,'returncode':code,'status':'completed' if ok else 'failed'}); (OUT/'state.json').write_text(json.dumps(state,indent=2)+'\n')
  if not ok: state['status']='failed'; (OUT/'state.json').write_text(json.dumps(state,indent=2)+'\n'); raise SystemExit(code or 1)
 state['status']='completed'; (OUT/'state.json').write_text(json.dumps(state,indent=2)+'\n')
if __name__=='__main__': main()
