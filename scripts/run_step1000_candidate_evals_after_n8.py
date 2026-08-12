#!/usr/bin/env python3
"""Wait for n=8 step-1000, then run matched OOD/retention evaluations.

This continuation is intentionally non-destructive and does not inspect test
data for model selection beyond the already-authorized PathMMU test999
diagnostic. It uses three candidates: the merged SFT parent, full-RL n=4, and
full-RL n=8. PathVQA, MMMU and two OmniMed jobs are parallelized over GPUs;
the final OmniMed job runs after a GPU is released.
"""
from __future__ import annotations

import json, os, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/dataset-assist-0/czy/wjy/myr1")
PYTHON = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/grpo/bin/python")
RUNROOT = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/full_language_rule_rl_clean_n4_n8_step1000_20260812")
EVALROOT = Path("/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/full_language_rule_rl_clean_n4_n8_step1000_20260812/candidate_ood_eval")
STATE = RUNROOT / "sequence_state.json"
N8_PATHMMU_SUMMARY = Path("/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/full_language_rule_rl_clean_n4_n8_step1000_20260812/n8_fresh_step1000_pathmmu_eval/complete_summary.json")
PATHVQA_PANEL = Path("/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/pathvqa_architecture_validation_v1_20260811/pathvqa_validation_balanced_image_unique_512.json")
MMMU_PANEL = Path("/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/mmmu_nonmedical_dev_retention_v1_20260811/panel.json")
OMNI_DATA = Path("/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/external_vqa_contract_v1_20260729/omnimedvqa_four_sources_8518.json")


def now(): return datetime.now(timezone.utc).isoformat()
def write(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + f".tmp-{os.getpid()}")
    t.write_text(json.dumps(x, indent=2, sort_keys=True) + "\n")
    os.replace(t, p)
def gpu_pids():
    return [x.strip() for x in subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout.splitlines() if x.strip()]


def wait_for_training():
    while True:
        s = json.loads(STATE.read_text())
        if s.get("status") == "failed": raise RuntimeError(s)
        if s.get("status") == "completed" and not gpu_pids(): return
        time.sleep(300)


def wait_for_n8_pathmmu_eval():
    """Serialize with the separate n=8 PathMMU waiter to avoid GPU contention."""
    while not N8_PATHMMU_SUMMARY.is_file():
        time.sleep(300)


def candidates():
    return {
        "sft_parent": RUNROOT.parent / "formal_selected_rule_rl1000_20260811/parents/sft_step080_merged",
        "full_rl_n4_step1000": RUNROOT / "n4_fresh_step1000/model_snapshots/checkpoint-1000",
        "full_rl_n8_step1000": RUNROOT / "n8_fresh_step1000/model_snapshots/checkpoint-1000",
    }


def launch(name, gpu, cmd, out, jobs):
    out.mkdir(parents=True, exist_ok=True)
    log = (EVALROOT / "logs" / f"{name}.log").open("w")
    env = os.environ.copy(); env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(REPO / "scripts"), "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    jobs.append((name, p, log, out))


def main():
    EVALROOT.mkdir(parents=True, exist_ok=True); (EVALROOT / "logs").mkdir(exist_ok=True)
    write(EVALROOT / "sequence_state.json", {"schema_version":1,"status":"waiting_for_n8","started_at":now()})
    wait_for_training()
    wait_for_n8_pathmmu_eval()
    models = candidates()
    for p in [PATHVQA_PANEL, MMMU_PANEL, OMNI_DATA, *models.values()]:
        if not p.is_file() and not p.is_dir(): raise FileNotFoundError(p)
    write(EVALROOT / "sequence_state.json", {"schema_version":1,"status":"running","stage":"pathvqa_mmmu_omni","started_at":now(),"models":{k:str(v) for k,v in models.items()}})
    jobs=[]
    # PathVQA A/B-format diagnostic, one GPU per candidate.
    for i,(name,model) in enumerate(models.items()):
        out=EVALROOT / "pathvqa" / name
        launch("pathvqa_"+name,i,[str(PYTHON),str(REPO/"scripts/run_pathvqa_yesno_prompt_calibration.py"),"--model",str(model),"--panel",str(PATHVQA_PANEL),"--panel-role","architecture_validation_512","--output-root",str(out),"--batch-size","16","--prompt-contracts","domain_think_answer_v2_2048,pathmmu_ab_v1_2048"],out,jobs)
    # MMMU retention, one GPU per candidate.
    for i,(name,model) in enumerate(models.items(), start=3):
        out=EVALROOT / "mmmu" / name
        launch("mmmu_"+name,i,[str(PYTHON),str(REPO/"scripts/run_mmmu_retention_diagnostic.py"),"--model",str(model),"--panel",str(MMMU_PANEL),"--output-dir",str(out),"--batch-size","2","--max-new-tokens","4096"],out,jobs)
    # Two OmniMed jobs first; the third is launched after one slot releases.
    omni_names=list(models)
    for i,name in enumerate(omni_names[:2], start=6):
        out=EVALROOT / "omnimedvqa" / name
        launch("omni_"+name,i,[str(PYTHON),str(REPO/"scripts/run_external_vqa_qwen.py"),"--task","omnimedvqa","--model",str(models[name]),"--backend","qwen2_5_vl","--data",str(OMNI_DATA),"--output-dir",str(out),"--max-new-tokens","1024","--generation-contract","omnimed_domain_think_answer_v4_1024","--batch-size","16","--split-role","external_test"],out,jobs)
    # Collect first wave.
    failed=[]
    for item in jobs:
        name,p,log,out=item; code=p.wait(); log.close()
        if code!=0 or not (out/"metrics.json").is_file(): failed.append(name)
    if failed: raise RuntimeError(f"evaluation failures: {failed}")
    # Third OmniMed candidate on GPU6 after the first wave is complete.
    name=omni_names[2]; out=EVALROOT/"omnimedvqa"/name
    env=os.environ.copy(); env.update({"CUDA_VISIBLE_DEVICES":"6","PYTHONPATH":str(REPO/"scripts"),"TOKENIZERS_PARALLELISM":"false"})
    with (EVALROOT/"logs"/("omni_"+name+".log")).open("w") as log:
        code=subprocess.run([str(PYTHON),str(REPO/"scripts/run_external_vqa_qwen.py"),"--task","omnimedvqa","--model",str(models[name]),"--backend","qwen2_5_vl","--data",str(OMNI_DATA),"--output-dir",str(out),"--max-new-tokens","1024","--generation-contract","omnimed_domain_think_answer_v4_1024","--batch-size","16","--split-role","external_test"],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    if code!=0 or not (out/"metrics.json").is_file(): raise RuntimeError("third OmniMed evaluation failed")
    summary={"schema_version":1,"status":"completed","finished_at":now(),"models":{k:str(v) for k,v in models.items()},"evaluation_roots":str(EVALROOT)}
    write(EVALROOT/"complete_summary.json",summary); write(EVALROOT/"sequence_state.json",{**summary,"stage":"complete"})
    print(json.dumps(summary,sort_keys=True))

if __name__ == "__main__": main()
