#!/usr/bin/env python3
"""Back up verified n=4/n=8 step-1000 model-only snapshots to private HF repos."""
from __future__ import annotations

import io, json, os, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/full_language_rule_rl_clean_n4_n8_step1000_20260812")
QUEUE = ROOT / "sequence_state.json"
STATE_DIR = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/reports/n4_n8_hf_backup_20260812")
STATE = STATE_DIR / "state.json"
HF = "/usr/local/bin/hf"
SPECS = (
    ("n4", ROOT / "n4_fresh_step1000/model_snapshots/checkpoint-1000", ROOT / "n4_fresh_step1000/completion_verification.json", "Freddie1946/PathVLM-R1-FullRuleRL-n4-step1000-seed42"),
    ("n8", ROOT / "n8_fresh_step1000/model_snapshots/checkpoint-1000", ROOT / "n8_fresh_step1000/completion_verification.json", "Freddie1946/PathVLM-R1-FullRuleRL-n8-step1000-seed42"),
)

def now(): return datetime.now(timezone.utc).isoformat()
def write(x):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    t=STATE.with_suffix(f".tmp-{os.getpid()}"); t.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n"); os.replace(t,STATE)

def wait_for_all():
    while True:
        q=json.loads(QUEUE.read_text())
        if q.get("status")=="failed": raise RuntimeError(f"training failed: {q}")
        if q.get("status")=="completed" and all(p.is_file() and (s/"model.safetensors.index.json").is_file() and (s/"snapshot_manifest.json").is_file() for _,s,p,_ in SPECS):
            return
        time.sleep(300)

def card(name, repo):
    return f"""---
license: apache-2.0
base_model: Qwen/Qwen2.5-VL-7B-Instruct
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- multimodal
- pathology
- grpo
- research
---

# {repo.split('/',1)[1]}

Verified model-only snapshot for the PathVLM-R1 n={name[1:]} full-language
rule-GRPO step-1000 comparison. Vision encoder and multimodal projector were
frozen; the language model was updated. This is a research checkpoint and is
not intended for clinical use.

- Seed: 42
- Optimizer/RNG/ZeRO state: not archived in this repository
- Completion and evaluation records: private evaluation archive
"""

def main():
    state={"schema_version":1,"status":"waiting_for_training","started_at":now(),"models":{}}
    write(state); wait_for_all(); api=HfApi()
    for name,snapshot,verification,repo in SPECS:
        state["models"][name]={"repo_id":repo,"status":"uploading","source":str(snapshot),"started_at":now()}; state["status"]="uploading"; write(state)
        api.create_repo(repo,repo_type="model",private=True,exist_ok=True)
        api.update_repo_settings(repo_id=repo,repo_type="model",private=True)
        api.upload_file(path_or_fileobj=io.BytesIO(card(name,repo).encode()),path_in_repo="README.md",repo_id=repo,repo_type="model",commit_message="Add private research model card")
        cmd=[HF,"upload-large-folder",repo,str(snapshot),"--repo-type","model","--num-workers","2","--no-bars","--private"]
        code=subprocess.run(cmd,cwd=snapshot).returncode
        if code:
            state["status"]="failed"; state["models"][name].update(status="failed",returncode=code,finished_at=now()); write(state); raise SystemExit(code)
        info=api.model_info(repo,files_metadata=True)
        remote={x.rfilename for x in info.siblings or []}
        required={"model.safetensors.index.json","snapshot_manifest.json","config.json"}
        if not required <= remote or not info.private:
            state["status"]="failed"; state["models"][name].update(status="verification_failed",finished_at=now()); write(state); raise RuntimeError(f"remote verification failed: {repo}")
        total=sum((getattr(x,"size",0) or 0) for x in info.siblings or [])
        state["models"][name].update(status="completed",remote_bytes=total,remote_revision=info.sha,finished_at=now(),completion_verification=str(verification)); write(state)
    state.update(status="completed",finished_at=now()); write(state); print(json.dumps(state,sort_keys=True))

if __name__=="__main__": main()
