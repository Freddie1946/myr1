#!/usr/bin/env python3
"""Start n=8 PathMMU validation/test999 after the queued training arm completes.

This is a background continuation helper: it polls a local state file only and
does not call the model or consume assistant tokens while waiting.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path("/home/dataset-assist-0/czy/wjy/myr1")
PYTHON = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/grpo/bin/python")
RUNROOT = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/full_language_rule_rl_clean_n4_n8_step1000_20260812")
EVALROOT = Path("/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/full_language_rule_rl_clean_n4_n8_step1000_20260812/n8_fresh_step1000_pathmmu_eval")
STATE = RUNROOT / "sequence_state.json"
MODEL = RUNROOT / "n8_fresh_step1000/model_snapshots/checkpoint-1000"
VAL = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/pathmmu_image_disjoint_v2/rewritten_records/validation_0385.json")
TEST = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/pathmmu_image_disjoint_v2/rewritten_records/test_0999.json")


def gpu_pids() -> list[str]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    return [x.strip() for x in out.splitlines() if x.strip()]


def main() -> None:
    while True:
        state = json.loads(STATE.read_text(encoding="utf-8"))
        if state.get("status") == "failed":
            raise RuntimeError(f"training queue failed: {state.get('error', '')}")
        if state.get("status") == "completed" and not gpu_pids():
            break
        time.sleep(300)
    if not MODEL.is_dir():
        raise FileNotFoundError(MODEL)
    EVALROOT.mkdir(parents=True, exist_ok=True)
    (EVALROOT / "logs").mkdir(exist_ok=True)
    jobs = [("val", 0, VAL, "validation_smoke"), ("test999", 1, TEST, "test999_development")]
    procs = []
    for name, gpu, data, role in jobs:
        output = EVALROOT / name
        if output.exists() and (output / "metrics.json").is_file():
            continue
        if output.exists():
            raise RuntimeError(f"incomplete evaluation output exists: {output}")
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(REPO / "scripts"), "TOKENIZERS_PARALLELISM": "false"})
        cmd = [str(PYTHON), str(REPO / "scripts/run_pathmmu_qwen_diagnostic.py"), "--model", str(MODEL), "--backend", "qwen2_5_vl", "--data", str(data), "--output-dir", str(output), "--batch-size", "32", "--split-role", role, "--max-new-tokens", "1024"]
        handle = (EVALROOT / "logs" / f"{name}.log").open("w", encoding="utf-8")
        procs.append((name, subprocess.Popen(cmd, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT), handle, output))
    results = []
    for name, proc, handle, output in procs:
        code = proc.wait(); handle.close()
        results.append({"name": name, "returncode": code, "success": code == 0 and (output / "metrics.json").is_file()})
    if not all(row["success"] for row in results):
        raise RuntimeError(f"n8 evaluation failed: {results}")
    metrics = {name: json.loads((EVALROOT / name / "metrics.json").read_text(encoding="utf-8")) for name, *_ in jobs}
    (EVALROOT / "complete_summary.json").write_text(json.dumps({"status": "completed", "model": str(MODEL), "metrics": metrics}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(EVALROOT), "metrics": {key: value.get("accuracy") for key, value in metrics.items()}}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
