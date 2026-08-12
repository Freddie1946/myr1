#!/usr/bin/env python3
"""Run n=4 fresh step1000 PathMMU validation/test999, then resume queued n=8."""

from __future__ import annotations

import json
import os
import subprocess
import traceback
from pathlib import Path

REPO = Path("/home/dataset-assist-0/czy/wjy/myr1")
PYTHON = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/grpo/bin/python")
RUNROOT = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/full_language_rule_rl_clean_n4_n8_step1000_20260812")
EVALROOT = Path("/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/full_language_rule_rl_clean_n4_n8_step1000_20260812/n4_fresh_step1000_pathmmu_eval")
MODEL = RUNROOT / "n4_fresh_step1000/model_snapshots/checkpoint-1000"
VAL = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/pathmmu_image_disjoint_v2/rewritten_records/validation_0385.json")
TEST = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/pathmmu_image_disjoint_v2/rewritten_records/test_0999.json")
QUEUE = RUNROOT / "sequence_state.json"


def write_state(value: dict) -> None:
    temporary = QUEUE.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, QUEUE)


def main() -> None:
    state = json.loads(QUEUE.read_text())
    state.update(stage="n4_pathmmu_eval_running", updated_at="2026-08-12T17:35:00+08:00")
    write_state(state)
    EVALROOT.mkdir(parents=True, exist_ok=True)
    (EVALROOT / "logs").mkdir(exist_ok=True)
    jobs = [
        ("val", 0, VAL, "validation_smoke"),
        ("test999", 1, TEST, "test999_development"),
    ]
    processes = []
    for name, gpu, data, role in jobs:
        output = EVALROOT / name
        if output.exists():
            if (output / "metrics.json").is_file():
                continue
            raise FileExistsError(f"incomplete evaluation output exists: {output}")
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(REPO / "scripts"), "TOKENIZERS_PARALLELISM": "false"})
        command = [str(PYTHON), str(REPO / "scripts/run_pathmmu_qwen_diagnostic.py"), "--model", str(MODEL), "--backend", "qwen2_5_vl", "--data", str(data), "--output-dir", str(output), "--batch-size", "32", "--split-role", role, "--max-new-tokens", "1024"]
        handle = (EVALROOT / "logs" / f"{name}.log").open("w")
        processes.append((name, subprocess.Popen(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT), handle, output))
    results = []
    for name, process, handle, output in processes:
        code = process.wait(); handle.close()
        results.append({"name": name, "returncode": code, "success": code == 0 and (output / "metrics.json").is_file()})
    if not all(row["success"] for row in results):
        raise RuntimeError(f"PathMMU evaluation failed: {results}")
    metrics = {name: json.loads((EVALROOT / name / "metrics.json").read_text()) for name, *_ in jobs}
    (EVALROOT / "complete_summary.json").write_text(json.dumps({"status": "completed", "model": str(MODEL), "metrics": metrics}, indent=2, sort_keys=True) + "\n")
    state = json.loads(QUEUE.read_text())
    state.update(stage="n4_pathmmu_eval_complete_n8_starting", n4_pathmmu_eval=str((EVALROOT / "complete_summary.json").resolve()), updated_at="2026-08-12T17:35:00+08:00")
    write_state(state)
    from run_full_language_rule_rl_clean_n4_n8_step1000 import run_arm
    run_arm(8, state)
    state.update(status="completed", stage="complete", finished_at="2026-08-12T17:35:00+08:00")
    write_state(state)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        state = json.loads(QUEUE.read_text())
        state.update(status="failed", stage="n4_eval_or_n8_failed", error=traceback.format_exc(), updated_at="2026-08-12T17:35:00+08:00")
        write_state(state)
        raise
