#!/usr/bin/env python3
"""Evaluate the two lean Phase-2 screening arms at matched exposure points."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/home/dataset-assist-0/czy/wjy/myr1")
PYTHON = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/grpo/bin/python")
RUNNER = REPO / "scripts/run_pathmmu_qwen_diagnostic.py"
MODEL = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/"
    "formal_selected_rule_rl1000_20260811/parents/sft_step080_merged"
)
TRAIN_PROBE = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/"
    "rule_rl_mechanism_train_probe256_20260811/records_n0256.json"
)
PATHMMU_VAL = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/"
    "pathmmu_image_disjoint_v2/rewritten_records/validation_0385.json"
)


@dataclass(frozen=True)
class Job:
    name: str
    adapter: Path
    data: Path
    split_role: str
    output: Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--eval-root", required=True, type=Path)
    args = parser.parse_args()
    if args.eval_root.exists():
        raise FileExistsError(args.eval_root)

    checkpoints = {
        "g20_lr3_step010": args.train_root / "g20_lr3/output/checkpoint-10",
        "g20_lr3_step025": args.train_root / "g20_lr3/output/checkpoint-25",
        "g2_lr1_step100": args.train_root / "g2_lr1/output/checkpoint-100",
        "g2_lr1_step250": args.train_root / "g2_lr1/output/checkpoint-250",
    }
    for adapter in checkpoints.values():
        if not (adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(adapter / "adapter_model.safetensors")

    logs = args.eval_root / "logs"
    logs.mkdir(parents=True)
    jobs: list[Job] = []
    for name, adapter in checkpoints.items():
        jobs.append(Job(name + "_train", adapter, TRAIN_PROBE, "rl_train_probe", args.eval_root / "train_probe" / name))
        jobs.append(Job(name + "_val", adapter, PATHMMU_VAL, "validation_smoke", args.eval_root / "pathmmu_val" / name))

    state_path = args.eval_root / "evaluation_state.json"
    state = {"schema_version": 1, "status": "running", "started_at": now(), "jobs": []}
    write_json(state_path, state)
    running = []
    for gpu, job in enumerate(jobs):
        job.output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(PYTHON), str(RUNNER), "--model", str(MODEL), "--adapter", str(job.adapter),
            "--backend", "qwen2_5_vl", "--data", str(job.data), "--output-dir", str(job.output),
            "--split-role", job.split_role, "--batch-size", "32", "--max-new-tokens", "1024",
        ]
        handle = (logs / f"{job.name}.log").open("w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONPATH"] = str(REPO / "scripts")
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
        running.append((process, job, handle))
        state["jobs"].append({"name": job.name, "gpu": gpu, "status": "running", "started_at": now()})
    write_json(state_path, state)

    failed = False
    while running:
        for process, job, handle in list(running):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            row = next(item for item in state["jobs"] if item["name"] == job.name)
            row.update(status="completed" if returncode == 0 else "failed", returncode=returncode, finished_at=now())
            running.remove((process, job, handle))
            failed = failed or returncode != 0 or not (job.output / "metrics.json").is_file()
            write_json(state_path, state)
        if running:
            time.sleep(1)
    state.update(status="failed" if failed else "completed", finished_at=now())
    write_json(state_path, state)
    if failed:
        raise SystemExit("one or more Phase-2 evaluation jobs failed")


if __name__ == "__main__":
    main()
