#!/usr/bin/env python3
"""Evaluate a parent and GRPO checkpoints on frozen train/validation probes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import deque
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
    adapter: Path | None
    data: Path
    split_role: str
    output: Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--arm", action="append", required=True, help="name:relative-run-dir")
    parser.add_argument("--steps", default="10,25,50")
    args = parser.parse_args()
    if args.eval_root.exists():
        raise FileExistsError(args.eval_root)
    steps = [int(value) for value in args.steps.split(",")]
    if not steps or len(set(steps)) != len(steps):
        raise ValueError("invalid steps")
    configs: list[tuple[str, Path | None]] = [("parent", None)]
    for specification in args.arm:
        name, relative = specification.split(":", 1)
        run_dir = args.train_root / relative
        for step in steps:
            adapter = run_dir / "output" / f"checkpoint-{step}"
            if not (adapter / "adapter_model.safetensors").is_file():
                raise FileNotFoundError(adapter / "adapter_model.safetensors")
            configs.append((f"{name}_step{step:03d}", adapter))

    args.eval_root.mkdir(parents=True)
    logs = args.eval_root / "logs"
    logs.mkdir()
    jobs: deque[Job] = deque()
    for name, adapter in configs:
        jobs.append(Job(name + "_train", adapter, TRAIN_PROBE, "rl_train_probe", args.eval_root / "train_probe" / name))
        jobs.append(Job(name + "_val", adapter, PATHMMU_VAL, "validation_smoke", args.eval_root / "pathmmu_val" / name))

    state_path = args.eval_root / "evaluation_state.json"
    state = {"schema_version": 1, "status": "running", "started_at": now(), "jobs": []}
    write_json(state_path, state)
    running: dict[int, tuple[subprocess.Popen, Job, object]] = {}
    failed = False
    while jobs or running:
        while jobs and len(running) < 8 and not failed:
            gpu = next(index for index in range(8) if index not in running)
            job = jobs.popleft()
            job.output.parent.mkdir(parents=True, exist_ok=True)
            command = [
                str(PYTHON), str(RUNNER), "--model", str(MODEL),
                "--backend", "qwen2_5_vl", "--data", str(job.data),
                "--output-dir", str(job.output), "--split-role", job.split_role,
                "--batch-size", "32", "--max-new-tokens", "1024",
            ]
            if job.adapter is not None:
                command.extend(("--adapter", str(job.adapter)))
            log_handle = (logs / f"{job.name}.log").open("w", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONPATH"] = str(REPO / "scripts")
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            running[gpu] = (process, job, log_handle)
            state["jobs"].append({"name": job.name, "gpu": gpu, "status": "running", "started_at": now()})
            write_json(state_path, state)
        for gpu, (process, job, handle) in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            row = next(item for item in state["jobs"] if item["name"] == job.name)
            row.update(status="completed" if returncode == 0 else "failed", returncode=returncode, finished_at=now())
            del running[gpu]
            if returncode != 0:
                failed = True
            elif not (job.output / "metrics.json").is_file():
                row.update(status="failed", error="missing metrics.json")
                failed = True
            write_json(state_path, state)
        if running:
            import time
            time.sleep(1)
        if failed:
            for process, _, handle in running.values():
                process.terminate()
                handle.close()
            state.update(status="failed", finished_at=now())
            write_json(state_path, state)
            raise SystemExit(1)
    state.update(status="completed", finished_at=now())
    write_json(state_path, state)


if __name__ == "__main__":
    main()
