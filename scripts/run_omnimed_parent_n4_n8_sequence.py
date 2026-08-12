#!/usr/bin/env python3
"""Parallel OmniMedVQA evaluation for the SFT parent and matched n4/n8 policies."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/home/dataset-assist-0/czy/wjy/myr1")
PYTHON = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/grpo/bin/python")
DATA = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/"
    "external_vqa_contract_v1_20260729/omnimedvqa_four_sources_8518.json"
)
OUTPUT = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/"
    "omnimedvqa_parent_n4_n8_20260812"
)
MODELS = (
    ("l_r16_sft80_parent", Path(
        "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/"
        "formal_selected_rule_rl1000_20260811/parents/sft_step080_merged"
    ), 0),
    ("full_rl_n4_step500", Path(
        "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/"
        "full_language_rule_rl_capacity_20260811/model_snapshots/checkpoint-500"
    ), 1),
    ("full_rl_n8_step500", Path(
        "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/"
        "full_language_rule_rl_n8_capacity_20260812/model_snapshots/checkpoint-500"
    ), 2),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def command(model: Path, output: Path, smoke: bool, resume: bool = False) -> list[str]:
    cmd = [
        str(PYTHON), str(REPO / "scripts/run_external_vqa_qwen.py"),
        "--task", "omnimedvqa", "--model", str(model), "--backend", "qwen2_5_vl",
        "--data", str(DATA), "--output-dir", str(output), "--max-new-tokens", "1024",
        "--generation-contract", "omnimed_domain_think_answer_v4_1024",
        "--batch-size", "16", "--split-role", "adapter_smoke" if smoke else "external_test",
    ]
    if smoke:
        cmd.extend(["--limit", "16"])
    if resume:
        cmd.append("--resume")
    return cmd


def run_parallel(phase: str, smoke: bool, state: dict, state_path: Path) -> None:
    jobs = []
    logs = OUTPUT / "logs"
    logs.mkdir(exist_ok=True)
    for name, model, gpu in MODELS:
        target = OUTPUT / phase / name
        handle = (logs / f"{phase}_{name}.log").open("w")
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(REPO / "scripts"),
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        })
        process = subprocess.Popen(command(model, target, smoke), cwd=REPO, env=env,
                                   stdout=handle, stderr=subprocess.STDOUT)
        row = {"phase": phase, "model": name, "gpu": gpu, "status": "running",
               "started_at": now(), "attempt": 1}
        state["jobs"].append(row)
        jobs.append((process, handle, row, model, target, env))
    write_json(state_path, state)
    failed = []
    while jobs:
        for item in list(jobs):
            process, handle, row, model, target, env = item
            code = process.poll()
            if code is None:
                continue
            handle.close()
            success = code == 0 and (target / "metrics.json").is_file()
            row.update(status="completed" if success else "failed", returncode=code,
                       finished_at=now())
            if not success:
                failed.append((row, model, target, env))
            jobs.remove(item)
            write_json(state_path, state)
        if jobs:
            time.sleep(2)
    if failed and smoke:
        raise RuntimeError(f"OmniMedVQA smoke failed: {[row['model'] for row, *_ in failed]}")
    for row, model, target, env in failed:
        row.update(status="retrying", attempt=2, retry_started_at=now())
        write_json(state_path, state)
        with (OUTPUT / "logs" / f"{phase}_{row['model']}_retry.log").open("w") as handle:
            code = subprocess.run(command(model, target, smoke=False, resume=True), cwd=REPO,
                                  env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
        success = code == 0 and (target / "metrics.json").is_file()
        row.update(status="completed" if success else "failed", returncode=code,
                   finished_at=now())
        write_json(state_path, state)
        if not success:
            raise RuntimeError(f"OmniMedVQA retry failed for {row['model']}")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for _, model, _ in MODELS:
        if not (model / "model.safetensors.index.json").is_file():
            raise FileNotFoundError(model)
    if not DATA.is_file():
        raise FileNotFoundError(DATA)
    active = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if active:
        raise RuntimeError("GPU compute process is active before OmniMedVQA sequence")
    OUTPUT.mkdir(parents=True)
    state_path = OUTPUT / "sequence_state.json"
    state = {"schema_version": 1, "status": "running", "stage": "smoke",
             "started_at": now(), "jobs": [], "automatic_retry_limit": 1,
             "test_used_for_selection": False}
    write_json(state_path, state)
    try:
        run_parallel("smoke16", True, state, state_path)
        for name, _, _ in MODELS:
            metrics = json.loads((OUTPUT / f"smoke16/{name}/metrics.json").read_text())
            if metrics["count"] != 16 or metrics["strict_final_answer_coverage"] < 0.95:
                raise RuntimeError(f"OmniMedVQA smoke behavior gate failed for {name}")
        state["stage"] = "full8518"
        write_json(state_path, state)
        run_parallel("full8518", False, state, state_path)
        summary = {"schema_version": 1, "status": "completed", "models": {}}
        for name, _, _ in MODELS:
            metrics = json.loads((OUTPUT / f"full8518/{name}/metrics.json").read_text())
            summary["models"][name] = {
                key: metrics[key] for key in (
                    "count", "strict_final_correct", "strict_final_accuracy",
                    "strict_final_answer_coverage", "contract_aligned_accuracy",
                    "official_accuracy", "generation_cap_hit_count", "by_source",
                )
            }
        write_json(OUTPUT / "complete_comparison.json", summary)
        state.update(status="completed", stage="complete", finished_at=now())
        write_json(state_path, state)
    except BaseException:
        state.update(status="failed", finished_at=now())
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
