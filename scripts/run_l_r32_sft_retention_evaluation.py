#!/usr/bin/env python3
"""Run the frozen retention panel for the shortlisted L-r32 SFT checkpoints."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/home/dataset-assist-0/czy/wjy/myr1")
PYTHON = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/sft/bin/python")
MODEL = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/models/"
    "Qwen2.5-VL-7B-Instruct-cc594898137f460bfe9f0759e9844b3ce807cfb5"
)
TRAIN_ROOT = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/"
    "rule_rl_mechanism_funnel_20260811/formal_v1/phase3/l_r32_sft3000/output"
)
PATHVQA = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/"
    "pathvqa_architecture_validation_v1_20260811/pathvqa_validation_balanced_image_unique_512.json"
)
MMMU = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/"
    "mmmu_nonmedical_dev_retention_v1_20260811/panel.json"
)
R16_EVAL = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/"
    "formal_selected_sft3000_20260811"
)
R32_PATHMMU = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/"
    "rule_rl_mechanism_funnel_20260811/formal_v1/l_r32_sft_core/core_summary.json"
)


@dataclass(frozen=True)
class Job:
    name: str
    gpu: int
    command: list[str]
    output: Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def metric(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(output_root: Path) -> None:
    r32_pathmmu = metric(R32_PATHMMU)
    pathmmu_by_step = {int(row["step"]): row for row in r32_pathmmu["rows"]}
    rows = []
    for step in (16, 64):
        name = f"step{step:03d}"
        original = metric(output_root / f"pathvqa/{name}/original/metrics.json")
        shuffle = metric(output_root / f"pathvqa/{name}/cyclic_mismatch/metrics.json")
        blank = metric(output_root / f"pathvqa/{name}/global_mean_blank/metrics.json")
        mmmu = metric(output_root / f"mmmu/{name}/metrics.json")
        r16_original = metric(R16_EVAL / f"pathvqa/step{step:03d}/original/metrics.json")
        r16_shuffle = metric(R16_EVAL / f"pathvqa/step{step:03d}/cyclic_mismatch/metrics.json")
        r16_blank = metric(R16_EVAL / f"pathvqa/step{step:03d}/global_mean_blank/metrics.json")
        r16_mmmu_path = R16_EVAL / f"mmmu/step{step:03d}_rescored_v2/metrics.json"
        r16_mmmu = metric(r16_mmmu_path)
        r32 = {
            "pathmmu_val": pathmmu_by_step[step]["l_r32_accuracy"],
            "pathvqa_normal": original["accuracy"],
            "pathvqa_shuffle": shuffle["accuracy"],
            "pathvqa_blank": blank["accuracy"],
            "delta_vision": original["accuracy"] - shuffle["accuracy"],
            "delta_blank": original["accuracy"] - blank["accuracy"],
            "mmmu": mmmu["accuracy"],
            "mmmu_generation_cap_hits": mmmu["generation_cap_hit_count"],
        }
        r16 = {
            "pathmmu_val": pathmmu_by_step[step]["l_r16_accuracy"],
            "pathvqa_normal": r16_original["accuracy"],
            "pathvqa_shuffle": r16_shuffle["accuracy"],
            "pathvqa_blank": r16_blank["accuracy"],
            "delta_vision": r16_original["accuracy"] - r16_shuffle["accuracy"],
            "delta_blank": r16_original["accuracy"] - r16_blank["accuracy"],
            "mmmu": r16_mmmu["accuracy"],
            "mmmu_generation_cap_hits": r16_mmmu["generation_cap_hit_count"],
        }
        rows.append(
            {
                "step": step,
                "r16": r16,
                "r32": r32,
                "r32_minus_r16": {key: r32[key] - r16[key] for key in (
                    "pathmmu_val", "pathvqa_normal", "delta_vision", "delta_blank", "mmmu"
                )},
            }
        )
    payload = {
        "schema_version": 1,
        "status": "completed_manual_pareto_analysis_required",
        "test_accessed": False,
        "claim_boundary": (
            "PathVQA Normal/Shuffle/Blank uses a forced-binary next-token diagnostic and measures "
            "decision-policy retention/visual dependence, not free-generation semantic VQA quality. "
            "MMMU uses deterministic free generation with max_new_tokens=4096."
        ),
        "rows": rows,
    }
    write_json(output_root / "retention_summary.json", payload)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    for step in (16, 64):
        adapter = TRAIN_ROOT / f"checkpoint-{step}"
        if not (adapter / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(adapter / "adapter_model.safetensors")
    args.output_root.mkdir(parents=True)
    logs = args.output_root / "logs"
    logs.mkdir()
    jobs = []
    gpu = 0
    for step in (16, 64):
        adapter = TRAIN_ROOT / f"checkpoint-{step}"
        name = f"step{step:03d}"
        for mode in ("original", "cyclic_mismatch", "global_mean_blank"):
            output = args.output_root / "pathvqa" / name / mode
            jobs.append(Job(
                f"{name}_pathvqa_{mode}", gpu,
                [str(PYTHON), str(REPO / "scripts/run_pathvqa_forced_binary_logits.py"),
                 "--model", str(MODEL), "--adapter", str(adapter), "--data", str(PATHVQA),
                 "--output-dir", str(output), "--batch-size", "32", "--split-role", "validation_diagnostic",
                 "--image-mode", mode], output,
            ))
            gpu += 1
        output = args.output_root / "mmmu" / name
        jobs.append(Job(
            f"{name}_mmmu", gpu,
            [str(PYTHON), str(REPO / "scripts/run_mmmu_retention_diagnostic.py"),
             "--model", str(MODEL), "--adapter", str(adapter), "--panel", str(MMMU),
             "--output-dir", str(output), "--batch-size", "4", "--max-new-tokens", "4096"], output,
        ))
        gpu += 1
    state = {"schema_version": 1, "status": "running", "started_at": now(), "test_accessed": False, "jobs": []}
    state_path = args.output_root / "evaluation_state.json"
    running = []
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = str(REPO / "scripts")
    for job in jobs:
        job.output.parent.mkdir(parents=True, exist_ok=True)
        handle = (logs / f"{job.name}.log").open("w", encoding="utf-8")
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
        process = subprocess.Popen(job.command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
        running.append((process, job, handle))
        state["jobs"].append({"name": job.name, "gpu": job.gpu, "status": "running", "started_at": now()})
    write_json(state_path, state)
    failed = False
    while running:
        for process, job, handle in list(running):
            returncode = process.poll()
            if returncode is None:
                continue
            handle.close()
            row = next(item for item in state["jobs"] if item["name"] == job.name)
            success = returncode == 0 and (job.output / "metrics.json").is_file()
            row.update(status="completed" if success else "failed", returncode=returncode, finished_at=now())
            failed = failed or not success
            running.remove((process, job, handle))
            write_json(state_path, state)
        if running:
            time.sleep(1)
    if not failed:
        summarize(args.output_root)
    state.update(status="failed" if failed else "completed", finished_at=now())
    write_json(state_path, state)
    if failed:
        raise SystemExit("one or more L-r32 retention jobs failed")


if __name__ == "__main__":
    main()
