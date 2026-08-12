#!/usr/bin/env python3
"""Paired PathVQA free-generation and A/B-format diagnostic on 512 validation cases."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/home/dataset-assist-0/czy/wjy/myr1")
PYTHON = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/grpo/bin/python")
PANEL = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/datasets/"
    "pathvqa_architecture_validation_v1_20260811/"
    "pathvqa_validation_balanced_image_unique_512.json"
)
OUTPUT = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/"
    "pathvqa_choice_parent_n4_n8_20260812"
)
CONTRACTS = (
    "domain_think_answer_v2_2048",
    "pathmmu_ab_v1_2048",
)
MODELS = (
    ("l_r16_sft80_parent", Path(
        "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/"
        "formal_selected_rule_rl1000_20260811/parents/sft_step080_merged"
    ), 3),
    ("full_rl_n4_step500", Path(
        "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/"
        "full_language_rule_rl_capacity_20260811/model_snapshots/checkpoint-500"
    ), 4),
    ("full_rl_n8_step500", Path(
        "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/"
        "full_language_rule_rl_n8_capacity_20260812/model_snapshots/checkpoint-500"
    ), 5),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def inferred_letter(completion: str) -> str | None:
    tagged = re.findall(
        r"<answer\b[^>]*>\s*(.*?)(?:</answer\s*>|$)", completion, re.I | re.S
    )
    candidate = tagged[-1].strip() if tagged else completion.strip()
    match = re.match(r"^\s*\(?\s*([AB])(?=[\s).,:;\-]|$)", candidate, re.I)
    return match.group(1).upper() if match else None


def summarize_model(root: Path) -> dict:
    baseline = read_rows(root / f"{CONTRACTS[0]}/predictions.jsonl")
    ab = read_rows(root / f"{CONTRACTS[1]}/predictions.jsonl")
    if not (len(baseline) == len(ab) == 512):
        raise RuntimeError("PathVQA paired diagnostic has an incomplete panel")
    for rows in zip(baseline, ab):
        if len({row["source_record_sha256"] for row in rows}) != 1:
            raise RuntimeError("PathVQA paired diagnostic source order differs")
    baseline_wrong = [index for index, row in enumerate(baseline) if not row["correct"]]
    baseline_right = [index for index, row in enumerate(baseline) if row["correct"]]
    return {
        "count": 512,
        "contracts": {
            name: json.loads((root / f"{name}/metrics.json").read_text())
            for name in CONTRACTS
        },
        "baseline_badcase_count": len(baseline_wrong),
        "baseline_wrong_to_ab_correct": sum(ab[index]["correct"] for index in baseline_wrong),
        "baseline_right_to_ab_wrong": sum(not ab[index]["correct"] for index in baseline_right),
        "ab_selected_A_rate": sum(inferred_letter(row["completion"]) == "A" for row in ab) / 512,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    for _, model, _ in MODELS:
        if not (model / "model.safetensors.index.json").is_file():
            raise FileNotFoundError(model)
    if not PANEL.is_file():
        raise FileNotFoundError(PANEL)
    OUTPUT.mkdir(parents=True)
    logs = OUTPUT / "logs"
    logs.mkdir()
    state_path = OUTPUT / "sequence_state.json"
    state = {"schema_version": 1, "status": "running", "started_at": now(),
             "panel_role": "architecture_validation_512", "jobs": [],
             "contracts": list(CONTRACTS), "test_accessed": False}
    jobs = []
    try:
        for name, model, gpu in MODELS:
            command = [
                str(PYTHON), str(REPO / "scripts/run_pathvqa_yesno_prompt_calibration.py"),
                "--model", str(model), "--panel", str(PANEL),
                "--panel-role", "architecture_validation_512",
                "--output-root", str(OUTPUT / name), "--batch-size", "16",
                "--prompt-contracts", ",".join(CONTRACTS),
            ]
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(REPO / "scripts"),
                "TOKENIZERS_PARALLELISM": "false",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            })
            handle = (logs / f"{name}.log").open("w")
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT)
            row = {"model": name, "gpu": gpu, "status": "running", "started_at": now()}
            state["jobs"].append(row)
            jobs.append((process, handle, row, name))
        write_json(state_path, state)
        failed = []
        while jobs:
            for item in list(jobs):
                process, handle, row, name = item
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                success = code == 0 and (OUTPUT / f"{name}/index.json").is_file()
                row.update(status="completed" if success else "failed", returncode=code,
                           finished_at=now())
                failed.extend([] if success else [name])
                jobs.remove(item)
                write_json(state_path, state)
            if jobs:
                time.sleep(2)
        if failed:
            raise RuntimeError(f"PathVQA choice diagnostic failed: {failed}")
        summary = {"schema_version": 1, "status": "completed", "models": {
            name: summarize_model(OUTPUT / name) for name, _, _ in MODELS
        }}
        write_json(OUTPUT / "complete_comparison.json", summary)
        state.update(status="completed", finished_at=now())
        write_json(state_path, state)
    except BaseException:
        state.update(status="failed", finished_at=now())
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
