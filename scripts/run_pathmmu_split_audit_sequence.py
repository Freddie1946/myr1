#!/usr/bin/env python3
"""Smoke, run, and merge the three-model PathMMU split audit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/home/dataset-assist-0/czy/wjy/myr1")
PYTHON = Path("/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/envs/grpo/bin/python")
BASE = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/models/"
    "Qwen2.5-VL-7B-Instruct-cc594898137f460bfe9f0759e9844b3ce807cfb5"
)
L_ADAPTER = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/"
    "formal_selected_sft3000_20260811/output/checkpoint-80"
)
HISTORICAL_FULL_SFT = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/transferred_checkpoints/"
    "sft_n3000_seed42_epoch03_step1125"
)
SFT = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/"
    "pathmmu_image_disjoint_v2/rewritten_records/sft_3000.json"
)
RL = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/data/"
    "pathmmu_image_disjoint_v2/rewritten_records/rl_1000.json"
)
N8_STATE = Path(
    "/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/"
    "full_language_rule_rl_n8_capacity_20260812/sequence_state.json"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run_model(
    output: Path, name: str, model: Path, adapter: Path | None,
    limit: int | None, prompt_batch_size: int, state: dict, state_path: Path,
) -> None:
    phase = "smoke" if limit is not None else "formal"
    root = output / phase / name
    shards = root / "shards"
    logs = root / "logs"
    shards.mkdir(parents=True)
    logs.mkdir()
    row = {
        "phase": phase, "model": name, "status": "running", "started_at": now(),
        "limit_per_split": limit, "prompt_batch_size": prompt_batch_size,
    }
    state["jobs"].append(row)
    write_json(state_path, state)
    pending = set(range(8))
    for attempt in range(1, 4):
        running = []
        for shard in sorted(pending):
            command = [
                str(PYTHON), str(REPO / "scripts/run_pathmmu_split_passk_shard.py"),
                "--model", str(model), "--sft-data", str(SFT), "--rl-data", str(RL),
                "--output-dir", str(shards / f"shard_{shard}"),
                "--shard-index", str(shard), "--shard-count", "8", "--rollouts", "8",
                "--temperature", "0.9", "--max-new-tokens", "384",
                "--prompt-batch-size", str(prompt_batch_size), "--seed", "424242",
            ]
            if adapter is not None:
                command.extend(["--adapter", str(adapter)])
            if limit is not None:
                command.extend(["--limit-per-split", str(limit)])
            if attempt > 1:
                command.append("--resume")
            handle = (logs / f"shard_{shard}_attempt_{attempt}.log").open("w")
            env = os.environ.copy()
            env.update({
                "CUDA_VISIBLE_DEVICES": str(shard),
                "PYTHONPATH": str(REPO / "scripts"),
                "TOKENIZERS_PARALLELISM": "false",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            })
            process = subprocess.Popen(
                command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT
            )
            running.append((shard, process, handle))
        failed = set()
        for shard, process, handle in running:
            code = process.wait()
            handle.close()
            if code != 0 or not (shards / f"shard_{shard}/metrics.json").is_file():
                failed.add(shard)
        pending = failed
        row[f"attempt_{attempt}_failed_shards"] = sorted(failed)
        write_json(state_path, state)
        if not pending:
            break
    if pending:
        row.update(status="failed", finished_at=now(), failed_shards=sorted(pending))
        write_json(state_path, state)
        raise RuntimeError(f"{phase}/{name} failed after bounded retries: {sorted(pending)}")
    merge = [
        str(PYTHON), str(REPO / "scripts/merge_pathmmu_split_passk.py"),
        "--sft-data", str(SFT), "--rl-data", str(RL),
        "--shard-root", str(shards), "--output-dir", str(root / "merged"),
    ]
    if limit is not None:
        merge.extend(["--limit-per-split", str(limit)])
    subprocess.run(merge, cwd=REPO, check=True)
    row.update(status="completed", finished_at=now())
    write_json(state_path, state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    required = [BASE / "config.json", L_ADAPTER / "adapter_model.safetensors",
                HISTORICAL_FULL_SFT / "model.safetensors.index.json", SFT, RL]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    active = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if active:
        raise RuntimeError(f"GPU compute processes already active: {active}")
    if not N8_STATE.exists():
        subprocess.run([
            str(PYTHON), str(REPO / "scripts/run_full_language_rule_rl_n8_step500.py")
        ], cwd=REPO, check=True)
    n8_state = json.loads(N8_STATE.read_text(encoding="utf-8"))
    if n8_state.get("status") != "completed" or n8_state.get("global_step") != 500:
        raise RuntimeError(f"n8 step500 prerequisite did not complete: {n8_state}")
    if subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip():
        raise RuntimeError("n8 prerequisite left GPU processes active")
    args.output.mkdir(parents=True)
    state_path = args.output / "sequence_state.json"
    state = {
        "schema_version": 1, "status": "running", "started_at": now(),
        "automatic_retry_limit": 2, "jobs": [],
        "n8_step500_prerequisite_state": str(N8_STATE),
        "sampling_contract": {
            "rollouts": 8, "temperature": 0.9, "top_p": 1.0,
            "top_k": 0, "max_new_tokens": 384,
        },
    }
    write_json(state_path, state)
    subprocess.run([
        str(PYTHON), str(REPO / "scripts/audit_pathmmu_static_split.py"),
        "--sft-data", str(SFT), "--rl-data", str(RL),
        "--output", str(args.output / "static_split_audit.json"),
    ], cwd=REPO, check=True)
    models = [
        ("base", BASE, None),
        ("l_r16_step80", BASE, L_ADAPTER),
        ("historical_full_sft", HISTORICAL_FULL_SFT, None),
    ]
    try:
        for name, model, adapter in models:
            run_model(args.output, name, model, adapter, 8, 1, state, state_path)
        for name, model, adapter in models:
            run_model(args.output, name, model, adapter, None, 2, state, state_path)
        comparison = {
            "schema_version": 1, "status": "completed", "models": {},
            "interpretation_boundary": (
                "Diagnostic pass@8 audit; thresholds describe RL suitability and are not "
                "a training-data selection rule."
            ),
        }
        for name, _, _ in models:
            metrics = json.loads(
                (args.output / f"formal/{name}/merged/metrics.json").read_text()
            )
            comparison["models"][name] = {
                "splits": metrics["splits"],
                "split_difference": metrics["split_difference"],
            }
        write_json(args.output / "complete_comparison.json", comparison)
        state.update(status="completed", finished_at=now())
        write_json(state_path, state)
    except BaseException:
        state.update(status="failed", finished_at=now())
        write_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
