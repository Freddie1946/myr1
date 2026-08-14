#!/usr/bin/env python3
"""Smoke-gated corrected OmniMedVQA runs for native baseline backends.

The queue begins once the compatible-baseline wave hands off to the core wave.
Three single-GPU native backends run in parallel.  Llama-90B then runs on three
GPUs after the upstream corrected-contract sequence has fully completed.
Every full output is uploaded to the manual-gated migration archive.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi


WORK = Path("/home/dataset-assist-0/czy/wjy")
REPO = WORK / "myr1"
DATA = WORK / "pathvlm_revision_eval_a100/datasets/external_vqa_contract_v1_20260729/omnimedvqa_four_sources_8518.json"
UPSTREAM_STATE = WORK / "pathvlm_r1_v1_a100/reports/omnimedvqa_unified_dual_contract_20260815/state.json"
OUTPUT_ROOT = WORK / "pathvlm_revision_eval_a100/runs/omnimedvqa_unified_native_baselines_20260815"
BACKUP_REPO = "Freddie1946/PathVLM-R1-Migration-Archive-20260815"
BACKUP_PREFIX = "omnimedvqa_unified_dual_contract_20260815/corrected_primary"
CONTRACT = "omnimed_domain_think_answer_v4_1024"
MAX_TOKENS = 1024

NATIVE_TASKS = (
    {
        "id": "huatuogpt_vision_7b",
        "gpu": "0",
        "python": WORK / "pathvlm_revision_eval_a100/envs/huatuo_llava/bin/python",
        "runner": REPO / "scripts/run_external_vqa_huatuo.py",
        "model": WORK / "pathvlm_revision_eval_a100/models/FreedomIntelligence--HuatuoGPT-Vision-7B--34dfcdbb7728ff38da865839f342b88c4cf6ef39",
        "pythonpath": WORK / "pathvlm_revision_eval_a100/sources/HuatuoGPT-Vision",
    },
    {
        "id": "internvl3_8b",
        "gpu": "1",
        "python": WORK / "pathvlm_revision_eval_a100/envs/internvl3/bin/python",
        "runner": REPO / "scripts/run_external_vqa_internvl.py",
        "model": WORK / "pathvlm_revision_eval_a100/models/OpenGVLab--InternVL3-8B--853e3a797a661694b1b8ece0cb72dc2b23e3dac9",
    },
    {
        "id": "deepseek_vl2",
        "gpu": "2",
        "python": WORK / "pathvlm_revision_eval_a100/envs/deepseek_vl2/bin/python",
        "runner": REPO / "scripts/run_external_vqa_deepseek_vl2.py",
        "model": WORK / "pathvlm_revision_eval_a100/models/deepseek-ai--deepseek-vl2--f363772d1c47f4239dd844015b4bd53beb87951b",
        "pythonpath": WORK / "pathvlm_revision_eval_a100/sources/DeepSeek-VL2",
    },
)

LLAMA90 = {
    "id": "llama3_2_vision_90b",
    "gpu": "0,1,2",
    "python": WORK / "pathvlm_r1_v1_a100/envs/grpo/bin/python",
    "runner": REPO / "scripts/run_external_vqa_qwen.py",
    "model": WORK / "pathvlm_r1_v1_a100/models/Llama-3.2-90B-Vision-Instruct-modelscope-master",
    "extra": ["--backend", "mllama", "--batch-size", "1"],
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_state(path: Path, state: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(f".tmp-{os.getpid()}")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        os.replace(temp, path)


def upstream() -> dict[str, Any]:
    try:
        return json.loads(UPSTREAM_STATE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def wait_for_native_handoff(poll: int) -> None:
    while True:
        value = upstream()
        if value.get("status") == "failed":
            raise RuntimeError("upstream compatible-baseline wave failed")
        if value.get("current_wave") == "core" or value.get("status") == "completed":
            return
        time.sleep(poll)


def wait_for_upstream_completion(poll: int) -> None:
    while True:
        value = upstream()
        if value.get("status") == "failed":
            raise RuntimeError("upstream corrected-contract sequence failed")
        if value.get("status") == "completed":
            return
        time.sleep(poll)


def gpu_memory_mib(gpu: int) -> int:
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True,
    )
    return int(result.stdout.strip())


def wait_for_gpus(gpus: str, poll: int) -> None:
    physical = [int(value) for value in gpus.split(",")]
    while any(gpu_memory_mib(gpu) > 2048 for gpu in physical):
        time.sleep(poll)


def command(task: dict[str, Any], output: Path, *, smoke: bool) -> list[str]:
    value = [
        str(task["python"]), str(task["runner"]),
        "--task", "omnimedvqa", "--model", str(task["model"]),
        "--data", str(DATA), "--output-dir", str(output),
        "--split-role", "adapter_smoke" if smoke else "external_test",
        "--max-new-tokens", str(MAX_TOKENS), "--generation-contract", CONTRACT,
    ]
    value.extend(task.get("extra", []))
    if smoke:
        value.extend(["--limit", "16"])
    if output.is_dir():
        value.append("--resume")
    return value


def run_command(task: dict[str, Any], output: Path, log: Path, *, smoke: bool) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(task["gpu"])
    additions = [str(REPO / "scripts")]
    if task.get("pythonpath"):
        additions.insert(0, str(task["pythonpath"]))
    env["PYTHONPATH"] = os.pathsep.join(additions + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        code = subprocess.run(
            command(task, output, smoke=smoke), cwd=REPO, env=env,
            stdout=handle, stderr=subprocess.STDOUT,
        ).returncode
    metrics = output / "metrics.json"
    if code != 0 or not metrics.is_file():
        raise RuntimeError(f"{task['id']} {'smoke' if smoke else 'full'} failed with {code}")
    result = json.loads(metrics.read_text(encoding="utf-8"))
    expected = 16 if smoke else 8518
    if result.get("count") != expected or result.get("generation_contract") != CONTRACT:
        raise RuntimeError(f"{task['id']} output contract/count mismatch")


def upload(output: Path, model_id: str, lock: threading.Lock) -> str:
    with lock:
        api = HfApi()
        api.create_repo(BACKUP_REPO, repo_type="dataset", private=False, exist_ok=True)
        api.update_repo_settings(BACKUP_REPO, repo_type="dataset", private=False, gated="manual")
        api.upload_folder(
            folder_path=str(output), repo_id=BACKUP_REPO, repo_type="dataset",
            path_in_repo=f"{BACKUP_PREFIX}/{model_id}",
            commit_message=f"Add corrected OmniMedVQA output for {model_id}",
        )
        return api.dataset_info(BACKUP_REPO, token=True).sha


def run_task(
    task: dict[str, Any], state: dict[str, Any], state_path: Path,
    state_lock: threading.Lock, upload_lock: threading.Lock, poll: int,
) -> None:
    key = task["id"]
    if not Path(task["model"]).is_dir():
        raise FileNotFoundError(task["model"])
    wait_for_gpus(str(task["gpu"]), poll)
    smoke = OUTPUT_ROOT / "smoke16" / key
    full = OUTPUT_ROOT / "full8518" / key
    logs = OUTPUT_ROOT / "logs"
    for attempt in range(1, 4):
        try:
            state["tasks"][key].update({"status": "smoke", "attempt": attempt, "updated_at": now()})
            write_state(state_path, state, state_lock)
            if not (smoke / "metrics.json").is_file():
                run_command(task, smoke, logs / f"{key}__smoke.log", smoke=True)
            state["tasks"][key].update({"status": "full", "updated_at": now()})
            write_state(state_path, state, state_lock)
            if not (full / "metrics.json").is_file():
                run_command(task, full, logs / f"{key}__full.log", smoke=False)
            state["tasks"][key].update({"status": "backing_up", "updated_at": now()})
            write_state(state_path, state, state_lock)
            revision = upload(full, key, upload_lock)
            state["tasks"][key].update({"status": "completed", "backup_revision": revision, "completed_at": now()})
            write_state(state_path, state, state_lock)
            return
        except Exception as error:
            state["tasks"][key].update({"status": "retry_wait", "error": repr(error), "updated_at": now()})
            write_state(state_path, state, state_lock)
            time.sleep(60)
            wait_for_gpus(str(task["gpu"]), poll)
    raise RuntimeError(f"{key} failed after three attempts")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    state_path = args.state_root.resolve() / "state.json"
    state_lock, upload_lock = threading.Lock(), threading.Lock()
    state: dict[str, Any] = {
        "schema_version": 1, "pid": os.getpid(), "started_at": now(),
        "status": "waiting_for_compatible_baselines", "tasks": {
            task["id"]: {"status": "queued", "gpu": task["gpu"], "model": str(task["model"])}
            for task in (*NATIVE_TASKS, LLAMA90)
        },
    }
    write_state(state_path, state, state_lock)
    wait_for_native_handoff(args.poll_seconds)
    state.update({"status": "running_native_single_gpu", "updated_at": now()})
    write_state(state_path, state, state_lock)
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(NATIVE_TASKS)) as executor:
        futures = {
            executor.submit(
                run_task, task, state, state_path, state_lock, upload_lock, args.poll_seconds
            ): task["id"]
            for task in NATIVE_TASKS
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                key = futures[future]
                failures.append(key)
                state["tasks"][key].update({"status": "failed", "error": repr(error), "updated_at": now()})
                write_state(state_path, state, state_lock)
    if failures:
        state.update({"status": "failed", "failures": failures, "updated_at": now()})
        write_state(state_path, state, state_lock)
        raise SystemExit(f"native OmniMedVQA failures: {failures}")

    state.update({"status": "waiting_for_llama90", "updated_at": now()})
    write_state(state_path, state, state_lock)
    wait_for_upstream_completion(args.poll_seconds)
    run_task(LLAMA90, state, state_path, state_lock, upload_lock, args.poll_seconds)
    state.update({"status": "completed", "completed_at": now()})
    write_state(state_path, state, state_lock)


if __name__ == "__main__":
    main()
