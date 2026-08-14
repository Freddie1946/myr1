#!/usr/bin/env python3
"""Resumable eight-GPU OmniMedVQA corrected-contract sequence.

The sequence runs only the corrected primary contract and admits work per GPU,
so it can coexist with an active PathMMU repeat-inference queue. Table baselines
run before the remaining core-lineage gaps so paper-table cells are filled
first. Each completed output is immediately backed up to the manual-gated
migration dataset. Uploads are serialized to avoid competing with the large
model backup queue.
"""

from __future__ import annotations

import argparse
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
PYTHON = WORK / "pathvlm_r1_v1_a100/envs/grpo/bin/python"
MEDGEMMA_PYTHON = WORK / "pathvlm_revision_eval_a100/envs/medgemma/bin/python"
DATA = WORK / "pathvlm_revision_eval_a100/datasets/external_vqa_contract_v1_20260729/omnimedvqa_four_sources_8518.json"
OUTPUT_ROOT = WORK / "pathvlm_revision_eval_a100/runs/omnimedvqa_unified_dual_contract_20260815"
PREREQUISITE_STATE = WORK / "pathvlm_r1_v1_a100/reports/gpu_eval_continuation_20260815/state.json"
BACKUP_REPO = "Freddie1946/PathVLM-R1-Migration-Archive-20260815"
BACKUP_PREFIX = "omnimedvqa_unified_dual_contract_20260815"
MODELS = (
    ("stage3_gpt4o_step500", 0, WORK / "pathvlm_r1_v1_a100/runs/stage3_process_grpo/gpt4o_n8_parent_full_20260813/epoch_model_snapshots/checkpoint-500"),
    ("stage3_gpt4o_step1000", 1, WORK / "pathvlm_r1_v1_a100/runs/stage3_process_grpo/gpt4o_n8_parent_full_20260813/epoch_model_snapshots/checkpoint-1000"),
    ("stage3_gpt4o_step1500", 2, WORK / "pathvlm_r1_v1_a100/runs/stage3_process_grpo/gpt4o_n8_parent_full_20260813/epoch_model_snapshots/checkpoint-1500"),
    ("base_qwen2_5_vl_7b", 3, WORK / "pathvlm_r1_v1_a100/models/Qwen2.5-VL-7B-Instruct-cc594898137f460bfe9f0759e9844b3ce807cfb5"),
    ("historical_sft3000", 4, WORK / "pathvlm_r1_v1_a100/transferred_checkpoints/sft_n3000_seed42_epoch03_step1125"),
    ("historical_sft4000_control", 5, WORK / "pathvlm_r1_v1_a100/runs/stage2_control_sft4000/n1000_seed0042/sft4000_control_rl1000_seed0042_epoch02_20260729_212746/epoch_snapshots/checkpoint-250"),
    ("historical_stage2_outcome_grpo", 6, WORK / "pathvlm_r1_v1_a100/transferred_checkpoints/outcome_grpo_n1000_seed42_epoch02_step1000"),
    ("stage2_continued_rule_rl1000", 7, WORK / "pathvlm_r1_v1_a100/runs/data_ratio_rule_rl_ablation_v1/formal_sequence/tasks/stage2_continue_rule_rl1000/output/checkpoint-1500"),
)
BASELINES = (
    ("qwen2_5_vl_3b", 0, WORK / "pathvlm_revision_eval_a100/models/Qwen--Qwen2.5-VL-3B-Instruct--66285546d2b821cf421d4f5eb2576359d3770cd3"),
    ("lingshu_7b", 1, WORK / "pathvlm_revision_eval_a100/models/lingshu-medical-mllm--Lingshu-7B--b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9"),
    ("medvlm_r1", 2, WORK / "pathvlm_revision_eval_a100/models/JZPeterPan--MedVLM-R1--d256f2cfdf98c6872c1dc9f20b7dd52f49374fe9"),
    ("medgemma_4b_it", 3, WORK / "pathvlm_revision_eval_a100/models/google--medgemma-4b-it--290cda5eeccbee130f987c4ad74a59ae6f196408"),
    ("scalereasoner_r1", 4, WORK / "pathvlm_revision_eval_a100/models/ChiPhan1110--ScaleReasoner-R1--ce7f51daa9731f106874ac2bee0e9a864f7a3636"),
    ("llama3_2_vision_11b", 5, WORK / "pathvlm_r1_v1_a100/models/Llama-3.2-11B-Vision-Instruct-modelscope-master"),
)
TRACKS = (("corrected_primary", "omnimed_domain_think_answer_v4_1024", 1024),)
BACKENDS = {
    "medvlm_r1": "qwen2_vl",
    "medgemma_4b_it": "gemma3",
    "llama3_2_vision_11b": "mllama",
}
BATCH_SIZES = {
    "lingshu_7b": 8,
    "medvlm_r1": 8,
    "medgemma_4b_it": 4,
    "scalereasoner_r1": 8,
    "llama3_2_vision_11b": 2,
}
PYTHONS = {
    # Gemma3Processor is available in the pinned MedGemma environment, while
    # the GRPO environment only contains the Gemma3 model class.
    "medgemma_4b_it": MEDGEMMA_PYTHON,
}
EXISTING_CORRECTED_RESULTS = {
    "stage3_gpt4o_step500": WORK / "pathvlm_revision_eval_a100/runs/stage3_gpt4o_n8_checkpoint_ood_comparison_20260813/checkpoint500/omnimedvqa_8518",
    "stage3_gpt4o_step1000": WORK / "pathvlm_revision_eval_a100/runs/stage3_gpt4o_n8_checkpoint_ood_comparison_20260813/checkpoint1000/omnimedvqa_8518",
    "stage3_gpt4o_step1500": WORK / "pathvlm_revision_eval_a100/runs/stage3_gpt4o_n8_checkpoint1500_final_eval_20260813/omnimedvqa_8518",
    "base_qwen2_5_vl_7b": WORK / "pathvlm_revision_eval_a100/runs/core_historical_corrected_eval_20260813/base/omnimedvqa_8518",
    "historical_sft3000": WORK / "pathvlm_revision_eval_a100/runs/core_historical_corrected_eval_20260813/sft3000/omnimedvqa_8518",
    "historical_stage2_outcome_grpo": WORK / "pathvlm_revision_eval_a100/runs/core_historical_corrected_eval_20260813/stage2/omnimedvqa_8518",
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def atomic_json(path: Path, value: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, path)


def wait_for_prerequisite(poll_seconds: int) -> None:
    while True:
        if PREREQUISITE_STATE.is_file():
            try:
                status = json.loads(PREREQUISITE_STATE.read_text(encoding="utf-8")).get("status")
            except (OSError, json.JSONDecodeError):
                status = None
            if status == "completed":
                return
            if status == "failed":
                raise RuntimeError(f"prerequisite GPU queue failed: {PREREQUISITE_STATE}")
        time.sleep(poll_seconds)


def gpu_memory_mib(gpu: int) -> int:
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip())


def backup_output(output: Path, track: str, model_id: str, upload_lock: threading.Lock) -> str:
    with upload_lock:
        api = HfApi()
        api.create_repo(BACKUP_REPO, repo_type="dataset", private=False, exist_ok=True)
        api.update_repo_settings(BACKUP_REPO, repo_type="dataset", private=False, gated="manual")
        api.upload_folder(
            folder_path=str(output), repo_id=BACKUP_REPO, repo_type="dataset",
            path_in_repo=f"{BACKUP_PREFIX}/{track}/{model_id}",
            commit_message=f"Add OmniMedVQA {track} output for {model_id}",
        )
        return api.dataset_info(BACKUP_REPO, token=True).sha


def result_output(track: str, model_id: str) -> Path:
    if track == "corrected_primary" and model_id in EXISTING_CORRECTED_RESULTS:
        return EXISTING_CORRECTED_RESULTS[model_id]
    return OUTPUT_ROOT / track / model_id


def run_one(
    model_id: str, gpu: int, model: Path, track: str, contract: str, max_tokens: int,
    state: dict[str, Any], state_path: Path, state_lock: threading.Lock,
    upload_lock: threading.Lock, poll_seconds: int, retry_seconds: int,
) -> None:
    if not model.is_dir():
        raise FileNotFoundError(model)
    output = result_output(track, model_id)
    metrics = output / "metrics.json"
    task_key = f"{track}:{model_id}"
    while gpu_memory_mib(gpu) > 2048:
        state["tasks"][task_key].update({"status": "waiting_for_gpu", "updated_at": now()})
        atomic_json(state_path, state, state_lock)
        time.sleep(poll_seconds)
    log_path = OUTPUT_ROOT / "logs" / f"{track}__{model_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        if metrics.is_file():
            existing_metrics = json.loads(metrics.read_text(encoding="utf-8"))
            if existing_metrics.get("count") != 8518:
                raise RuntimeError(f"{model_id} has non-full OmniMedVQA metrics: {metrics}")
            if existing_metrics.get("generation_contract") != contract:
                raise RuntimeError(
                    f"{model_id} contract mismatch: {existing_metrics.get('generation_contract')} != {contract}"
                )
        if not metrics.is_file():
            command = [
                str(PYTHONS.get(model_id, PYTHON)), str(REPO / "scripts/run_external_vqa_qwen.py"),
                "--task", "omnimedvqa", "--model", str(model),
                "--backend", BACKENDS.get(model_id, "qwen2_5_vl"),
                "--data", str(DATA), "--output-dir", str(output), "--split-role", "external_test",
                "--batch-size", str(BATCH_SIZES.get(model_id, 16)), "--max-new-tokens", str(max_tokens),
                "--generation-contract", contract,
            ]
            # A previous attempt can initialize run_config.json before writing
            # its first prediction; that is still a valid resumable state.
            if output.is_dir():
                command.append("--resume")
            state["tasks"][task_key].update({"status": "running", "attempt": attempt, "updated_at": now()})
            atomic_json(state_path, state, state_lock)
            child_env = os.environ.copy()
            # The runner uses logical cuda:0 internally. Restrict each child to
            # its assigned physical GPU so parallel baseline jobs cannot all
            # collide on host GPU 0.
            child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            with log_path.open("a", encoding="utf-8") as log:
                code = subprocess.run(
                    command, cwd=REPO, env=child_env,
                    stdout=log, stderr=subprocess.STDOUT,
                ).returncode
            if code != 0 or not metrics.is_file():
                state["tasks"][task_key].update({"status": "retry_wait", "returncode": code, "updated_at": now()})
                atomic_json(state_path, state, state_lock)
                time.sleep(retry_seconds)
                continue
        state["tasks"][task_key].update({"status": "backing_up", "updated_at": now()})
        atomic_json(state_path, state, state_lock)
        try:
            revision = backup_output(output, track, model_id, upload_lock)
        except Exception as error:
            state["tasks"][task_key].update({"status": "backup_retry_wait", "error": repr(error), "updated_at": now()})
            atomic_json(state_path, state, state_lock)
            time.sleep(retry_seconds)
            continue
        state["tasks"][task_key].update({"status": "completed", "completed_at": now(), "backup_revision": revision})
        atomic_json(state_path, state, state_lock)
        return
    raise RuntimeError(f"{task_key} failed after three attempts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--retry-seconds", type=int, default=60)
    args = parser.parse_args()
    state_root = args.state_root.resolve()
    state_path = state_root / "state.json"
    state_lock, upload_lock = threading.Lock(), threading.Lock()
    state: dict[str, Any] = {
        "schema_version": 1, "status": "waiting_for_per_gpu_availability",
        "started_at": now(), "pid": os.getpid(), "tasks": {},
    }
    for track, _, _ in TRACKS:
        for model_id, gpu, model in MODELS + BASELINES:
            state["tasks"][f"{track}:{model_id}"] = {
                "status": "queued", "gpu": gpu, "model": str(model),
            }
    atomic_json(state_path, state, state_lock)
    for track, contract, max_tokens in TRACKS:
        # The 64-token historical-reproduction track was explicitly deferred.
        # Fill manuscript-table baseline cells before the remaining core gaps.
        for wave_name, roster in (("compatible_baselines", BASELINES), ("core", MODELS)):
            state.update({"status": "running", "current_track": track, "current_wave": wave_name, "updated_at": now()})
            atomic_json(state_path, state, state_lock)
            failures: list[str] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(roster)) as executor:
                futures = {
                    executor.submit(
                        run_one, model_id, gpu, model, track, contract, max_tokens,
                        state, state_path, state_lock, upload_lock,
                        args.poll_seconds, args.retry_seconds,
                    ): model_id
                    for model_id, gpu, model in roster
                }
                for future in concurrent.futures.as_completed(futures):
                    model_id = futures[future]
                    try:
                        future.result()
                    except Exception as error:
                        failures.append(model_id)
                        state["tasks"][f"{track}:{model_id}"].update({"status": "failed", "error": repr(error), "updated_at": now()})
                        atomic_json(state_path, state, state_lock)
            if failures:
                state.update({"status": "failed", "failed_track": track, "failed_wave": wave_name, "failures": failures, "updated_at": now()})
                atomic_json(state_path, state, state_lock)
                raise SystemExit(f"OmniMedVQA failures in {track}/{wave_name}: {failures}")

    summary: dict[str, Any] = {"schema_version": 1, "status": "completed", "completed_at": now(), "tracks": {}}
    for track, _, _ in TRACKS:
        summary["tracks"][track] = {}
        for model_id, _, _ in MODELS + BASELINES:
            metrics = json.loads((result_output(track, model_id) / "metrics.json").read_text(encoding="utf-8"))
            summary["tracks"][track][model_id] = metrics
    summary_path = OUTPUT_ROOT / "complete_summary.json"
    atomic_json(summary_path, summary, state_lock)
    with upload_lock:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(summary_path), path_in_repo=f"{BACKUP_PREFIX}/complete_summary.json",
            repo_id=BACKUP_REPO, repo_type="dataset", commit_message="Complete unified OmniMedVQA core summary",
        )
    state.update({"status": "completed", "completed_at": now(), "summary": str(summary_path)})
    atomic_json(state_path, state, state_lock)


if __name__ == "__main__":
    main()
