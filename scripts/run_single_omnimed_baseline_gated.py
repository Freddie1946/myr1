#!/usr/bin/env python3
"""Run one OmniMedVQA baseline through its own smoke gate and optional full job.

Each invocation owns exactly one model and one state file.  There is no shared
baseline queue: a failure or tuning change for one backend cannot advance,
block, or mutate another model's task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORK = Path("/home/dataset-assist-0/czy/wjy")
REPO = WORK / "myr1"
DATA = WORK / "pathvlm_revision_eval_a100/datasets/external_vqa_contract_v1_20260729/omnimedvqa_four_sources_8518.json"
SMOKE_DATA = WORK / "pathvlm_revision_eval_a100/datasets/omnimedvqa_stratified_smoke64_20260815/omnimedvqa_four_sources_stratified_64.json"
OUTPUT_ROOT = WORK / "pathvlm_revision_eval_a100/runs/omnimedvqa_per_model_gated_v2_20260815"
CONTRACT = "omnimed_domain_think_answer_v4_1024"
MAX_TOKENS = 1024
GRPO_PYTHON = WORK / "pathvlm_r1_v1_a100/envs/grpo/bin/python"


def qwen_task(model: str, backend: str = "qwen2_5_vl", batch: int = 8, **extra: Any) -> dict[str, Any]:
    return {
        "python": GRPO_PYTHON,
        "runner": REPO / "scripts/run_external_vqa_qwen.py",
        "model": Path(model),
        "args": ["--backend", backend, "--batch-size", str(batch)],
        **extra,
    }


TASKS: dict[str, dict[str, Any]] = {
    "qwen2_5_vl_3b": qwen_task(
        WORK / "pathvlm_revision_eval_a100/models/Qwen--Qwen2.5-VL-3B-Instruct--66285546d2b821cf421d4f5eb2576359d3770cd3",
        batch=16,
    ),
    "lingshu_7b": qwen_task(
        WORK / "pathvlm_revision_eval_a100/models/lingshu-medical-mllm--Lingshu-7B--b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9",
        batch=8,
    ),
    "medvlm_r1": qwen_task(
        WORK / "pathvlm_revision_eval_a100/models/JZPeterPan--MedVLM-R1--d256f2cfdf98c6872c1dc9f20b7dd52f49374fe9",
        backend="qwen2_vl",
        batch=8,
    ),
    "medgemma_4b_it": qwen_task(
        WORK / "pathvlm_revision_eval_a100/models/google--medgemma-4b-it--290cda5eeccbee130f987c4ad74a59ae6f196408",
        backend="gemma3",
        batch=4,
        python=WORK / "pathvlm_revision_eval_a100/envs/medgemma/bin/python",
        generation_args=["--repetition-penalty", "1.05", "--no-repeat-ngram-size", "8"],
    ),
    "scalereasoner_r1": qwen_task(
        WORK / "pathvlm_revision_eval_a100/models/ChiPhan1110--ScaleReasoner-R1--ce7f51daa9731f106874ac2bee0e9a864f7a3636",
        batch=8,
    ),
    "llama3_2_vision_11b": qwen_task(
        WORK / "pathvlm_r1_v1_a100/models/Llama-3.2-11B-Vision-Instruct-modelscope-master",
        backend="mllama",
        batch=2,
        # Mllama inserts image_token_index=128256 while the language logits
        # cover IDs 0..128255. Transformers' repetition-penalty processor
        # indexes logits with every input ID and therefore crashes on that
        # visual placeholder. Adjacent n-gram suppression does not perform
        # that unsafe global gather and is the model-specific loop control.
        generation_args=[
            "--generated-token-repetition-penalty", "1.10",
            "--no-repeat-ngram-size", "8",
        ],
        args=[
            "--backend", "mllama", "--batch-size", "2",
            "--omnimed-prompt-style", "native_choice_only",
        ],
    ),
    "huatuogpt_vision_7b": {
        "python": WORK / "pathvlm_revision_eval_a100/envs/huatuo_llava/bin/python",
        "runner": REPO / "scripts/run_external_vqa_huatuo.py",
        "model": WORK / "pathvlm_revision_eval_a100/models/FreedomIntelligence--HuatuoGPT-Vision-7B--34dfcdbb7728ff38da865839f342b88c4cf6ef39",
        "pythonpath": WORK / "pathvlm_revision_eval_a100/sources/HuatuoGPT-Vision",
        "args": [],
    },
    "internvl3_8b": {
        "python": WORK / "pathvlm_revision_eval_a100/envs/internvl3/bin/python",
        "runner": REPO / "scripts/run_external_vqa_internvl.py",
        "model": WORK / "pathvlm_revision_eval_a100/models/OpenGVLab--InternVL3-8B--853e3a797a661694b1b8ece0cb72dc2b23e3dac9",
        "args": [],
    },
    "deepseek_vl2": {
        "python": WORK / "pathvlm_revision_eval_a100/envs/deepseek_vl2/bin/python",
        "runner": REPO / "scripts/run_external_vqa_deepseek_vl2.py",
        "model": WORK / "pathvlm_revision_eval_a100/models/deepseek-ai--deepseek-vl2--f363772d1c47f4239dd844015b4bd53beb87951b",
        "pythonpath": WORK / "pathvlm_revision_eval_a100/sources/DeepSeek-VL2",
        "args": ["--omnimed-prompt-style", "native_choice_only"],
    },
    "llava_med_7b": {
        "python": WORK / "pathvlm_revision_eval_a100/envs/llava_med/bin/python",
        "runner": REPO / "scripts/run_external_vqa_llava_med.py",
        "model": WORK / "pathvlm_revision_eval_a100/models/microsoft--llava-med-v1.5-mistral-7b--91bb16c122001ddc9cf1fd36ce1dae09448943a2",
        "pythonpath": WORK / "pathvlm_revision_eval_a100/sources/LLaVA-Med",
        "args": ["--omnimed-prompt-style", "native_choice_only"],
        "minimum_strict_final_coverage": 0.75,
        # One of 64 deterministic prompts terminates immediately with EOS.
        # This is recorded as model non-compliance and counted wrong; it is not
        # a transport/truncation failure and must not be silently imputed.
        "maximum_empty_count": 1,
    },
    "llama3_2_vision_90b": qwen_task(
        WORK / "pathvlm_r1_v1_a100/models/Llama-3.2-90B-Vision-Instruct-modelscope-master",
        backend="mllama",
        batch=1,
        generation_args=[
            "--generated-token-repetition-penalty", "1.10",
            "--no-repeat-ngram-size", "8",
        ],
        args=[
            "--backend", "mllama", "--batch-size", "1",
            "--omnimed-prompt-style", "native_choice_only",
        ],
    ),
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def execute(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {' '.join(command)}")


def gpu_memory_used_mib(gpus: str) -> dict[str, int]:
    usage = {}
    for gpu in gpus.split(","):
        gpu = gpu.strip()
        result = subprocess.run(
            [
                "nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        usage[gpu] = int(result.stdout.strip())
    return usage


def inference_command(task: dict[str, Any], data: Path, output: Path, role: str) -> list[str]:
    return [
        str(task["python"]),
        str(task["runner"]),
        "--task", "omnimedvqa",
        "--model", str(task["model"]),
        "--data", str(data),
        "--output-dir", str(output),
        "--split-role", role,
        "--generation-contract", CONTRACT,
        "--max-new-tokens", str(MAX_TOKENS),
        *task.get("args", []),
        *task.get("generation_args", []),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=sorted(TASKS), required=True)
    parser.add_argument("--gpus", required=True, help="CUDA_VISIBLE_DEVICES value owned by this task")
    parser.add_argument("--run-full-after-smoke", action="store_true")
    parser.add_argument("--wait-for-free-gpus", action="store_true")
    parser.add_argument("--gpu-wait-threshold-mib", type=int, default=1024)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()

    task = TASKS[args.model_id]
    if not task["model"].is_dir():
        raise FileNotFoundError(task["model"])
    root = OUTPUT_ROOT / args.model_id
    state_path = root / "state.json"
    smoke = root / "smoke64"
    full = root / "full8518"
    state: dict[str, Any] = {
        "schema_version": 1,
        "pid": os.getpid(),
        "model_id": args.model_id,
        "model": str(task["model"].resolve()),
        "gpus": args.gpus,
        "generation_contract": CONTRACT,
        "run_full_after_smoke": args.run_full_after_smoke,
        "status": "starting",
        "started_at": now(),
    }
    atomic_json(state_path, state)
    if args.wait_for_free_gpus:
        while True:
            usage = gpu_memory_used_mib(args.gpus)
            if all(value <= args.gpu_wait_threshold_mib for value in usage.values()):
                break
            state.update({
                "status": "waiting_for_free_gpus",
                "gpu_memory_used_mib": usage,
                "updated_at": now(),
            })
            atomic_json(state_path, state)
            time.sleep(args.poll_seconds)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    if task.get("pythonpath"):
        env["PYTHONPATH"] = str(task["pythonpath"]) + (
            ":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
    try:
        if not (smoke / "metrics.json").is_file():
            state.update({"status": "running_smoke", "updated_at": now()})
            atomic_json(state_path, state)
            command = inference_command(task, SMOKE_DATA, smoke, "adapter_smoke")
            if smoke.is_dir():
                command.append("--resume")
            execute(command, root / "smoke.log", env)
        state.update({"status": "verifying_smoke", "updated_at": now()})
        atomic_json(state_path, state)
        gate = smoke / "smoke_gate.json"
        execute(
            [
                str(GRPO_PYTHON), str(REPO / "scripts/verify_omnimed_stratified_smoke.py"),
                "--metrics", str(smoke / "metrics.json"),
                "--predictions", str(smoke / "predictions.jsonl"),
                "--expected-data-sha256", sha256_file(SMOKE_DATA),
                "--expected-model-config-sha256", sha256_file(task["model"] / "config.json"),
                "--minimum-strict-final-coverage", str(
                    task.get("minimum_strict_final_coverage", 0.95)
                ),
                "--maximum-empty-count", str(task.get("maximum_empty_count", 0)),
                "--output", str(gate),
            ],
            root / "gate.log",
            env,
        )
        if not args.run_full_after_smoke:
            state.update({"status": "smoke_passed_awaiting_full", "smoke_gate": str(gate), "completed_at": now()})
            atomic_json(state_path, state)
            return
        if not (full / "metrics.json").is_file():
            state.update({"status": "running_full", "updated_at": now()})
            atomic_json(state_path, state)
            command = inference_command(task, DATA, full, "external_test")
            if full.is_dir():
                command.append("--resume")
            execute(command, root / "full.log", env)
        state.update({"status": "completed", "full_metrics": str(full / "metrics.json"), "completed_at": now()})
        atomic_json(state_path, state)
    except Exception as error:
        state.update({"status": "failed", "error": repr(error), "failed_at": now()})
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    main()
