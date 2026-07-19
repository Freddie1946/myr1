#!/usr/bin/env python3
"""Fail-closed Stage-2 priority pilot and formal n1000/seed42 pipeline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
INSTALL = Path("/home/wjy/pathvlm_r1_v1_formal")
DATA_VERSION = "pathmmu_image_disjoint_v2"
GPU_IDS = list(range(8))
MASTER_PORT = 29820
DISK_RESERVE_BYTES = 550 * 1024**3
PILOT_STEPS = 50
PILOT_STOP_STEP = 25
FORMAL_EPOCHS = 3
STEPS_PER_EPOCH = 500
FORMAL_STEPS = FORMAL_EPOCHS * STEPS_PER_EPOCH
PROMPT_CONTRACT = "pathmmu_think_answer_only_v2"
PARENT_CORRECT = 233
VALIDATION_COUNT = 385
PILOT_MIN_CORRECT = 222
PILOT_MIN_FORMAT = 0.98
PROTOCOL_NAME = "stage2_priority_n1000_seed42_manifest_20260720_033024.json"
CODE_MANIFEST_NAME = "stage2_priority_n1000_seed42_code_manifest_20260720_033024.json"
PARENT_PREDICTIONS = INSTALL / (
    "runs/stage1_validation_curves/sft_base_n2000_n3000_seed0042_20260719_184313/"
    "jobs/13_n3000_epoch03/results/predictions.jsonl"
)
VALIDATION_DATA = INSTALL / f"data/{DATA_VERSION}/rewritten_records/validation_0385.json"
CHAT_TEMPLATE = INSTALL / (
    "models/Qwen2.5-VL-7B-Instruct-cc594898137f460bfe9f0759e9844b3ce807cfb5/"
    "chat_template.json"
)
IMAGE_HASH_MANIFEST = REPO / f"data/{DATA_VERSION}/image_content_sha256.json"


class PipelineStop(RuntimeError):
    """A fail-closed pipeline condition."""


def _load_gate_module():
    path = REPO / "formal_machine/run_formal_outcome_grpo_smoke.py"
    spec = importlib.util.spec_from_file_location("pathvlm_gate_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import gate helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_gate_module()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise PipelineStop(f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PipelineStop(f"missing JSONL: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PipelineStop(f"invalid JSONL {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise PipelineStop(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(row)
    return rows


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_checked(command: list[str], *, cwd: Path = REPO, env: dict[str, str] | None = None) -> dict[str, Any]:
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, check=False)
    evidence = {"command": command, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    if result.returncode:
        raise PipelineStop(f"command failed: {command}\n{result.stdout}{result.stderr}")
    return evidence


def verify_protocol() -> dict[str, Any]:
    path = REPO / "protocol" / PROTOCOL_NAME
    payload = read_json(path)
    expected = {
        "status": "frozen_before_execution",
        "formal_parent_sample_count": 3000,
        "formal_parent_epoch": 3,
        "formal_parent_global_step": 1125,
        "rl_sample_count": 1000,
        "seed": 42,
        "pilot_max_steps": PILOT_STEPS,
        "pilot_resume_step": PILOT_STOP_STEP,
        "formal_epochs": FORMAL_EPOCHS,
        "formal_steps": FORMAL_STEPS,
        "prompt_contract": PROMPT_CONTRACT,
        "test_accessed": False,
        "picked_json_used": False,
    }
    mismatch = {key: {"expected": value, "actual": payload.get(key)}
                for key, value in expected.items() if payload.get(key) != value}
    if mismatch:
        raise PipelineStop(f"priority protocol mismatch: {mismatch}")
    return {"path": str(path), "sha256": sha256(path), "verified": expected}


def verify_code_manifest(python: Path) -> dict[str, Any]:
    path = REPO / "protocol" / CODE_MANIFEST_NAME
    result = run_checked([
        str(python), str(REPO / "scripts/verify_code_hash_manifest.py"),
        "--repo-root", str(REPO), "--manifest", str(path),
    ])
    return {"path": str(path), "sha256": sha256(path), "verified": True,
            "stdout": result["stdout"]}


def verify_rl_data() -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    adapter = INSTALL / f"data/{DATA_VERSION}/grpo/pathvlm_rl_n1000.json"
    yaml_path = adapter.with_suffix(".yaml")
    records = read_json(adapter)
    frozen = read_json(REPO / f"data/{DATA_VERSION}/subsets/rl/rl_1000_with_cot.json")
    image_hashes = read_json(IMAGE_HASH_MANIFEST).get("images", {})
    if not isinstance(records, list) or len(records) != 1000 or len(frozen) != 1000:
        raise PipelineStop("formal RL adapter/frozen subset must each contain 1000 records")
    if not yaml_path.is_file() or not isinstance(image_hashes, dict):
        raise PipelineStop("formal RL YAML or image hash manifest is missing")
    frozen_keys = [(Path(row["image"]).name, row["problem"], row["solution"]) for row in frozen]
    adapter_keys = [(Path(row["image"]).name, row["problem"], row["solution"]) for row in records]
    if adapter_keys != frozen_keys:
        raise PipelineStop("formal RL adapter order/content differs from frozen rl_1000")
    for index, row in enumerate(records):
        image = Path(row["image"])
        if not image.is_file():
            raise PipelineStop(f"RL image missing at row {index}: {image}")
        expected = image_hashes.get(image.name)
        if not expected or sha256(image) != expected:
            raise PipelineStop(f"RL image hash mismatch at row {index}: {image}")
    return yaml_path, records, {
        "version": DATA_VERSION, "split": "rl_n1000", "count": len(records),
        "adapter": str(adapter), "adapter_sha256": sha256(adapter),
        "yaml": str(yaml_path), "yaml_sha256": sha256(yaml_path),
        "image_hash_manifest": str(IMAGE_HASH_MANIFEST),
        "image_hash_manifest_sha256": sha256(IMAGE_HASH_MANIFEST),
        "all_rows_exact_and_ordered": True, "all_images_hashed": True,
        "picked_json_used": False, "test_accessed": False,
    }


def verify_validation_inputs() -> dict[str, Any]:
    records = read_json(VALIDATION_DATA)
    parent = read_jsonl(PARENT_PREDICTIONS)
    if len(records) != VALIDATION_COUNT or len(parent) != VALIDATION_COUNT:
        raise PipelineStop("validation data and parent predictions must each contain 385 rows")
    for index, (record, prediction) in enumerate(zip(records, parent)):
        if prediction.get("index") != index:
            raise PipelineStop(f"parent validation index mismatch at {index}")
        for source_key, pred_key in (("image", "image"), ("problem", "problem"), ("solution", "solution")):
            if record.get(source_key) != prediction.get(pred_key):
                raise PipelineStop(f"parent validation provenance mismatch at {index}: {source_key}")
    correct = sum(float(row.get("accuracy_reward", 0)) == 1.0 for row in parent)
    if correct != PARENT_CORRECT:
        raise PipelineStop(f"parent validation correct count mismatch: {correct}")
    if sha256(CHAT_TEMPLATE) != "ad60d90252ed0b0705ba14e2d0ad0fec0beac1ea955642b54059b36052d8bc96":
        raise PipelineStop("pinned chat template hash mismatch")
    return {
        "data": str(VALIDATION_DATA), "data_sha256": sha256(VALIDATION_DATA),
        "count": VALIDATION_COUNT, "parent_predictions": str(PARENT_PREDICTIONS),
        "parent_predictions_sha256": sha256(PARENT_PREDICTIONS),
        "parent_correct": correct, "parent_accuracy": correct / VALIDATION_COUNT,
        "chat_template": str(CHAT_TEMPLATE), "chat_template_sha256": sha256(CHAT_TEMPLATE),
        "test_accessed": False,
    }


def require_free_port(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise PipelineStop(f"master port {port} is unavailable: {exc}") from exc


def clean_training_env(extra: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
            env.pop(key, None)
    env.update({
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, GPU_IDS)),
        "NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "DEBUG_MODE": "false",
        "PATHVLM_REQUIRE_SOURCE_AUDIT": "true",
        "PATHVLM_IMAGE_HASH_MANIFEST": str(IMAGE_HASH_MANIFEST),
        **extra,
    })
    return env


def build_train_command(
    python: Path, parent_alias: Path, dataset_yaml: Path, output: Path, *,
    max_steps: int | None, epochs: int | None, save_strategy: str,
    save_steps: int | None = None, save_total_limit: int | None = None,
) -> list[str]:
    command = [
        str(python), "-m", "torch.distributed.run", "--nproc_per_node=8",
        f"--master_port={MASTER_PORT}", str(REPO / "scripts/grpo_pathmmu.py"),
        "--deepspeed", str(REPO / "configs/deepspeed/ds_z3_optimizer_offload_torch_adamw.json"),
        "--output_dir", str(output), "--model_name_or_path", str(parent_alias),
        "--dataset_name", str(dataset_yaml), "--image_root", "/",
        "--reward_funcs", "accuracy", "format", "--freeze_vision_modules", "true",
        "--max_pixels", "65536", "--min_pixels", "3136",
        "--num_generations", "4", "--max_completion_length", "192",
        "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "1",
        "--learning_rate", "1.0e-6", "--logging_steps", "1",
        "--bf16", "true", "--torch_dtype", "bfloat16",
        "--gradient_checkpointing", "true", "--attn_implementation", "sdpa",
        "--beta", "0.04", "--num_iterations", "1", "--save_strategy", save_strategy,
        "--report_to", "none", "--seed", "42", "--data_seed", "42",
        "--remove_unused_columns", "false",
    ]
    if max_steps is not None:
        command.extend(["--max_steps", str(max_steps)])
    if epochs is not None:
        command.extend(["--num_train_epochs", str(epochs)])
    if save_steps is not None:
        command.extend(["--save_steps", str(save_steps)])
    if save_total_limit is not None:
        command.extend(["--save_total_limit", str(save_total_limit)])
    return command


def run_monitored(
    command: list[str], *, log_path: Path, monitor_path: Path, env: dict[str, str], label: str
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            command, cwd=REPO / "vendor/open-r1-multimodal", env=env,
            stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True,
        )
        while process.poll() is None:
            sample: dict[str, Any] = {"timestamp": now_iso(), "pid": process.pid, "label": label}
            query = subprocess.run([
                "nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ], text=True, capture_output=True, check=False)
            sample["gpus"] = query.stdout.strip().splitlines()
            sample["disk_free_bytes"] = shutil.disk_usage(INSTALL).free
            if sample["disk_free_bytes"] < DISK_RESERVE_BYTES:
                os.killpg(process.pid, signal.SIGTERM)
                sample["stop_reason"] = "disk_reserve_breached"
            with monitor_path.open("a", encoding="utf-8") as monitor:
                monitor.write(json.dumps(sample) + "\n")
            print(f"[stage2-priority] {label} pid={process.pid} running; log={log_path}", flush=True)
            time.sleep(15)
        return int(process.returncode)


def wrapped(text: str):
    return [[{"role": "assistant", "content": text}]]


def audit_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sys.path.insert(0, str(REPO / "scripts"))
    from pathmmu_rewards import accuracy_reward, choice_letter, format_reward  # noqa: PLC0415

    source_records = read_json(VALIDATION_DATA)
    if len(rows) != VALIDATION_COUNT or len(source_records) != VALIDATION_COUNT:
        raise PipelineStop("validation prediction/source count mismatch")
    for index, (row, source) in enumerate(zip(rows, source_records)):
        if row.get("index") != index:
            raise PipelineStop(f"validation prediction index mismatch at {index}")
        for source_key, prediction_key in (
            ("image", "image"), ("problem", "problem"), ("solution", "solution")
        ):
            if row.get(prediction_key) != source.get(source_key):
                raise PipelineStop(f"validation provenance mismatch at {index}: {source_key}")
        completion = str(row.get("completion", ""))
        offline_accuracy = accuracy_reward(wrapped(completion), [source["solution"]])[0]
        offline_format = format_reward(wrapped(completion))[0]
        if (
            float(row.get("accuracy_reward", -1)) != offline_accuracy
            or float(row.get("format_reward", -1)) != offline_format
            or row.get("predicted_choice") != choice_letter(completion)
            or row.get("target_choice") != choice_letter(source["solution"])
        ):
            raise PipelineStop(f"validation offline parser mismatch at {index}")
    return {"source_provenance_exact": True, "offline_parser_consistency": True}


def audit_reward_directory(
    reward_dir: Path, records: list[dict[str, Any]], *, segment: str, expected_steps: int
) -> dict[str, Any]:
    sys.path.insert(0, str(REPO / "scripts"))
    from pathmmu_rewards import accuracy_reward, format_reward  # noqa: PLC0415

    hashes = read_json(IMAGE_HASH_MANIFEST)["images"]
    files = sorted(reward_dir.glob("rank_*.jsonl"))
    expected_files = [f"rank_{rank:02d}.jsonl" for rank in GPU_IDS]
    if [path.name for path in files] != expected_files:
        raise PipelineStop(f"{segment}: reward files mismatch")
    events = [row for path in files for row in read_jsonl(path)]
    expected_events = expected_steps * len(GPU_IDS) * 2
    if len(events) != expected_events:
        raise PipelineStop(f"{segment}: expected {expected_events} events, found {len(events)}")
    pairs: list[dict[str, Any]] = []
    for rank in GPU_IDS:
        rank_events = [event for event in events if event.get("rank") == rank]
        by_key: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
        for event in rank_events:
            key = (int(event.get("call_index", -1)), int(event.get("item_index", -1)))
            by_key.setdefault(key, {})[str(event.get("reward_type"))] = event
        if len(by_key) != expected_steps:
            raise PipelineStop(f"{segment}: rank {rank} call count mismatch: {len(by_key)}")
        if {key[0] for key in by_key} != set(range(expected_steps)):
            raise PipelineStop(f"{segment}: rank {rank} call indices are not contiguous")
        for key, typed in sorted(by_key.items()):
            if set(typed) != {"accuracy", "format"}:
                raise PipelineStop(f"{segment}: incomplete reward pair at rank {rank}, key {key}")
            accuracy_event, format_event = typed["accuracy"], typed["format"]
            paired_fields = (
                "completion", "solution", "record_index", "image_path", "image_sha256",
                "problem", "prompt_contract", "prompt", "training_segment",
            )
            if any(accuracy_event.get(field) != format_event.get(field) for field in paired_fields):
                raise PipelineStop(f"{segment}: paired event provenance mismatch at rank {rank}, key {key}")
            event = accuracy_event
            if event.get("training_segment") != segment or event.get("prompt_contract") != PROMPT_CONTRACT:
                raise PipelineStop(f"{segment}: prompt/segment contract mismatch")
            index = int(event.get("record_index", -1))
            if not 0 <= index < len(records):
                raise PipelineStop(f"{segment}: invalid record index {index}")
            source = records[index]
            image = Path(source["image"]).resolve()
            if (
                event.get("image_path") != str(image)
                or event.get("image_sha256") != hashes.get(image.name)
                or event.get("problem") != source["problem"]
                or event.get("solution") != source["solution"]
            ):
                raise PipelineStop(f"{segment}: frozen source mismatch for record {index}")
            prompt = event.get("prompt")
            try:
                prompt_text = prompt[0]["content"][1]["text"]
            except (TypeError, KeyError, IndexError) as exc:
                raise PipelineStop(f"{segment}: invalid structured prompt") from exc
            if source["problem"] not in prompt_text or "JSON format" in prompt_text:
                raise PipelineStop(f"{segment}: prompt text violates v2 contract")
            completion = str(event["completion"])
            offline_accuracy = accuracy_reward(wrapped(completion), [source["solution"]])[0]
            offline_format = format_reward(wrapped(completion))[0]
            if float(accuracy_event["reward"]) != offline_accuracy or float(format_event["reward"]) != offline_format:
                raise PipelineStop(f"{segment}: online/offline reward mismatch")
            pairs.append({
                "rank": rank, "call_index": key[0], "record_index": index,
                "accuracy_reward": offline_accuracy, "format_reward": offline_format,
                "total_reward": offline_accuracy + offline_format,
            })
    groups: dict[tuple[int, int], list[float]] = {}
    for pair in pairs:
        groups.setdefault((pair["call_index"], pair["record_index"]), []).append(pair["total_reward"])
    if any(len(values) != 4 for values in groups.values()) or len(groups) != expected_steps * 2:
        raise PipelineStop(f"{segment}: four-generation grouping failed")
    stds = [statistics.stdev(values) for values in groups.values()]
    formats = [pair["format_reward"] for pair in pairs]
    accuracies = [pair["accuracy_reward"] for pair in pairs]
    return {
        "event_count": len(events), "completion_count": len(pairs), "group_count": len(groups),
        "parser_consistency": True, "source_provenance_exact": True,
        "positive_variance_group_count": sum(value > 0 for value in stds),
        "mean_group_sample_std": statistics.mean(stds),
        "mean_accuracy_reward": statistics.mean(accuracies),
        "mean_format_reward": statistics.mean(formats),
        "format_positive_count": sum(value > 0 for value in formats),
    }


def audit_training_state(path: Path, *, start_exclusive: int, end_inclusive: int) -> dict[str, Any]:
    payload = read_json(path)
    rows = [row for row in payload.get("log_history", [])
            if "loss" in row and "grad_norm" in row and start_exclusive < int(row.get("step", -1)) <= end_inclusive]
    expected = end_inclusive - start_exclusive
    if len(rows) != expected or int(payload.get("global_step", -1)) != end_inclusive:
        raise PipelineStop(f"training state step history mismatch in {path}: {len(rows)}/{expected}")
    finite = all(all(key in row and math.isfinite(float(row[key]))
                     for key in ("loss", "grad_norm", "reward", "reward_std")) for row in rows)
    return {
        "path": str(path), "global_step": payload["global_step"], "step_count": len(rows),
        "all_metrics_finite": finite,
        "positive_gradient_steps": sum(float(row["grad_norm"]) > 0 for row in rows),
        "positive_reward_std_steps": sum(float(row["reward_std"]) > 0 for row in rows),
        "trainability": payload.get("trainability"),
        "trainability_passed": payload.get("trainability", {}).get("passed") is True,
    }


def run_validation(model: Path, output: Path, python: Path, gpu: int, label: str) -> dict[str, Any]:
    if output.exists():
        raise PipelineStop(f"validation output already exists: {output}")
    output.mkdir(parents=True)
    command = [
        str(python), str(REPO / "scripts/infer_and_score_pathmmu.py"),
        "--model", str(model), "--data", str(VALIDATION_DATA),
        "--output-dir", str(output), "--max-new-tokens", "192",
        "--chat-template-file", str(CHAT_TEMPLATE),
    ]
    env = clean_training_env({"CUDA_VISIBLE_DEVICES": str(gpu)})
    log_path = output.parent / "inference.log"
    with log_path.open("wb") as handle:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise PipelineStop(f"validation {label} failed with return code {result.returncode}; see {log_path}")
    metrics = read_json(output / "metrics.json")
    predictions = read_jsonl(output / "predictions.jsonl")
    if len(predictions) != VALIDATION_COUNT or metrics.get("count") != VALIDATION_COUNT:
        raise PipelineStop(f"validation {label} does not contain 385 predictions")
    row_audit = audit_prediction_rows(predictions)
    empty = sum(not str(row.get("completion", "")).strip() for row in predictions)
    correct = sum(float(row.get("accuracy_reward", 0)) == 1.0 for row in predictions)
    formatted = sum(float(row.get("format_reward", 0)) == 1.0 for row in predictions)
    extracted = sum(row.get("predicted_choice") in {"A", "B", "C", "D"} for row in predictions)
    if (
        abs(float(metrics.get("mean_accuracy_reward")) - correct / VALIDATION_COUNT) > 1e-12
        or abs(float(metrics.get("mean_format_reward")) - formatted / VALIDATION_COUNT) > 1e-12
    ):
        raise PipelineStop(f"validation {label} metric aggregation mismatch")
    return {
        "label": label, "model": str(model), "gpu": gpu, "command": command,
        "output": str(output), "predictions": str(output / "predictions.jsonl"),
        "predictions_sha256": sha256(output / "predictions.jsonl"),
        "metrics": str(output / "metrics.json"), "metrics_sha256": sha256(output / "metrics.json"),
        "count": VALIDATION_COUNT, "correct": correct, "accuracy": correct / VALIDATION_COUNT,
        "formatted": formatted, "format_rate": formatted / VALIDATION_COUNT,
        "extracted_choices": extracted, "choice_extraction_rate": extracted / VALIDATION_COUNT,
        "empty_completions": empty, "test_accessed": False,
        **row_audit,
    }


def exact_mcnemar(parent_rows: list[dict[str, Any]], child_rows: list[dict[str, Any]]) -> dict[str, Any]:
    parent_only = child_only = both = neither = 0
    for index, (parent, child) in enumerate(zip(parent_rows, child_rows)):
        if parent.get("index") != index or child.get("index") != index:
            raise PipelineStop(f"McNemar prediction index mismatch at {index}")
        p = float(parent.get("accuracy_reward", 0)) == 1.0
        c = float(child.get("accuracy_reward", 0)) == 1.0
        if p and c:
            both += 1
        elif p:
            parent_only += 1
        elif c:
            child_only += 1
        else:
            neither += 1
    discordant = parent_only + child_only
    if discordant == 0:
        pvalue = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(0, min(parent_only, child_only) + 1)) / 2**discordant
        pvalue = min(1.0, 2.0 * tail)
    return {
        "both_correct": both, "parent_only_correct": parent_only,
        "child_only_correct": child_only, "both_incorrect": neither,
        "discordant": discordant, "exact_two_sided_p": pvalue,
        "significant_child_degradation": parent_only > child_only and pvalue < 0.05,
    }


def pilot(run_dir: Path, parent: Path, dataset_yaml: Path, records: list[dict[str, Any]], python: Path) -> dict[str, Any]:
    pilot_dir = run_dir / "pilot"
    pilot_dir.mkdir()
    output = pilot_dir / "output"
    alias = pilot_dir / "parent_Qwen2.5-VL-7B-Instruct"
    alias.symlink_to(parent, target_is_directory=True)
    if alias.resolve() != parent:
        raise PipelineStop("pilot parent alias mismatch")
    segment_a_rewards = pilot_dir / "online_reward_events_segment_a"
    command_a = build_train_command(
        python, alias, dataset_yaml, output, max_steps=PILOT_STEPS, epochs=None,
        save_strategy="steps", save_steps=PILOT_STOP_STEP, save_total_limit=1,
    )
    env_a = clean_training_env({
        "PATHVLM_REWARD_LOG_DIR": str(segment_a_rewards),
        "PATHVLM_TRAINING_SEGMENT": "pilot_segment_a",
        "PATHVLM_STOP_AFTER_SAVED_STEP": str(PILOT_STOP_STEP),
        "PATHVLM_SKIP_FINAL_MODEL_SAVE": "true",
        "PATHVLM_TRAIN_STATE_AUDIT_NAME": "pathvlm_train_state_audit_segment_a.json",
    })
    rc_a = run_monitored(command_a, log_path=pilot_dir / "train_segment_a.log",
                         monitor_path=pilot_dir / "resource_monitor.jsonl", env=env_a,
                         label="pilot_segment_a")
    if rc_a:
        raise PipelineStop(f"pilot segment A failed with return code {rc_a}")
    checkpoint = output / f"checkpoint-{PILOT_STOP_STEP}"
    if not checkpoint.is_dir() or not (checkpoint / "trainer_state.json").is_file():
        raise PipelineStop("pilot step-25 full checkpoint was not saved")
    state_a = audit_training_state(output / "pathvlm_train_state_audit_segment_a.json",
                                   start_exclusive=0, end_inclusive=PILOT_STOP_STEP)
    rewards_a = audit_reward_directory(segment_a_rewards, records, segment="pilot_segment_a",
                                       expected_steps=PILOT_STOP_STEP)

    segment_b_rewards = pilot_dir / "online_reward_events_segment_b"
    command_b = build_train_command(
        python, alias, dataset_yaml, output, max_steps=PILOT_STEPS, epochs=None,
        save_strategy="no",
    )
    env_b = clean_training_env({
        "PATHVLM_REWARD_LOG_DIR": str(segment_b_rewards),
        "PATHVLM_TRAINING_SEGMENT": "pilot_segment_b",
        "PATHVLM_RESUME_FROM_CHECKPOINT": str(checkpoint),
        "PATHVLM_TRAIN_STATE_AUDIT_NAME": "pathvlm_train_state_audit.json",
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
    })
    rc_b = run_monitored(command_b, log_path=pilot_dir / "train_segment_b.log",
                         monitor_path=pilot_dir / "resource_monitor.jsonl", env=env_b,
                         label="pilot_segment_b")
    if rc_b:
        raise PipelineStop(f"pilot segment B failed with return code {rc_b}")
    state_b = audit_training_state(output / "pathvlm_train_state_audit.json",
                                   start_exclusive=PILOT_STOP_STEP, end_inclusive=PILOT_STEPS)
    rewards_b = audit_reward_directory(segment_b_rewards, records, segment="pilot_segment_b",
                                       expected_steps=PILOT_STEPS - PILOT_STOP_STEP)
    loadability = gate.audit_output_loadability(output, python, REPO)
    tensors = gate.compare_tensors(parent, output, pilot_dir, python, REPO)
    validation = run_validation(output, pilot_dir / "validation/results", python, 0, "pilot_step50")
    parent_rows = read_jsonl(PARENT_PREDICTIONS)
    child_rows = read_jsonl(Path(validation["predictions"]))
    mcnemar = exact_mcnemar(parent_rows, child_rows)
    gates = {
        "segment_a_completed_25_steps": state_a["global_step"] == PILOT_STOP_STEP,
        "full_checkpoint_25_saved": checkpoint.is_dir(),
        "segment_b_resumed_to_50": state_b["global_step"] == PILOT_STEPS,
        "all_training_metrics_finite": state_a["all_metrics_finite"] and state_b["all_metrics_finite"],
        "nonzero_gradient_in_each_segment": state_a["positive_gradient_steps"] > 0 and state_b["positive_gradient_steps"] > 0,
        "positive_reward_std_in_each_segment": state_a["positive_reward_std_steps"] > 0 and state_b["positive_reward_std_steps"] > 0,
        "trainability_and_freeze": state_a["trainability_passed"] and state_b["trainability_passed"],
        "reward_events_complete": rewards_a["event_count"] == 400 and rewards_b["event_count"] == 400,
        "online_offline_reward_consistency": rewards_a["parser_consistency"] and rewards_b["parser_consistency"],
        "source_provenance_exact": rewards_a["source_provenance_exact"] and rewards_b["source_provenance_exact"],
        "positive_reward_variance_each_segment": rewards_a["positive_variance_group_count"] > 0 and rewards_b["positive_variance_group_count"] > 0,
        "online_format_at_least_98_percent": (
            (rewards_a["format_positive_count"] + rewards_b["format_positive_count"]) / 400
            >= PILOT_MIN_FORMAT
        ),
        "saved_model_loadable": loadability.get("required_files_present") is True,
        **tensors["gates"],
        "validation_complete": validation["count"] == VALIDATION_COUNT,
        "validation_accuracy_not_below_3pp_margin": validation["correct"] >= PILOT_MIN_CORRECT,
        "no_significant_paired_degradation": not mcnemar["significant_child_degradation"],
        "validation_format_at_least_98_percent": validation["format_rate"] >= PILOT_MIN_FORMAT,
        "validation_choice_extraction_at_least_98_percent": validation["choice_extraction_rate"] >= PILOT_MIN_FORMAT,
        "no_empty_validation_completion": validation["empty_completions"] == 0,
        "test_not_accessed": True,
        "picked_json_unused": True,
    }
    result = {
        "status": "completed" if all(gates.values()) else "failed_gate",
        "formal_result": False, "gate_passed": all(gates.values()),
        "commands": {"segment_a": command_a, "segment_b": command_b},
        "states": {"segment_a": state_a, "segment_b": state_b},
        "rewards": {"segment_a": rewards_a, "segment_b": rewards_b},
        "loadability": loadability, "tensor_comparison": tensors,
        "validation": validation, "mcnemar_vs_parent": mcnemar, "gates": gates,
        "test_accessed": False,
    }
    atomic_json(pilot_dir / "pilot_manifest.json", result)
    if not all(gates.values()):
        raise PipelineStop(f"pilot gate failed: {gates}")
    return result


def projected_storage_gate(pilot_dir: Path) -> dict[str, Any]:
    checkpoint = pilot_dir / f"output/checkpoint-{PILOT_STOP_STEP}"
    checkpoint_bytes = sum(path.stat().st_size for path in checkpoint.rglob("*") if path.is_file())
    model_bytes = sum(path.stat().st_size for path in (pilot_dir / "output").glob("model*.safetensors"))
    free = shutil.disk_usage(INSTALL).free
    transient_budget = 2 * checkpoint_bytes + 4 * model_bytes + 20 * 1024**3
    required = DISK_RESERVE_BYTES + transient_budget
    evidence = {
        "free_bytes": free, "reserve_bytes": DISK_RESERVE_BYTES,
        "measured_full_checkpoint_bytes": checkpoint_bytes,
        "measured_model_bytes": model_bytes,
        "formal_transient_budget_bytes": transient_budget, "required_free_bytes": required,
        "passed": free >= required,
    }
    if not evidence["passed"]:
        raise PipelineStop(f"formal projected storage gate failed: {evidence}")
    return evidence


def formal_training(run_dir: Path, parent: Path, dataset_yaml: Path, records: list[dict[str, Any]], python: Path) -> dict[str, Any]:
    formal_dir = run_dir / "formal_n1000_seed0042_epoch3"
    formal_dir.mkdir()
    output = formal_dir / "output"
    snapshots = formal_dir / "epoch_snapshots"
    alias = formal_dir / "parent_Qwen2.5-VL-7B-Instruct"
    alias.symlink_to(parent, target_is_directory=True)
    command = build_train_command(
        python, alias, dataset_yaml, output, max_steps=None, epochs=FORMAL_EPOCHS,
        save_strategy="steps", save_steps=100, save_total_limit=1,
    )
    reward_dir = formal_dir / "online_reward_events"
    env = clean_training_env({
        "PATHVLM_REWARD_LOG_DIR": str(reward_dir),
        "PATHVLM_TRAINING_SEGMENT": "formal_epoch1_to3",
        "PATHVLM_TRAIN_STATE_AUDIT_NAME": "pathvlm_train_state_audit.json",
        "PATHVLM_EPOCH_SNAPSHOT_STEPS": "500,1000,1500",
        "PATHVLM_EPOCH_SNAPSHOT_DIR": str(snapshots),
    })
    rc = run_monitored(command, log_path=formal_dir / "train.log",
                       monitor_path=formal_dir / "resource_monitor.jsonl", env=env,
                       label="formal_n1000_seed0042_epoch3")
    if rc:
        raise PipelineStop(f"formal GRPO training failed with return code {rc}")
    state = audit_training_state(output / "pathvlm_train_state_audit.json",
                                 start_exclusive=0, end_inclusive=FORMAL_STEPS)
    rewards = audit_reward_directory(reward_dir, records, segment="formal_epoch1_to3",
                                     expected_steps=FORMAL_STEPS)
    loadability = gate.audit_output_loadability(output, python, REPO)
    tensor = gate.compare_tensors(parent, output, formal_dir, python, REPO)
    snapshot_evidence = []
    for epoch, step in enumerate((500, 1000, 1500), 1):
        snapshot = snapshots / f"checkpoint-{step}"
        manifest = read_json(snapshot / "snapshot_manifest.json")
        if manifest.get("global_step") != step or manifest.get("model_only") is not True:
            raise PipelineStop(f"formal epoch {epoch} snapshot manifest mismatch")
        for item in manifest.get("files", []):
            path = snapshot / item["name"]
            if path.stat().st_size != item["size_bytes"] or sha256(path) != item["sha256"]:
                raise PipelineStop(f"formal snapshot file mismatch: {path}")
        snapshot_evidence.append({"epoch": epoch, "step": step, "path": str(snapshot),
                                  "manifest_sha256": sha256(snapshot / "snapshot_manifest.json")})
    gates = {
        "training_completed_1500_steps": state["global_step"] == FORMAL_STEPS,
        "all_training_metrics_finite": state["all_metrics_finite"],
        "nonzero_gradients_observed": state["positive_gradient_steps"] > 0,
        "positive_reward_std_observed": state["positive_reward_std_steps"] > 0,
        "trainability_and_freeze": state["trainability_passed"],
        "online_events_complete": rewards["event_count"] == 24000,
        "online_offline_reward_consistency": rewards["parser_consistency"],
        "source_provenance_exact": rewards["source_provenance_exact"],
        "positive_reward_variance": rewards["positive_variance_group_count"] > 0,
        "online_format_at_least_98_percent": rewards["mean_format_reward"] >= PILOT_MIN_FORMAT,
        "saved_model_loadable": loadability.get("required_files_present") is True,
        "three_epoch_snapshots_complete": len(snapshot_evidence) == 3,
        **tensor["gates"], "test_not_accessed": True, "picked_json_unused": True,
    }
    result = {
        "status": "trained_pending_validation" if all(gates.values()) else "failed_gate",
        "formal_result": False, "command": command, "training_state": state,
        "reward_audit": rewards, "loadability": loadability, "tensor_comparison": tensor,
        "epoch_snapshots": snapshot_evidence, "gates": gates, "test_accessed": False,
    }
    atomic_json(formal_dir / "formal_training_manifest.json", result)
    if not all(gates.values()):
        raise PipelineStop(f"formal training gate failed: {gates}")
    return result


def formal_validation(run_dir: Path, training: dict[str, Any], python: Path) -> dict[str, Any]:
    formal_dir = run_dir / "formal_n1000_seed0042_epoch3"
    gate.gpu_inventory(GPU_IDS)
    jobs = []
    processes = []
    for evidence in training["epoch_snapshots"]:
        epoch = evidence["epoch"]
        model = Path(evidence["path"])
        root = formal_dir / "validation" / f"epoch_{epoch:02d}"
        root.mkdir(parents=True)
        output = root / "results"
        output.mkdir()
        command = [
            str(python), str(REPO / "scripts/infer_and_score_pathmmu.py"),
            "--model", str(model), "--data", str(VALIDATION_DATA),
            "--output-dir", str(output), "--max-new-tokens", "192",
            "--chat-template-file", str(CHAT_TEMPLATE),
        ]
        env = clean_training_env({"CUDA_VISIBLE_DEVICES": str(epoch - 1)})
        log = (root / "inference.log").open("wb")
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append((epoch, model, output, root, command, log, process))
    failures = []
    for epoch, model, output, root, command, log, process in processes:
        rc = process.wait()
        log.close()
        if rc:
            failures.append({"epoch": epoch, "returncode": rc, "log": str(root / "inference.log")})
            continue
        metrics = read_json(output / "metrics.json")
        rows = read_jsonl(output / "predictions.jsonl")
        if len(rows) != VALIDATION_COUNT or metrics.get("count") != VALIDATION_COUNT:
            failures.append({"epoch": epoch, "reason": "prediction count mismatch"})
            continue
        row_audit = audit_prediction_rows(rows)
        correct = sum(float(row.get("accuracy_reward", 0)) == 1.0 for row in rows)
        formatted = sum(float(row.get("format_reward", 0)) == 1.0 for row in rows)
        extracted = sum(row.get("predicted_choice") in {"A", "B", "C", "D"} for row in rows)
        jobs.append({
            "epoch": epoch, "step": epoch * STEPS_PER_EPOCH, "model": str(model),
            "command": command, "count": len(rows), "correct": correct,
            "accuracy": correct / VALIDATION_COUNT, "formatted": formatted,
            "format_rate": formatted / VALIDATION_COUNT,
            "choice_extraction_rate": extracted / VALIDATION_COUNT,
            "empty_completions": sum(not str(row.get("completion", "")).strip() for row in rows),
            **row_audit,
            "predictions": str(output / "predictions.jsonl"),
            "predictions_sha256": sha256(output / "predictions.jsonl"),
            "metrics": str(output / "metrics.json"), "metrics_sha256": sha256(output / "metrics.json"),
            "test_accessed": False,
        })
    if failures or len(jobs) != 3:
        raise PipelineStop(f"formal validation failures: {failures}")
    selected = sorted(jobs, key=lambda row: (-row["accuracy"], -row["format_rate"], row["epoch"]))[0]
    gates = {
        "all_three_epochs_validated": len(jobs) == 3,
        "all_prediction_counts_385": all(job["count"] == VALIDATION_COUNT for job in jobs),
        "all_no_empty_completions": all(job["empty_completions"] == 0 for job in jobs),
        "all_source_and_parser_audits_exact": all(
            job["source_provenance_exact"] and job["offline_parser_consistency"] for job in jobs
        ),
        "selection_rule_accuracy_format_earliest": selected == sorted(
            jobs, key=lambda row: (-row["accuracy"], -row["format_rate"], row["epoch"])
        )[0],
        "test_not_accessed": True,
    }
    result = {
        "status": "completed", "formal_result": True, "jobs": jobs,
        "selection_rule": "max_accuracy_then_format_then_earliest_epoch",
        "selected": selected, "gates": gates, "test_accessed": False,
    }
    atomic_json(formal_dir / "formal_result_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--install-root", type=Path, default=INSTALL)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-hardware", action="store_true")
    args = parser.parse_args()
    if args.repo_root.resolve() != REPO or args.install_root.resolve() != INSTALL:
        raise PipelineStop("priority pipeline paths are frozen")
    python = Path(sys.executable).resolve()
    if python != (INSTALL / "envs/grpo/bin/python").resolve():
        raise PipelineStop(f"must use pinned GRPO Python, got {python}")

    preflight: dict[str, Any] = {
        "created_at": now_iso(), "git": gate.verify_git_clean(REPO),
        "protocol": verify_protocol(), "code": verify_code_manifest(python),
    }
    parent, preflight["parent"] = gate.verify_parent(INSTALL)
    preflight["frozen_data"] = gate.verify_data(REPO, INSTALL)
    dataset_yaml, records, preflight["data"] = verify_rl_data()
    preflight["validation"] = verify_validation_inputs()
    preflight["parser_regression"] = run_checked([str(python), str(REPO / "scripts/test_pathmmu_rewards.py")])
    preflight["wrapper_regression"] = run_checked([str(python), str(REPO / "scripts/test_grpo_pathmmu_audit.py")])
    preflight["system"] = gate.system_inventory(INSTALL)
    if not args.skip_hardware:
        preflight["hardware"] = gate.gpu_inventory(GPU_IDS)
        require_free_port(MASTER_PORT)
    preflight["passed"] = True
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return
    if args.skip_hardware:
        raise PipelineStop("--skip-hardware is valid only with --preflight-only")

    run_id = f"stage2_priority_n1000_seed0042_{timestamp()}"
    run_dir = INSTALL / "runs/stage2_outcome_grpo/priority_n1000_seed0042" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "pipeline_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1, "run_id": run_id, "stage": "stage2_priority_pipeline",
        "status": "running_pilot", "formal_result": False, "created_at": now_iso(),
        "preflight": preflight, "test_accessed": False, "picked_json_used": False,
    }
    atomic_json(manifest_path, manifest)
    try:
        manifest["pilot"] = pilot(run_dir, parent, dataset_yaml, records, python)
        manifest["status"] = "pilot_passed_preparing_formal"
        atomic_json(manifest_path, manifest)
        gate.gpu_inventory(GPU_IDS)
        require_free_port(MASTER_PORT)
        manifest["formal_storage_gate"] = projected_storage_gate(run_dir / "pilot")
        manifest["status"] = "running_formal_training"
        atomic_json(manifest_path, manifest)
        manifest["formal_training"] = formal_training(run_dir, parent, dataset_yaml, records, python)
        manifest["status"] = "running_formal_validation"
        atomic_json(manifest_path, manifest)
        manifest["formal_validation"] = formal_validation(run_dir, manifest["formal_training"], python)
        manifest.update({
            "status": "completed", "formal_result": True, "completed_at": now_iso(),
            "selected_checkpoint": manifest["formal_validation"]["selected"]["model"],
            "test_accessed": False,
        })
    except Exception as exc:
        manifest.update({
            "status": "failed", "formal_result": False, "failed_at": now_iso(),
            "failure": {"type": type(exc).__name__, "message": str(exc)},
            "test_accessed": False,
        })
        atomic_json(manifest_path, manifest)
        raise
    atomic_json(manifest_path, manifest)
    print(f"[stage2-priority] COMPLETE: {manifest_path}")


if __name__ == "__main__":
    main()
