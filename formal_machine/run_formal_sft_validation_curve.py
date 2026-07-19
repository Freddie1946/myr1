#!/usr/bin/env python3
"""Audited validation-only curve for base plus formal n=2000/n=3000 epoch snapshots."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml


REPO_AT_IMPORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_AT_IMPORT / "scripts"))
from pathmmu_rewards import accuracy_reward, choice_letter, format_reward  # noqa: E402


BASE_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
PARENT_RUNS = {
    2000: "formal_sft_n2000_seed0042_20260717_140336",
    3000: "formal_sft_n3000_seed0042_20260717_014544",
}
EXPECTED_CONFIG_HASHES = {
    2000: "e495c64057a810e1c21bb44279e8b81f8159115eb5554aba7abc9d62cbccc8bd",
    3000: "d15428f642fdc522254e7b3322c977d5bb985e6516df0350c816d988638ab273",
}
VALIDATION_DATA_SHA256 = "6434da3e89e81c4e6a01736a1eda885858b56c28f8bfef284bd730693f37a2ca"
CURVE_CODE_MANIFEST = "sft_validation_curve_manifest_20260719_152742.json"
FORMAL_GPU_IDS = list(range(8))
SFT_GATES = {
    "training_completed", "final_checkpoint_saved", "final_checkpoint_reloaded",
    "trainability_freeze_gate", "losses_finite", "gradients_finite_nonzero",
    "language_tensor_changed", "visual_tensor_exactly_equal",
    "gradient_checkpointing_observed", "all_epoch_model_snapshots_saved",
    "all_epoch_snapshot_metadata_present", "latest_two_full_resume_checkpoints_retained",
    "disk_reserve_maintained", "test_not_accessed",
}


class CurveStop(RuntimeError):
    """A condition that invalidates or stops the validation curve."""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_yaml(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise CurveStop(f"missing YAML file: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CurveStop(f"YAML is not a mapping: {path}")
    return payload


def verify_code_manifest(repo: Path, name: str, python: Path) -> None:
    manifest = repo / "protocol" / name
    command = [str(python), str(repo / "scripts/verify_code_hash_manifest.py"),
               "--repo-root", str(repo), "--manifest", str(manifest)]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        raise CurveStop(f"code hash manifest failed: {result.stdout}{result.stderr}")


def verify_parent(install: Path, count: int) -> tuple[Path, dict]:
    run_id = PARENT_RUNS[count]
    manifest_path = (
        install / "runs" / "stage1_sft" / f"n{count:04d}_seed0042" / run_id
        / "run_manifest.yaml"
    ).resolve()
    parent = read_yaml(manifest_path)
    fixed = {"run_id": run_id, "status": "completed", "stage": "stage1_sft",
             "formal_result": True, "test_accessed": False}
    mismatches = {key: {"expected": value, "actual": parent.get(key)}
                  for key, value in fixed.items() if parent.get(key) != value}
    if mismatches:
        raise CurveStop(f"n={count} parent mismatch: {mismatches}")
    data = parent.get("data", {})
    if (
        data.get("version") != "pathmmu_image_disjoint_v1"
        or data.get("dataset") != f"pathvlm_sft_n{count:04d}"
        or data.get("qa_count") != count
        or data.get("all_image_paths_exist") is not True
        or "test" in str(data.get("adapter", "")).lower()
    ):
        raise CurveStop(f"n={count} parent data mismatch")
    model = parent.get("model", {})
    if (
        model.get("base_id") != "Qwen/Qwen2.5-VL-7B-Instruct"
        or model.get("base_revision") != BASE_REVISION
    ):
        raise CurveStop(f"n={count} parent base model mismatch")
    training = parent.get("training", {})
    expected_training = {
        "seed": 42, "epochs": 10, "learning_rate": 2e-5,
        "finetuning_type": "full", "language_model_trainable": True,
        "vision_tower_frozen": True, "multimodal_projector_frozen": True,
        "backend": "deepspeed_zero2_gpu_fused_adamw_gc",
    }
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise CurveStop(f"n={count} parent training policy mismatch")
    gates = parent.get("gates", {})
    if set(gates) != SFT_GATES or any(gates.get(key) is not True for key in SFT_GATES):
        raise CurveStop(f"n={count} parent formal gates are not exactly all true")
    if parent.get("provenance", {}).get("source_config_sha256") != EXPECTED_CONFIG_HASHES[count]:
        raise CurveStop(f"n={count} parent source config hash mismatch")
    snapshots = parent.get("retention", {}).get("snapshots", [])
    stride = math.ceil(count / 8)
    if [item.get("global_step") for item in snapshots] != [stride * i for i in range(1, 11)]:
        raise CurveStop(f"n={count} parent snapshot steps mismatch")
    return manifest_path, parent


def verify_snapshot(snapshot: Path, expected_epoch: int, expected_step: int) -> dict:
    manifest_path = snapshot / "snapshot_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        payload.get("epoch") != float(expected_epoch)
        or payload.get("global_step") != expected_step
        or payload.get("resumable") is not False
    ):
        raise CurveStop(f"snapshot metadata mismatch: {snapshot}")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise CurveStop(f"snapshot file manifest is empty: {snapshot}")
    names = {entry.get("name") for entry in files}
    required = {"config.json", "model.safetensors.index.json", "preprocessor_config.json",
                "tokenizer_config.json", "trainer_state.json"}
    if not required.issubset(names):
        raise CurveStop(f"snapshot required files missing: {snapshot}")
    for entry in files:
        path = snapshot / str(entry.get("name"))
        if not path.is_file() or path.stat().st_size != entry.get("size_bytes"):
            raise CurveStop(f"snapshot file size mismatch: {path}")
        if sha256(path) != entry.get("sha256"):
            raise CurveStop(f"snapshot file hash mismatch: {path}")
    return {"manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
            "file_count": len(files), "verified": True}


def verify_base(repo: Path, install: Path) -> tuple[Path, dict]:
    model = (
        install / "models"
        / f"Qwen2.5-VL-7B-Instruct-{BASE_REVISION}"
    ).resolve()
    frozen = json.loads((repo / "protocol/base_model_manifest.json").read_text(encoding="utf-8"))
    if frozen.get("model_id") != "Qwen/Qwen2.5-VL-7B-Instruct" or frozen.get("revision") != BASE_REVISION:
        raise CurveStop("base model manifest identity mismatch")
    checks = {
        "config.json": frozen["config_sha256"],
        "model.safetensors.index.json": frozen["model_index_sha256"],
        "tokenizer_config.json": frozen["tokenizer_config_sha256"],
    }
    for name, expected in checks.items():
        if sha256(model / name) != expected:
            raise CurveStop(f"base model hash mismatch: {name}")
    return model, {"manifest": str(repo / "protocol/base_model_manifest.json"),
                   "manifest_sha256": sha256(repo / "protocol/base_model_manifest.json"),
                   "verified_hashes": checks}


def build_jobs(repo: Path, install: Path) -> tuple[list[dict], list[dict]]:
    base, base_evidence = verify_base(repo, install)
    jobs = [{"ordinal": 0, "label": "base", "kind": "base", "sample_count": 0,
             "epoch": 0, "global_step": 0, "checkpoint": str(base),
             "checkpoint_evidence": base_evidence}]
    parents = []
    ordinal = 1
    for count in (2000, 3000):
        manifest_path, parent = verify_parent(install, count)
        parents.append({"sample_count": count, "manifest": str(manifest_path),
                        "manifest_sha256": sha256(manifest_path), "run_id": parent["run_id"]})
        stride = math.ceil(count / 8)
        root = manifest_path.parent / "epoch_snapshots"
        for epoch in range(1, 11):
            step = stride * epoch
            snapshot = root / f"checkpoint-{step}"
            evidence = verify_snapshot(snapshot, epoch, step)
            jobs.append({"ordinal": ordinal, "label": f"n{count:04d}_epoch{epoch:02d}",
                         "kind": "epoch_snapshot", "sample_count": count, "epoch": epoch,
                         "global_step": step, "checkpoint": str(snapshot),
                         "checkpoint_evidence": evidence})
            ordinal += 1
    if len(jobs) != 21:
        raise CurveStop(f"expected 21 validation jobs, found {len(jobs)}")
    return jobs, parents


def gpu_inventory(selected: list[int]) -> dict:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False,
    )
    if result.returncode:
        raise CurveStop(f"nvidia-smi inventory failed: {result.stderr.strip()}")
    rows, uuids = [], set()
    for line in result.stdout.splitlines():
        index, uuid, name, total, used, free, util = [item.strip() for item in line.split(",")]
        if int(index) in selected:
            rows.append({"index": int(index), "uuid": uuid, "name": name,
                         "memory_total_mib": int(total), "memory_used_mib": int(used),
                         "memory_free_mib": int(free), "utilization_percent": int(util)})
            uuids.add(uuid)
    if [row["index"] for row in rows] != selected:
        raise CurveStop(f"selected GPU inventory mismatch: {rows}")
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False,
    )
    if processes.returncode:
        raise CurveStop(f"nvidia-smi process query failed: {processes.stderr.strip()}")
    conflicts = []
    for line in processes.stdout.splitlines():
        if line.strip():
            uuid, pid, name, used = [item.strip() for item in line.split(",")]
            if uuid in uuids:
                conflicts.append({"gpu_uuid": uuid, "pid": int(pid), "process_name": name,
                                  "used_memory_mib": int(used)})
    if conflicts:
        raise CurveStop(f"selected GPUs have compute processes: {conflicts}")
    return {"gpus": rows, "compute_processes": []}


def wrapped(text: str):
    return [[{"role": "assistant", "content": text}]]


def audit_results(results: Path, records: list[dict]) -> tuple[dict, dict]:
    predictions = results / "predictions.jsonl"
    metrics_path = results / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != 385 or len(records) != 385:
        raise CurveStop(f"prediction count mismatch: {len(rows)}")
    accuracy_values, format_values = [], []
    for index, (row, record) in enumerate(zip(rows, records)):
        if row.get("index") != index:
            raise CurveStop(f"prediction index mismatch at {index}")
        for key in ("image", "problem", "solution"):
            if row.get(key) != record.get(key):
                raise CurveStop(f"prediction source field mismatch at {index}: {key}")
        completion = str(row.get("completion", ""))
        if not completion:
            raise CurveStop(f"empty completion at {index}")
        accuracy = accuracy_reward(wrapped(completion), [record["solution"]])[0]
        form = format_reward(wrapped(completion))[0]
        predicted = choice_letter(completion)
        target = choice_letter(record["solution"])
        if (
            row.get("accuracy_reward") != accuracy
            or row.get("format_reward") != form
            or row.get("predicted_choice") != predicted
            or row.get("target_choice") != target
        ):
            raise CurveStop(f"offline parser mismatch at prediction {index}")
        accuracy_values.append(accuracy)
        format_values.append(form)
    accuracy_mean = sum(accuracy_values) / len(accuracy_values)
    format_mean = sum(format_values) / len(format_values)
    if (
        metrics.get("count") != 385 or metrics.get("do_sample") is not False
        or metrics.get("test_accessed") is not False
        or abs(metrics.get("mean_accuracy_reward") - accuracy_mean) > 1e-12
        or abs(metrics.get("mean_format_reward") - format_mean) > 1e-12
    ):
        raise CurveStop("metrics/offline rescore mismatch")
    audit = {"prediction_count": 385, "indices_exact": True, "source_fields_exact": True,
             "parser_consistency": True, "accuracy_correct": int(sum(accuracy_values)),
             "format_correct": int(sum(format_values))}
    return metrics, audit


def run_job(job: dict, gpu: int, repo: Path, python: Path, data: Path, records: list[dict],
            jobs_root: Path, stop: threading.Event) -> dict:
    if stop.is_set():
        return {**job, "status": "not_started_after_peer_failure", "formal_result": False}
    job_dir = jobs_root / f"{job['ordinal']:02d}_{job['label']}"
    job_dir.mkdir()
    results = job_dir / "results"
    command = [str(python), str(repo / "scripts/infer_and_score_pathmmu.py"),
               "--model", job["checkpoint"], "--data", str(data),
               "--output-dir", str(results), "--max-new-tokens", "192"]
    (job_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    payload = {**job, "status": "running", "formal_result": False, "gpu": gpu,
               "created_at": now_iso(), "generation": {"do_sample": False,
               "max_new_tokens": 192}, "test_accessed": False}
    manifest_path = job_dir / "run_manifest.yaml"
    write_yaml(manifest_path, payload)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1",
                "WANDB_MODE": "disabled", "PYTHONPATH": str(repo / "scripts")})
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        env.pop(key, None)
    try:
        with (job_dir / "inference.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=repo, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, text=True, check=False)
        if result.returncode:
            raise CurveStop(f"inference exited with {result.returncode}")
        metrics, audit = audit_results(results, records)
        gates = {"inference_completed": True, "exact_validation_count": True,
                 "deterministic_decoding": metrics.get("do_sample") is False,
                 "raw_predictions_saved": True, "offline_parser_consistency": True,
                 "checkpoint_integrity_verified": job["checkpoint_evidence"][
                     "verified" if job["kind"] == "epoch_snapshot" else "verified_hashes"
                 ] is not None,
                 "test_not_accessed": metrics.get("test_accessed") is False}
        payload.update({"status": "completed" if all(gates.values()) else "failed_gate",
                        "formal_result": all(gates.values()), "metrics": metrics,
                        "offline_audit": audit, "gates": gates,
                        "outputs": {"predictions": str(results / "predictions.jsonl"),
                                    "metrics": str(results / "metrics.json")},
                        "completed_at": now_iso()})
        write_yaml(manifest_path, payload)
        if not all(gates.values()):
            raise CurveStop(f"job gates failed: {gates}")
        return payload
    except Exception as exc:
        stop.set()
        payload.update({"status": "failed", "formal_result": False,
                        "failure": {"type": type(exc).__name__, "message": str(exc)},
                        "completed_at": now_iso()})
        write_yaml(manifest_path, payload)
        return payload


def worker(slot: int, gpu: int, jobs: list[dict], repo: Path, python: Path, data: Path,
           records: list[dict], jobs_root: Path, stop: threading.Event) -> list[dict]:
    time.sleep(slot * 15)
    completed = []
    for job in jobs:
        result = run_job(job, gpu, repo, python, data, records, jobs_root, stop)
        completed.append(result)
        if result.get("formal_result") is not True:
            break
    return completed


def select_epoch(rows: list[dict], count: int) -> dict:
    candidates = [row for row in rows if row.get("sample_count") == count]
    if len(candidates) != 10:
        raise CurveStop(f"cannot select n={count}: expected 10 completed epochs")
    return sorted(candidates, key=lambda row: (-row["metrics"]["mean_accuracy_reward"],
                                                -row["metrics"]["mean_format_reward"],
                                                row["epoch"]))[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--install-root", required=True, type=Path)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    repo, install = args.repo_root.resolve(), args.install_root.resolve()
    python = Path(sys.executable).resolve()
    gpus = [int(item) for item in args.gpus.split(",")]
    if gpus != FORMAL_GPU_IDS:
        raise CurveStop("formal validation curve is frozen to physical GPUs 0-7")
    status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                            text=True, capture_output=True, check=False)
    if status.returncode or status.stdout.strip():
        raise CurveStop(f"repository is not clean: {status.stdout.strip()}")
    verify_code_manifest(repo, CURVE_CODE_MANIFEST, python)
    preflight = json.loads((install / "reports/preflight_report.json").read_text(encoding="utf-8"))
    if preflight.get("passed") is not True:
        raise CurveStop("formal preflight is not passing")
    reward_test = subprocess.run([str(python), str(repo / "scripts/test_pathmmu_rewards.py")],
                                 cwd=repo / "scripts", text=True, capture_output=True, check=False)
    if reward_test.returncode:
        raise CurveStop(f"parser regression tests failed: {reward_test.stdout}{reward_test.stderr}")
    data = install / "data/pathmmu_image_disjoint_v1/rewritten_records/validation_0385.json"
    if sha256(data) != VALIDATION_DATA_SHA256:
        raise CurveStop("frozen validation data hash mismatch")
    records = json.loads(data.read_text(encoding="utf-8"))
    if len(records) != 385 or any(not Path(row["image"]).is_file() for row in records):
        raise CurveStop("validation count or image paths failed")
    jobs, parents = build_jobs(repo, install)
    hardware = gpu_inventory(gpus)
    if args.preflight_only:
        print(json.dumps({"passed": True, "job_count": len(jobs), "parents": parents,
                          "validation_data_sha256": VALIDATION_DATA_SHA256,
                          "hardware": hardware, "test_accessed": False},
                         ensure_ascii=False, indent=2))
        return
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = (install / "runs" / "stage1_validation_curves"
               / f"sft_base_n2000_n3000_seed0042_{timestamp}")
    run_dir.mkdir(parents=True)
    jobs_root = run_dir / "jobs"
    jobs_root.mkdir()
    manifest_path = run_dir / "run_manifest.yaml"
    manifest = {"schema_version": 1, "run_id": run_dir.name, "status": "running",
                "stage": "stage1_sft_validation_curve", "formal_result": False,
                "created_at": now_iso(), "scope": {"base": True, "sample_counts": [2000, 3000],
                "epochs": list(range(1, 11)), "job_count": 21}, "parents": parents,
                "data": {"version": "pathmmu_image_disjoint_v1", "split": "validation_0385",
                         "count": 385, "path": str(data), "sha256": VALIDATION_DATA_SHA256},
                "generation": {"do_sample": False, "max_new_tokens": 192},
                "selection_rule": "per sample count: maximum validation accuracy; tie: maximum format; tie: earliest epoch",
                "hardware": hardware, "jobs": jobs,
                "provenance": {"repo_commit": subprocess.run(
                    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
                    capture_output=True, check=True).stdout.strip(),
                    "code_manifest": str(repo / "protocol" / CURVE_CODE_MANIFEST),
                    "inference_sha256": sha256(repo / "scripts/infer_and_score_pathmmu.py"),
                    "reward_sha256": sha256(repo / "scripts/pathmmu_rewards.py")},
                "parser_regression_tests": {"passed": True, "stdout": reward_test.stdout},
                "test_accessed": False}
    write_yaml(manifest_path, manifest)
    assignments = [[job for index, job in enumerate(jobs) if index % len(gpus) == slot]
                   for slot in range(len(gpus))]
    stop = threading.Event()
    results = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = [executor.submit(worker, slot, gpu, assignments[slot], repo, python,
                                       data, records, jobs_root, stop)
                       for slot, gpu in enumerate(gpus)]
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        results.sort(key=lambda row: row["ordinal"])
        completed = [row for row in results if row.get("formal_result") is True]
        gates = {"exact_21_jobs_completed": len(completed) == 21,
                 "all_job_gates_pass": len(completed) == 21,
                 "all_raw_predictions_saved": len(completed) == 21 and all(
                     Path(row["outputs"]["predictions"]).is_file() for row in completed),
                 "all_offline_parser_checks_pass": len(completed) == 21 and all(
                     row["gates"]["offline_parser_consistency"] for row in completed),
                 "test_not_accessed": True}
        if len(completed) == 21:
            selections = {str(count): {"label": (selected := select_epoch(completed, count))["label"],
                                       "epoch": selected["epoch"],
                                       "global_step": selected["global_step"],
                                       "metrics": selected["metrics"]}
                          for count in (2000, 3000)}
        else:
            selections = {}
        manifest.update({"results": results, "selections": selections, "gates": gates,
                         "outputs": {"run_dir": str(run_dir), "jobs_dir": str(jobs_root)},
                         "status": "completed" if all(gates.values()) else "failed_gate",
                         "formal_result": all(gates.values()), "completed_at": now_iso()})
        write_yaml(manifest_path, manifest)
        if not all(gates.values()):
            raise CurveStop(f"curve gates failed: {gates}")
    except Exception as exc:
        if manifest.get("status") == "running":
            manifest.update({"status": "failed", "formal_result": False,
                             "failure": {"type": type(exc).__name__, "message": str(exc)},
                             "completed_at": now_iso()})
            write_yaml(manifest_path, manifest)
        raise
    print(json.dumps({"manifest": str(manifest_path), "selections": manifest["selections"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
