#!/usr/bin/env python3
"""Fail-closed 1024-token validation for every frozen seed-42 candidate."""

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
sys.path.insert(0, str(REPO_AT_IMPORT / "formal_machine"))
from pathmmu_rewards import accuracy_reward, choice_letter, format_reward  # noqa: E402
import run_formal_sft_validation_curve as legacy  # noqa: E402


BASE_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
VALIDATION_SHA256 = "6434da3e89e81c4e6a01736a1eda885858b56c28f8bfef284bd730693f37a2ca"
CHAT_TEMPLATE_SHA256 = "ad60d90252ed0b0705ba14e2d0ad0fec0beac1ea955642b54059b36052d8bc96"
MAX_NEW_TOKENS = 1024
FORMAL_GPUS = list(range(8))
CODE_MANIFEST = "validation_1024_all_candidates_code_manifest_20260721.json"
SFT_FINALS = {
    500: ("formal_sft_n0500_seed0042_20260715_230037",
          "f22b0ea946e8f41924436af2a746e49ac9ac07b74fb262a997edb29afe9d7d12"),
    1000: ("formal_sft_n1000_seed0042_20260716_043832",
           "889b99138b506ba9b3482194ed1a2fb680f37655fc4b624619cb001e7b7a7c18"),
}
RL_ROOT = Path(
    "runs/stage2_outcome_grpo/priority_n1000_seed0042/"
    "stage2_priority_n1000_seed0042_20260720_035156/formal_n1000_seed0042_epoch3"
)
RL_RESULT_SHA256 = "cd6f57b24b3f1cc424ddaf2e1d868d575fbf8c16875a28b1d25c0a415ab7cd97"


class ValidationStop(RuntimeError):
    """A condition that invalidates the corrected validation run."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def write_yaml(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def wrapped(text: str):
    return [[{"role": "assistant", "content": text}]]


def verify_model_directory(path: Path) -> dict:
    required = {"config.json", "model.safetensors.index.json", "preprocessor_config.json",
                "tokenizer_config.json"}
    missing = sorted(name for name in required if not (path / name).is_file())
    if missing:
        raise ValidationStop(f"model directory is incomplete: {path}: {missing}")
    index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_files = sorted(set(index.get("weight_map", {}).values()))
    if not weight_files or any(not (path / name).is_file() for name in weight_files):
        raise ValidationStop(f"model weight shards are incomplete: {path}")
    sizes = {name: (path / name).stat().st_size for name in weight_files}
    if any(size <= 0 for size in sizes.values()):
        raise ValidationStop(f"empty model shard: {path}")
    return {
        "verified": True,
        "required_file_hashes": {name: sha256(path / name) for name in sorted(required)},
        "weight_file_count": len(weight_files),
        "weight_total_bytes": sum(sizes.values()),
        "weight_file_sizes": sizes,
    }


def verify_sft_final(install: Path, count: int) -> tuple[Path, dict, dict]:
    run_id, expected_manifest_hash = SFT_FINALS[count]
    root = install / "runs/stage1_sft" / f"n{count:04d}_seed0042" / run_id
    manifest_path = root / "run_manifest.yaml"
    if sha256(manifest_path) != expected_manifest_hash:
        raise ValidationStop(f"n={count} SFT manifest hash mismatch")
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    fixed = {"run_id": run_id, "status": "completed", "stage": "stage1_sft",
             "formal_result": True, "test_accessed": False}
    if any(payload.get(key) != value for key, value in fixed.items()):
        raise ValidationStop(f"n={count} SFT final manifest identity mismatch")
    if payload.get("data", {}).get("qa_count") != count:
        raise ValidationStop(f"n={count} SFT data count mismatch")
    if payload.get("training", {}).get("seed") != 42:
        raise ValidationStop(f"n={count} SFT seed mismatch")
    gates = payload.get("gates", {})
    if not gates or any(value is not True for value in gates.values()):
        raise ValidationStop(f"n={count} SFT gates failed")
    checkpoint = Path(payload["outputs"]["final_checkpoint"]).resolve()
    evidence = verify_model_directory(checkpoint)
    evidence.update({"parent_manifest": str(manifest_path),
                     "parent_manifest_sha256": expected_manifest_hash})
    parent = {"family": "sft", "sample_count": count, "run_id": run_id,
              "manifest": str(manifest_path), "manifest_sha256": expected_manifest_hash}
    return checkpoint, evidence, parent


def verify_rl(install: Path) -> tuple[list[dict], dict]:
    root = install / RL_ROOT
    result_manifest = root / "formal_result_manifest.json"
    if sha256(result_manifest) != RL_RESULT_SHA256:
        raise ValidationStop("Outcome-GRPO result manifest hash mismatch")
    result = json.loads(result_manifest.read_text(encoding="utf-8"))
    if (result.get("status") != "completed" or result.get("formal_result") is not True
            or result.get("test_accessed") is not False
            or any(value is not True for value in result.get("gates", {}).values())):
        raise ValidationStop("Outcome-GRPO formal result gates failed")
    jobs = []
    for epoch in range(1, 4):
        step = epoch * 500
        checkpoint = root / "epoch_snapshots" / f"checkpoint-{step}"
        evidence = legacy.verify_snapshot(checkpoint, epoch, step)
        jobs.append({"label": f"outcome_grpo_n1000_epoch{epoch:02d}", "family": "outcome_grpo",
                     "sample_count": 1000, "epoch": epoch, "global_step": step,
                     "checkpoint": str(checkpoint), "checkpoint_evidence": evidence})
    parent = {"family": "outcome_grpo", "manifest": str(result_manifest),
              "manifest_sha256": RL_RESULT_SHA256}
    return jobs, parent


def build_jobs(repo: Path, install: Path) -> tuple[list[dict], list[dict]]:
    legacy_jobs, parents = legacy.build_jobs(repo, install)
    base = legacy_jobs[0]
    jobs = [{**base, "family": "base", "label": "base"}]
    for count in (500, 1000):
        checkpoint, evidence, parent = verify_sft_final(install, count)
        parents.append(parent)
        jobs.append({"label": f"sft_n{count:04d}_epoch10_final", "family": "sft",
                     "sample_count": count, "epoch": 10, "global_step": 0,
                     "checkpoint": str(checkpoint), "checkpoint_evidence": evidence})
    for job in legacy_jobs[1:]:
        jobs.append({**job, "family": "sft"})
    rl_jobs, rl_parent = verify_rl(install)
    jobs.extend(rl_jobs)
    parents.append(rl_parent)
    for ordinal, job in enumerate(jobs):
        job["ordinal"] = ordinal
    if len(jobs) != 26 or [job["ordinal"] for job in jobs] != list(range(26)):
        raise ValidationStop(f"expected exactly 26 ordered jobs, found {len(jobs)}")
    return jobs, parents


def audit_results(results: Path, records: list[dict], chat_template: Path) -> tuple[dict, dict]:
    predictions = results / "predictions.jsonl"
    metrics_path = results / "metrics.json"
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if len(rows) != 385 or len(records) != 385:
        raise ValidationStop(f"prediction count mismatch: {len(rows)}")
    accuracy_values, format_values, lengths = [], [], []
    for index, (row, record) in enumerate(zip(rows, records)):
        if row.get("index") != index:
            raise ValidationStop(f"prediction index mismatch at {index}")
        if any(row.get(key) != record.get(key) for key in ("image", "problem", "solution")):
            raise ValidationStop(f"prediction source mismatch at {index}")
        completion = str(row.get("completion", ""))
        if not completion:
            raise ValidationStop(f"empty completion at {index}")
        length = row.get("generated_token_count")
        if not isinstance(length, int) or length < 1 or length > MAX_NEW_TOKENS:
            raise ValidationStop(f"invalid generated token count at {index}: {length}")
        if row.get("reached_generation_cap") is not (length >= MAX_NEW_TOKENS):
            raise ValidationStop(f"generation-cap flag mismatch at {index}")
        accuracy = accuracy_reward(wrapped(completion), [record["solution"]])[0]
        form = format_reward(wrapped(completion))[0]
        if (row.get("accuracy_reward") != accuracy or row.get("format_reward") != form
                or row.get("predicted_choice") != choice_letter(completion)
                or row.get("target_choice") != choice_letter(record["solution"])):
            raise ValidationStop(f"offline parser mismatch at {index}")
        accuracy_values.append(accuracy)
        format_values.append(form)
        lengths.append(length)
    cap_hits = sum(length >= MAX_NEW_TOKENS for length in lengths)
    expected_metrics = {
        "count": 385,
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "test_accessed": False,
        "chat_template_file": str(chat_template.resolve()),
        "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "generation_cap_hit_count": cap_hits,
        "empty_completion_count": 0,
    }
    if any(metrics.get(key) != value for key, value in expected_metrics.items()):
        raise ValidationStop("metrics identity or generation gate mismatch")
    if (abs(metrics.get("mean_accuracy_reward") - sum(accuracy_values) / 385) > 1e-12
            or abs(metrics.get("mean_format_reward") - sum(format_values) / 385) > 1e-12):
        raise ValidationStop("metrics/offline rescore mismatch")
    if cap_hits:
        raise ValidationStop(f"1024-token truncation gate failed: {cap_hits} cap hits")
    audit = {"prediction_count": 385, "indices_exact": True, "source_fields_exact": True,
             "parser_consistency": True, "accuracy_correct": int(sum(accuracy_values)),
             "format_correct": int(sum(format_values)),
             "choice_extracted": sum(row["predicted_choice"] is not None for row in rows),
             "maximum_generated_tokens": max(lengths), "generation_cap_hit_count": cap_hits,
             "predictions_sha256": sha256(predictions), "metrics_sha256": sha256(metrics_path)}
    return metrics, audit


def run_job(job: dict, gpu: int, repo: Path, python: Path, data: Path, records: list[dict],
            chat_template: Path, jobs_root: Path, stop: threading.Event) -> dict:
    if stop.is_set():
        return {**job, "status": "not_started_after_peer_failure", "formal_result": False}
    job_dir = jobs_root / f"{job['ordinal']:02d}_{job['label']}"
    job_dir.mkdir()
    results = job_dir / "results"
    command = [str(python), str(repo / "scripts/infer_and_score_pathmmu_with_tokens.py"),
               "--model", job["checkpoint"], "--data", str(data), "--output-dir", str(results),
               "--max-new-tokens", str(MAX_NEW_TOKENS),
               "--chat-template-file", str(chat_template)]
    (job_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    payload = {**job, "status": "running", "formal_result": False, "gpu": gpu,
               "created_at": now_iso(), "generation": {"do_sample": False,
               "max_new_tokens": MAX_NEW_TOKENS, "chat_template_sha256": CHAT_TEMPLATE_SHA256},
               "test_accessed": False}
    manifest_path = job_dir / "run_manifest.yaml"
    write_yaml(manifest_path, payload)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "WANDB_MODE": "disabled",
                "PYTHONPATH": str(repo / "scripts")})
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        env.pop(key, None)
    try:
        with (job_dir / "inference.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=repo, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, text=True, check=False)
        if result.returncode:
            raise ValidationStop(f"inference exited with {result.returncode}")
        metrics, audit = audit_results(results, records, chat_template)
        gates = {"inference_completed": True, "exact_validation_count": True,
                 "deterministic_decoding": True, "raw_predictions_saved": True,
                 "offline_parser_consistency": True, "checkpoint_integrity_verified": True,
                 "no_generation_cap_hit": audit["generation_cap_hit_count"] == 0,
                 "test_not_accessed": True}
        payload.update({"status": "completed" if all(gates.values()) else "failed_gate",
                        "formal_result": all(gates.values()), "metrics": metrics,
                        "offline_audit": audit, "gates": gates,
                        "outputs": {"predictions": str(results / "predictions.jsonl"),
                                    "metrics": str(results / "metrics.json")},
                        "completed_at": now_iso()})
        write_yaml(manifest_path, payload)
        if not all(gates.values()):
            raise ValidationStop(f"job gates failed: {gates}")
        return payload
    except Exception as exc:
        stop.set()
        payload.update({"status": "failed", "formal_result": False,
                        "failure": {"type": type(exc).__name__, "message": str(exc)},
                        "completed_at": now_iso()})
        write_yaml(manifest_path, payload)
        return payload


def worker(slot: int, gpu: int, jobs: list[dict], repo: Path, python: Path, data: Path,
           records: list[dict], chat_template: Path, jobs_root: Path,
           stop: threading.Event) -> list[dict]:
    time.sleep(slot * 10)
    completed = []
    for job in jobs:
        result = run_job(job, gpu, repo, python, data, records, chat_template, jobs_root, stop)
        completed.append(result)
        if result.get("formal_result") is not True:
            break
    return completed


def select(rows: list[dict], family: str, count: int, expected: int) -> dict:
    candidates = [row for row in rows if row.get("family") == family
                  and row.get("sample_count") == count]
    if len(candidates) != expected:
        raise ValidationStop(f"selection candidate mismatch for {family} n={count}")
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
    if gpus != FORMAL_GPUS:
        raise ValidationStop("formal validation is frozen to physical GPUs 0-7")
    status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                            text=True, capture_output=True, check=False)
    if status.returncode or status.stdout.strip():
        raise ValidationStop(f"repository is not clean: {status.stdout.strip()}")
    legacy.verify_code_manifest(repo, CODE_MANIFEST, python)
    preflight = json.loads((install / "reports/preflight_report.json").read_text(encoding="utf-8"))
    if (preflight.get("passed") is not True
            or preflight.get("data", {}).get("data_version") != "pathmmu_image_disjoint_v2"
            or preflight.get("gates", {}).get("no_exact_content_overlap") is not True):
        raise ValidationStop("formal PathMMU v2 preflight is not passing")
    reward_test = subprocess.run([str(python), str(repo / "scripts/test_pathmmu_rewards.py")],
                                 text=True, capture_output=True, check=False)
    if reward_test.returncode:
        raise ValidationStop(f"parser regression tests failed: {reward_test.stdout}{reward_test.stderr}")
    data = install / "data/pathmmu_image_disjoint_v2/rewritten_records/validation_0385.json"
    if sha256(data) != VALIDATION_SHA256:
        raise ValidationStop("validation data hash mismatch")
    records = json.loads(data.read_text(encoding="utf-8"))
    if len(records) != 385 or any(not Path(row["image"]).is_file() for row in records):
        raise ValidationStop("validation count or image paths failed")
    chat_template = install / "models" / f"Qwen2.5-VL-7B-Instruct-{BASE_REVISION}" / "chat_template.json"
    if sha256(chat_template) != CHAT_TEMPLATE_SHA256:
        raise ValidationStop("chat template hash mismatch")
    jobs, parents = build_jobs(repo, install)
    hardware = legacy.gpu_inventory(gpus)
    if args.preflight_only:
        print(json.dumps({"passed": True, "job_count": len(jobs), "parents": parents,
                          "maximum_new_tokens": MAX_NEW_TOKENS, "hardware": hardware,
                          "test_accessed": False}, ensure_ascii=False, indent=2))
        return

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = install / "runs/validation_1024_all_candidates" / f"validation_1024_{timestamp}"
    run_dir.mkdir(parents=True)
    jobs_root = run_dir / "jobs"
    jobs_root.mkdir()
    manifest_path = run_dir / "run_manifest.yaml"
    manifest = {"schema_version": 1, "run_id": run_dir.name, "status": "running",
                "stage": "formal_validation_1024_all_candidates", "formal_result": False,
                "created_at": now_iso(), "scope": {"job_count": 26,
                "base": True, "sft_sample_counts": [500, 1000, 2000, 3000],
                "sft_curve_epochs": {"2000": list(range(1, 11)), "3000": list(range(1, 11))},
                "outcome_grpo_epochs": [1, 2, 3]}, "parents": parents,
                "data": {"version": "pathmmu_image_disjoint_v2", "split": "validation_0385",
                         "count": 385, "path": str(data), "sha256": VALIDATION_SHA256},
                "generation": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS,
                               "chat_template_file": str(chat_template),
                               "chat_template_sha256": CHAT_TEMPLATE_SHA256,
                               "cap_hit_policy": "fail entire run"},
                "selection_rule": "maximum validation accuracy; tie maximum format; tie earliest epoch",
                "hardware": hardware, "jobs": jobs,
                "provenance": {"repo_commit": subprocess.run(
                    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
                    capture_output=True, check=True).stdout.strip(),
                    "code_manifest": str(repo / "protocol" / CODE_MANIFEST),
                    "inference_sha256": sha256(repo / "scripts/infer_and_score_pathmmu_with_tokens.py"),
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
            futures = [executor.submit(worker, slot, gpu, assignments[slot], repo, python, data,
                                       records, chat_template, jobs_root, stop)
                       for slot, gpu in enumerate(gpus)]
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        results.sort(key=lambda row: row["ordinal"])
        completed = [row for row in results if row.get("formal_result") is True]
        gates = {"exact_26_jobs_completed": len(completed) == 26,
                 "all_job_gates_pass": len(completed) == 26,
                 "all_raw_predictions_saved": len(completed) == 26 and all(
                     Path(row["outputs"]["predictions"]).is_file() for row in completed),
                 "all_offline_parser_checks_pass": len(completed) == 26 and all(
                     row["gates"]["offline_parser_consistency"] for row in completed),
                 "no_generation_cap_hits": len(completed) == 26 and all(
                     row["offline_audit"]["generation_cap_hit_count"] == 0 for row in completed),
                 "test_not_accessed": True}
        if len(completed) == 26:
            selections = {}
            for family, count, expected in (("sft", 500, 1), ("sft", 1000, 1),
                                             ("sft", 2000, 10), ("sft", 3000, 10),
                                             ("outcome_grpo", 1000, 3)):
                selected = select(completed, family, count, expected)
                selections[f"{family}_n{count:04d}"] = {
                    "label": selected["label"], "epoch": selected["epoch"],
                    "global_step": selected["global_step"], "checkpoint": selected["checkpoint"],
                    "metrics": selected["metrics"], "offline_audit": selected["offline_audit"]}
        else:
            selections = {}
        manifest.update({"results": results, "selections": selections, "gates": gates,
                         "outputs": {"run_dir": str(run_dir), "jobs_dir": str(jobs_root)},
                         "status": "completed" if all(gates.values()) else "failed_gate",
                         "formal_result": all(gates.values()), "completed_at": now_iso()})
        write_yaml(manifest_path, manifest)
        if not all(gates.values()):
            raise ValidationStop(f"aggregate validation gates failed: {gates}")
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
