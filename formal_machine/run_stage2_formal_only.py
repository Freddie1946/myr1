#!/usr/bin/env python3
"""Formal-only continuation from the passing Stage-2 pilot evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from formal_machine import run_stage2_priority_pipeline as pipeline  # noqa: E402


INSTALL = Path("/home/wjy/pathvlm_r1_v1_formal")
SOURCE_RUN_ID = "stage2_priority_n1000_seed0042_20260720_035156"
SOURCE_RUN = INSTALL / "runs/stage2_outcome_grpo/priority_n1000_seed0042" / SOURCE_RUN_ID
PROTOCOL_NAME = "stage2_formal_n1000_seed42_continuation_manifest_20260720_064352.json"
CODE_MANIFEST_NAME = "stage2_formal_n1000_seed42_continuation_code_manifest_20260720_064352.json"
PIPELINE_MANIFEST_SHA256 = "5e1cf2bece85a73aa6af6e06c91e54ca211dc363376a337555bc722879a2b1be"
PILOT_MANIFEST_SHA256 = "520aae6474c7a71f7093b9070b09900ed8640f4e461242a60429c3b31666828d"
PRUNING_RECORD_SHA256 = "1507ea57cbad257ff6f8c4e784c26470457463a2c095b9b0a4d6a06e38a67423"
INVENTORY_SHA256 = "9483d4ce1ece4731e75a2c2e1a7375b27e5107f2914bc71900473c5ed1d6bebb"
REQUIRED_FREE_BYTES = 897060103530


class ContinuationStop(RuntimeError):
    """A formal-only fail-closed condition."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file():
        raise ContinuationStop(f"missing frozen evidence: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ContinuationStop(f"frozen evidence hash mismatch: {path}: {actual}")
    return {"path": str(path), "sha256": actual}


def verify_continuation_protocol() -> dict[str, Any]:
    path = REPO / "protocol" / PROTOCOL_NAME
    payload = pipeline.read_json(path)
    expected = {
        "status": "frozen_before_execution",
        "source_pipeline_run_id": SOURCE_RUN_ID,
        "source_pipeline_manifest_sha256": PIPELINE_MANIFEST_SHA256,
        "passing_pilot_manifest_sha256": PILOT_MANIFEST_SHA256,
        "pruning_record_sha256": PRUNING_RECORD_SHA256,
        "checkpoint_inventory_sha256": INVENTORY_SHA256,
        "pilot_rerun_authorized": False,
        "pilot_output_is_parent": False,
        "data_version": pipeline.DATA_VERSION,
        "rl_sample_count": 1000,
        "seed": 42,
        "formal_epochs": 3,
        "formal_steps": 1500,
        "required_free_bytes": REQUIRED_FREE_BYTES,
        "test_authorized": False,
        "test_accessed": False,
        "picked_json_used": False,
    }
    mismatch = {key: {"expected": value, "actual": payload.get(key)}
                for key, value in expected.items() if payload.get(key) != value}
    if mismatch:
        raise ContinuationStop(f"formal-only protocol mismatch: {mismatch}")
    parent = payload.get("formal_parent", {})
    if parent != {
        "sft_sample_count": 3000, "seed": 42, "epoch": 3, "global_step": 1125,
        "snapshot_manifest_sha256": pipeline.gate.SNAPSHOT_MANIFEST_SHA256,
    }:
        raise ContinuationStop(f"formal-only parent protocol mismatch: {parent}")
    return {"path": str(path), "sha256": sha256(path), "verified": expected}


def verify_new_code_manifest(python: Path) -> dict[str, Any]:
    path = REPO / "protocol" / CODE_MANIFEST_NAME
    result = pipeline.run_checked([
        str(python), str(REPO / "scripts/verify_code_hash_manifest.py"),
        "--repo-root", str(REPO), "--manifest", str(path),
    ])
    original = pipeline.verify_code_manifest(python)
    return {"path": str(path), "sha256": sha256(path), "verified": True,
            "stdout": result["stdout"], "original_priority_manifest": original}


def verify_passing_pilot_and_pruning() -> dict[str, Any]:
    pipeline_path = SOURCE_RUN / "pipeline_manifest.json"
    pilot_path = SOURCE_RUN / "pilot/pilot_manifest.json"
    pruning_path = SOURCE_RUN / "pilot/checkpoint25_pruning_record.json"
    inventory_path = SOURCE_RUN / "pilot/checkpoint25_pre_pruning_sha256.txt"
    for path, expected in (
        (pipeline_path, PIPELINE_MANIFEST_SHA256), (pilot_path, PILOT_MANIFEST_SHA256),
        (pruning_path, PRUNING_RECORD_SHA256), (inventory_path, INVENTORY_SHA256),
    ):
        require_hash(path, expected)
    outer = pipeline.read_json(pipeline_path)
    pilot = pipeline.read_json(pilot_path)
    pruning = pipeline.read_json(pruning_path)
    checkpoint = SOURCE_RUN / "pilot/output/checkpoint-25"
    formal_dir = SOURCE_RUN / "formal_n1000_seed0042_epoch3"
    if (
        outer.get("status") != "failed"
        or outer.get("formal_result") is not False
        or "storage gate failed" not in outer.get("failure", {}).get("message", "")
        or outer.get("pilot", {}).get("gate_passed") is not True
        or len(outer.get("storage_pruning", [])) != 1
    ):
        raise ContinuationStop("source pipeline disposition is not the frozen storage stop")
    if (
        pilot.get("status") != "completed"
        or pilot.get("formal_result") is not False
        or pilot.get("gate_passed") is not True
        or not pilot.get("gates")
        or not all(pilot["gates"].values())
        or pilot.get("validation", {}).get("count") != 385
        or pilot.get("validation", {}).get("correct") != 232
        or pilot.get("validation", {}).get("test_accessed") is not False
    ):
        raise ContinuationStop("passing pilot evidence is incomplete or changed")
    predictions = Path(pilot["validation"]["predictions"])
    metrics = Path(pilot["validation"]["metrics"])
    if (
        sha256(predictions) != pilot["validation"]["predictions_sha256"]
        or sha256(metrics) != pilot["validation"]["metrics_sha256"]
        or len(pipeline.read_jsonl(predictions)) != 385
    ):
        raise ContinuationStop("pilot validation raw evidence changed")
    if (
        pruning.get("deletion_completed") is not True
        or pruning.get("post_delete_verification_passed") is not True
        or pruning.get("formal_training_started") is not False
        or pruning.get("test_accessed") is not False
        or pruning.get("sha256_inventory_sha256") != INVENTORY_SHA256
        or checkpoint.exists()
        or formal_dir.exists()
    ):
        raise ContinuationStop("pilot pruning/formal-absence evidence mismatch")
    return {
        "source_run": str(SOURCE_RUN), "pipeline_manifest": str(pipeline_path),
        "pipeline_manifest_sha256": PIPELINE_MANIFEST_SHA256,
        "pilot_manifest": str(pilot_path), "pilot_manifest_sha256": PILOT_MANIFEST_SHA256,
        "pruning_record": str(pruning_path), "pruning_record_sha256": PRUNING_RECORD_SHA256,
        "inventory": str(inventory_path), "inventory_sha256": INVENTORY_SHA256,
        "pilot_gate_passed": True, "pilot_validation_correct": 232,
        "pilot_validation_count": 385, "checkpoint_pruned": True,
        "formal_directory_absent": True, "test_accessed": False,
    }


def verify_storage() -> dict[str, Any]:
    disk = shutil.disk_usage(INSTALL)
    evidence = {
        "free_bytes": disk.free, "required_free_bytes": REQUIRED_FREE_BYTES,
        "margin_bytes": disk.free - REQUIRED_FREE_BYTES,
        "disk_reserve_bytes": pipeline.DISK_RESERVE_BYTES,
        "passed": disk.free >= REQUIRED_FREE_BYTES,
    }
    if not evidence["passed"]:
        raise ContinuationStop(f"formal-only storage gate failed: {evidence}")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-hardware", action="store_true")
    args = parser.parse_args()
    python = Path(sys.executable).resolve()
    if python != (INSTALL / "envs/grpo/bin/python").resolve():
        raise ContinuationStop(f"must use pinned GRPO Python, got {python}")

    preflight: dict[str, Any] = {
        "created_at": pipeline.now_iso(), "git": pipeline.gate.verify_git_clean(REPO),
        "protocol": verify_continuation_protocol(), "code": verify_new_code_manifest(python),
        "pilot_and_pruning": verify_passing_pilot_and_pruning(),
    }
    parent, preflight["parent"] = pipeline.gate.verify_parent(INSTALL)
    preflight["frozen_data"] = pipeline.gate.verify_data(REPO, INSTALL)
    dataset_yaml, records, preflight["data"] = pipeline.verify_rl_data()
    preflight["validation"] = pipeline.verify_validation_inputs()
    preflight["parser_regression"] = pipeline.run_checked(
        [str(python), str(REPO / "scripts/test_pathmmu_rewards.py")]
    )
    preflight["wrapper_regression"] = pipeline.run_checked(
        [str(python), str(REPO / "scripts/test_grpo_pathmmu_audit.py")]
    )
    preflight["storage"] = verify_storage()
    preflight["system"] = pipeline.gate.system_inventory(INSTALL)
    if not args.skip_hardware:
        preflight["hardware"] = pipeline.gate.gpu_inventory(pipeline.GPU_IDS)
        pipeline.require_free_port(pipeline.MASTER_PORT)
    preflight["passed"] = True
    if args.preflight_only:
        print(json.dumps(preflight, indent=2, ensure_ascii=False))
        return
    if args.skip_hardware:
        raise ContinuationStop("--skip-hardware is valid only with --preflight-only")

    manifest_path = SOURCE_RUN / "formal_only_continuation_manifest.json"
    if manifest_path.exists():
        raise ContinuationStop(f"continuation manifest already exists: {manifest_path}")
    manifest: dict[str, Any] = {
        "schema_version": 1, "run_id": f"formal_only_{pipeline.timestamp()}",
        "stage": "stage2_outcome_grpo_formal_n1000_seed0042",
        "status": "running_formal_training", "formal_result": False,
        "created_at": pipeline.now_iso(), "preflight": preflight,
        "parent": preflight["parent"], "pilot_output_used_as_parent": False,
        "pilot_rerun": False, "test_accessed": False, "picked_json_used": False,
    }
    pipeline.atomic_json(manifest_path, manifest)
    start_commit = preflight["git"]["commit"]
    try:
        manifest["formal_training"] = pipeline.formal_training(
            SOURCE_RUN, parent, dataset_yaml, records, python
        )
        transition_git = pipeline.gate.verify_git_clean(REPO)
        if transition_git["commit"] != start_commit:
            raise ContinuationStop("Git commit changed between formal training and validation")
        manifest["pre_validation_code"] = verify_new_code_manifest(python)
        manifest["status"] = "running_formal_validation"
        pipeline.atomic_json(manifest_path, manifest)
        manifest["formal_validation"] = pipeline.formal_validation(
            SOURCE_RUN, manifest["formal_training"], python
        )
        if not all(manifest["formal_validation"].get("gates", {}).values()):
            raise ContinuationStop(
                f"formal validation gate failed: {manifest['formal_validation'].get('gates')}"
            )
        manifest.update({
            "status": "completed", "formal_result": True,
            "completed_at": pipeline.now_iso(),
            "selected_checkpoint": manifest["formal_validation"]["selected"]["model"],
            "test_accessed": False,
        })
    except Exception as exc:
        manifest.update({
            "status": "failed", "formal_result": False, "failed_at": pipeline.now_iso(),
            "failure": {"type": type(exc).__name__, "message": str(exc)},
            "test_accessed": False,
        })
        pipeline.atomic_json(manifest_path, manifest)
        raise
    pipeline.atomic_json(manifest_path, manifest)
    print(f"[stage2-formal-only] COMPLETE: {manifest_path}")


if __name__ == "__main__":
    main()
