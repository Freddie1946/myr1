#!/usr/bin/env python3
"""Regression tests for the fail-closed formal SFT scale supervisor."""

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "formal_sft_scale_sequence",
    REPO / "formal_machine" / "run_formal_sft_scale_sequence.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class SupervisorGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.install = self.root / "install"
        self.sft_manifest = (
            self.install
            / "runs/stage1_sft/n0500_seed0042/formal_sft_n0500_seed0042_20990101_000000"
            / "run_manifest.yaml"
        )
        checkpoint = self.sft_manifest.parent / "output"
        checkpoint.mkdir(parents=True)
        for name in ("model.safetensors.index.json", "trainer_state.json", "config.json"):
            (checkpoint / name).write_text("{}\n", encoding="utf-8")
        self.sft_payload = {
            "schema_version": 1,
            "run_id": "formal_sft_n0500_seed0042_20990101_000000",
            "status": "completed",
            "stage": "stage1_sft",
            "formal_result": True,
            "test_accessed": False,
            "data": {
                "version": "pathmmu_image_disjoint_v1",
                "dataset": "pathvlm_sft_n0500",
                "qa_count": 500,
                "all_image_paths_exist": True,
                "adapter": str(self.install / "data/pathvlm_sft_n0500.json"),
            },
            "model": {
                "base_id": "Qwen/Qwen2.5-VL-7B-Instruct",
                "base_revision": MODULE.BASE_REVISION,
            },
            "training": {
                "seed": 42,
                "epochs": 10,
                "learning_rate": 2e-5,
                "finetuning_type": "full",
                "language_model_trainable": True,
                "vision_tower_frozen": True,
                "multimodal_projector_frozen": True,
            },
            "hardware": {"cuda_visible_devices": list(range(8)), "gpu_count": 8},
            "gates": {key: True for key in MODULE.SFT_GATES},
            "outputs": {"final_checkpoint": str(checkpoint), "run_dir": str(self.sft_manifest.parent)},
        }
        write_yaml(self.sft_manifest, self.sft_payload)
        self.checkpoint = checkpoint

        self.validation_manifest = (
            self.sft_manifest.parent
            / "validation/validation_0385_20990101_010000/run_manifest.yaml"
        )
        results = self.validation_manifest.parent / "results"
        results.mkdir(parents=True)
        metrics_path = results / "metrics.json"
        predictions_path = results / "predictions.jsonl"
        metrics = {
            "count": 385,
            "do_sample": False,
            "test_accessed": False,
            "mean_accuracy_reward": 0.5,
        }
        metrics_path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")
        predictions_path.write_text("{}\n" * 385, encoding="utf-8")
        self.validation_payload = {
            "schema_version": 1,
            "run_id": "validation_0385_20990101_010000",
            "status": "completed",
            "stage": "stage1_validation",
            "formal_result": True,
            "test_accessed": False,
            "parent": {
                "manifest": str(self.sft_manifest),
                "manifest_sha256": MODULE.sha256(self.sft_manifest),
                "checkpoint": str(checkpoint),
            },
            "data": {
                "version": "pathmmu_image_disjoint_v1",
                "split": "validation_0385",
                "count": 385,
                "path": str(self.install / "data/rewritten_records/validation_0385.json"),
            },
            "generation": {"do_sample": False, "max_new_tokens": 192},
            "metrics": metrics,
            "gates": {key: True for key in MODULE.VALIDATION_GATES},
            "outputs": {"metrics": str(metrics_path), "predictions": str(predictions_path)},
        }
        write_yaml(self.validation_manifest, self.validation_payload)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def rewrite_sft(self, mutate) -> None:
        payload = copy.deepcopy(self.sft_payload)
        mutate(payload)
        write_yaml(self.sft_manifest, payload)

    def rewrite_validation(self, mutate) -> None:
        payload = copy.deepcopy(self.validation_payload)
        mutate(payload)
        write_yaml(self.validation_manifest, payload)

    def test_valid_sft_and_validation_pass(self) -> None:
        payload, checkpoint = MODULE.validate_sft_manifest(
            self.sft_manifest, self.install, 500
        )
        self.assertTrue(payload["formal_result"])
        self.assertEqual(checkpoint, self.checkpoint.resolve())
        validation = MODULE.validate_validation_manifest(
            self.validation_manifest, self.install, self.sft_manifest, self.checkpoint
        )
        self.assertEqual(validation["metrics"]["count"], 385)

    def test_sft_false_gate_stops(self) -> None:
        self.rewrite_sft(lambda payload: payload["gates"].update({"language_tensor_changed": False}))
        with self.assertRaises(MODULE.SequenceStop):
            MODULE.validate_sft_manifest(self.sft_manifest, self.install, 500)

    def test_sft_test_access_stops(self) -> None:
        self.rewrite_sft(lambda payload: payload.update({"test_accessed": True}))
        with self.assertRaises(MODULE.SequenceStop):
            MODULE.validate_sft_manifest(self.sft_manifest, self.install, 500)

    def test_sft_wrong_gpu_topology_stops(self) -> None:
        self.rewrite_sft(
            lambda payload: payload["hardware"].update(
                {"cuda_visible_devices": [0, 1, 2, 3], "gpu_count": 4}
            )
        )
        with self.assertRaises(MODULE.SequenceStop):
            MODULE.validate_sft_manifest(self.sft_manifest, self.install, 500)

    def test_sft_checkpoint_outside_run_stops(self) -> None:
        outside = self.install / "outside"
        self.rewrite_sft(
            lambda payload: payload["outputs"].update({"final_checkpoint": str(outside)})
        )
        with self.assertRaises(MODULE.SequenceStop):
            MODULE.validate_sft_manifest(self.sft_manifest, self.install, 500)

    def test_validation_parent_hash_mismatch_stops(self) -> None:
        self.rewrite_validation(
            lambda payload: payload["parent"].update({"manifest_sha256": "0" * 64})
        )
        with self.assertRaises(MODULE.SequenceStop):
            MODULE.validate_validation_manifest(
                self.validation_manifest, self.install, self.sft_manifest, self.checkpoint
            )

    def test_validation_test_access_stops(self) -> None:
        self.rewrite_validation(lambda payload: payload["metrics"].update({"test_accessed": True}))
        with self.assertRaises(MODULE.SequenceStop):
            MODULE.validate_validation_manifest(
                self.validation_manifest, self.install, self.sft_manifest, self.checkpoint
            )

    def test_validation_wrong_prediction_count_stops(self) -> None:
        predictions = Path(self.validation_payload["outputs"]["predictions"])
        predictions.write_text("{}\n" * 384, encoding="utf-8")
        with self.assertRaises(MODULE.SequenceStop):
            MODULE.validate_validation_manifest(
                self.validation_manifest, self.install, self.sft_manifest, self.checkpoint
            )

    def test_lock_rejects_second_supervisor(self) -> None:
        lock = self.root / "supervisor.lock"
        first = MODULE.acquire_lock(lock)
        try:
            with self.assertRaises(MODULE.SequenceStop):
                MODULE.acquire_lock(lock)
        finally:
            first.close()

    def test_atomic_state_has_no_temporary_residue(self) -> None:
        state = self.root / "state.json"
        MODULE.atomic_write_json(state, {"status": "running"})
        self.assertEqual(json.loads(state.read_text(encoding="utf-8"))["status"], "running")
        self.assertFalse(state.with_name("state.json.tmp").exists())

    def test_unresolved_sft_launch_intent_forbids_retry(self) -> None:
        supervisor = object.__new__(MODULE.Supervisor)
        supervisor.state = {
            "runs": {},
            "pending_launch": {"label": "sft_n1000", "declared_at": "earlier"},
        }
        supervisor.install = self.install
        supervisor.discover_sft_manifests = lambda count: []
        with self.assertRaises(MODULE.SequenceStop):
            supervisor.ensure_sft(1000)

    def test_unresolved_validation_launch_intent_forbids_retry(self) -> None:
        supervisor = object.__new__(MODULE.Supervisor)
        supervisor.state = {
            "runs": {"500": {}},
            "pending_launch": {"label": "validation_n0500", "declared_at": "earlier"},
        }
        supervisor.install = self.install
        supervisor.discover_validation_manifests = lambda parent: []
        with self.assertRaises(MODULE.SequenceStop):
            supervisor.ensure_validation(500, self.sft_manifest, self.checkpoint)


if __name__ == "__main__":
    unittest.main(verbosity=2)
