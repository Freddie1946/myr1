#!/usr/bin/env python3
"""Regression tests for the exact n=3000 to n=2000 SFT handoff."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "n3000_to_n2000_handoff",
    REPO / "formal_machine" / "run_n3000_to_n2000_handoff.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class HandoffGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.install = self.root / "install"
        self.manifest = self.make_run(3000, MODULE.PARENT_RUN_ID)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_run(self, count: int, run_id: str) -> Path:
        manifest = (
            self.install / "runs" / "stage1_sft" / f"n{count:04d}_seed0042"
            / run_id / "run_manifest.yaml"
        )
        output = manifest.parent / "output"
        output.mkdir(parents=True)
        for name in ("model.safetensors.index.json", "trainer_state.json", "config.json"):
            (output / name).write_text("{}\n", encoding="utf-8")
        stride = (count + 7) // 8
        retention = []
        for epoch in range(1, 11):
            step = stride * epoch
            snapshot = manifest.parent / "epoch_snapshots" / f"checkpoint-{step}"
            snapshot.mkdir(parents=True)
            (snapshot / "snapshot_manifest.json").write_text("{}\n", encoding="utf-8")
            retention.append({"global_step": step, "epoch": float(epoch)})
        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "completed",
            "stage": "stage1_sft",
            "formal_result": True,
            "test_accessed": False,
            "data": {
                "version": "pathmmu_image_disjoint_v1",
                "dataset": f"pathvlm_sft_n{count:04d}",
                "qa_count": count,
                "image_reference_count": count,
                "all_image_paths_exist": True,
                "adapter": str(self.install / f"pathvlm_sft_n{count:04d}.json"),
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
                "backend": "deepspeed_zero2_gpu_fused_adamw_gc",
                "checkpoint_policy": {
                    "save_strategy": "epoch",
                    "model_only_epoch_snapshots": 10,
                    "full_resume_checkpoints_retained": 2,
                    "final_model_retained": True,
                    "automatic_scientific_snapshot_pruning": False,
                },
            },
            "hardware": {"cuda_visible_devices": list(range(8)), "gpu_count": 8},
            "provenance": {
                "source_config_sha256": (
                    "d15428f642fdc522254e7b3322c977d5bb985e6516df0350c816d988638ab273"
                    if count == 3000 else MODULE.TARGET_CONFIG_SHA256
                )
            },
            "gates": {key: True for key in MODULE.SFT_GATES},
            "retention": {"snapshots": retention},
            "outputs": {
                "final_checkpoint": str(output),
                "run_dir": str(manifest.parent),
                "epoch_snapshots": str(manifest.parent / "epoch_snapshots"),
            },
        }
        write_yaml(manifest, payload)
        return manifest

    def rewrite(self, mutation) -> None:
        payload = yaml.safe_load(self.manifest.read_text(encoding="utf-8"))
        mutation(payload)
        write_yaml(self.manifest, payload)

    def test_exact_parent_passes(self) -> None:
        payload = MODULE.validate_completed_sft_manifest(
            self.manifest, self.install, 3000, MODULE.PARENT_RUN_ID
        )
        self.assertTrue(payload["formal_result"])

    def test_fresh_target_passes(self) -> None:
        target = self.make_run(2000, "formal_sft_n2000_seed0042_20990101_010000")
        payload = MODULE.validate_completed_sft_manifest(target, self.install, 2000)
        self.assertEqual(payload["data"]["qa_count"], 2000)

    def test_false_gate_stops(self) -> None:
        self.rewrite(lambda payload: payload["gates"].update({"language_tensor_changed": False}))
        with self.assertRaises(MODULE.HandoffStop):
            MODULE.validate_completed_sft_manifest(
                self.manifest, self.install, 3000, MODULE.PARENT_RUN_ID
            )

    def test_test_access_stops(self) -> None:
        self.rewrite(lambda payload: payload.update({"test_accessed": True}))
        with self.assertRaises(MODULE.HandoffStop):
            MODULE.validate_completed_sft_manifest(
                self.manifest, self.install, 3000, MODULE.PARENT_RUN_ID
            )

    def test_wrong_parent_run_id_stops(self) -> None:
        self.rewrite(lambda payload: payload.update({"run_id": "formal_sft_n3000_seed0042_wrong"}))
        with self.assertRaises(MODULE.HandoffStop):
            MODULE.validate_completed_sft_manifest(
                self.manifest, self.install, 3000, MODULE.PARENT_RUN_ID
            )

    def test_missing_snapshot_stops(self) -> None:
        snapshot = self.manifest.parent / "epoch_snapshots" / "checkpoint-375"
        (snapshot / "snapshot_manifest.json").unlink()
        with self.assertRaises(MODULE.HandoffStop):
            MODULE.validate_completed_sft_manifest(
                self.manifest, self.install, 3000, MODULE.PARENT_RUN_ID
            )

    def test_wrong_config_hash_stops(self) -> None:
        self.rewrite(
            lambda payload: payload["provenance"].update({"source_config_sha256": "0" * 64})
        )
        with self.assertRaises(MODULE.HandoffStop):
            MODULE.validate_completed_sft_manifest(
                self.manifest, self.install, 3000, MODULE.PARENT_RUN_ID
            )

    def test_failed_parent_status_stops(self) -> None:
        self.rewrite(lambda payload: payload.update({"status": "failed", "formal_result": False}))
        with self.assertRaises(MODULE.HandoffStop):
            MODULE.validate_completed_sft_manifest(
                self.manifest, self.install, 3000, MODULE.PARENT_RUN_ID
            )

    def test_manifest_path_escape_stops(self) -> None:
        outside = self.root / "outside" / "run_manifest.yaml"
        outside.parent.mkdir(parents=True)
        outside.write_text(self.manifest.read_text(encoding="utf-8"), encoding="utf-8")
        with self.assertRaises(MODULE.HandoffStop):
            MODULE.validate_completed_sft_manifest(
                outside, self.install, 3000, MODULE.PARENT_RUN_ID
            )

    def test_unique_new_manifest(self) -> None:
        old = (self.root / "old.yaml").resolve()
        new = (self.root / "new.yaml").resolve()
        self.assertEqual(MODULE.select_unique_new_manifest({old}, [old, new]), new)
        self.assertIsNone(MODULE.select_unique_new_manifest({old}, [old]))

    def test_multiple_new_manifests_stop(self) -> None:
        one = self.root / "one.yaml"
        two = self.root / "two.yaml"
        with self.assertRaises(MODULE.HandoffStop):
            MODULE.select_unique_new_manifest(set(), [one, two])

    def test_disk_below_projected_start_gate_stops(self) -> None:
        with self.assertRaises(MODULE.HandoffStop):
            MODULE.require_start_disk(MODULE.PROJECTED_START_FREE_BYTES - 1)
        MODULE.require_start_disk(MODULE.PROJECTED_START_FREE_BYTES)

    def test_busy_gpu_list_is_not_idle(self) -> None:
        self.assertFalse(MODULE.gpus_are_idle([{"pid": 123}]))
        self.assertTrue(MODULE.gpus_are_idle([]))

    def test_unresolved_launch_intent_forbids_retry(self) -> None:
        handoff = object.__new__(MODULE.Handoff)
        handoff.state = {"pending_launch": {"declared_at": "earlier"}}
        with self.assertRaises(MODULE.HandoffStop):
            handoff.launch_target()

    def test_lock_rejects_second_watcher(self) -> None:
        lock = self.root / "handoff.lock"
        first = MODULE.acquire_lock(lock)
        try:
            with self.assertRaises(MODULE.HandoffStop):
                MODULE.acquire_lock(lock)
        finally:
            first.close()

    def test_atomic_state_has_no_temporary_residue(self) -> None:
        state = self.root / "state.json"
        MODULE.atomic_write_json(state, {"status": "running"})
        self.assertFalse(state.with_name("state.json.tmp").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
