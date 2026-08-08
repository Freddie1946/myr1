#!/usr/bin/env python3
"""CUDA-free regression tests for ratio data preparation and sequencing."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PREP = load_module("ratio_prep", "scripts/prepare_data_ratio_ablation.py")
SEQ = load_module("ratio_sequence", "scripts/run_data_ratio_ablation_sequence.py")


class DataPartitionTests(unittest.TestCase):
    def synthetic_rows(self, prefix: str):
        rows = []
        for index in range(500):
            letter = "ABCD"[index % 4]
            rows.append({
                "image": f"/{prefix}_{index:04d}.png",
                "problem": f"question {index}",
                "solution": f"<think>x</think><answer>{letter}) answer</answer>",
            })
        return rows

    def test_partition_is_exact_exhaustive_image_disjoint_and_balanced(self):
        rows = self.synthetic_rows("sft")
        bins = PREP.partition_source(rows, "sft", 42)
        self.assertEqual([len(rows) for rows in bins], [125, 125, 125, 125])
        keys = [{PREP.canonical_record_key(row) for row in rows} for rows in bins]
        self.assertEqual(len(set().union(*keys)), 500)
        for left in range(4):
            for right in range(left + 1, 4):
                self.assertFalse(keys[left].intersection(keys[right]))
        for rows in bins:
            counts = PREP.stats(rows)["answer_counts"]
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 2)

    def test_task_order_and_rule_only_contract(self):
        manifest = {
            "arms": {
                arm: {
                    "sft": {"dataset_name": f"{arm}_sft", "qa_count": sft},
                    "rl": {"dataset_yaml": f"{arm}_rl.yaml", "qa_count": rl},
                }
                for arm, sft, rl in (
                    ("sft0750_rl0250", 750, 250),
                    ("sft0500_rl0500", 500, 500),
                    ("sft0250_rl0750", 250, 750),
                )
            }
        }
        tasks = SEQ.build_tasks("formal", manifest)
        self.assertEqual(
            [task.task_id for task in tasks],
            [
                "sft0750_rl0250_sft", "sft0750_rl0250_rule_rl",
                "sft0500_rl0500_sft", "sft0500_rl0500_rule_rl",
                "sft0250_rl0750_sft", "sft0250_rl0750_rule_rl",
                "stage2_continue_rule_rl1000", "base_rule_rl4000_gate50",
                "base_rule_rl4000",
            ],
        )
        self.assertEqual(tasks[-1].max_steps, 6000)
        self.assertEqual(tasks[-2].max_steps, 50)
        self.assertEqual(tasks[-3].max_steps, 1500)


class StateMachineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.state = Path(self.temporary.name) / "state.json"

    def tearDown(self):
        self.temporary.cleanup()

    def test_multiple_tasks_run_in_order_and_completed_tasks_skip(self):
        calls = []

        def execute(task_id, attempt):
            calls.append((task_id, attempt))
            return {"artifact": task_id}

        def validate(task_id, result):
            self.assertEqual(result["artifact"], task_id)

        state = SEQ.run_task_plan_for_test(["a", "b", "c"], self.state, execute, validate)
        self.assertEqual(calls, [("a", 1), ("b", 1), ("c", 1)])
        self.assertEqual(state["status"], "completed")
        SEQ.run_task_plan_for_test(["a", "b", "c"], self.state, execute, validate)
        self.assertEqual(calls, [("a", 1), ("b", 1), ("c", 1)])

    def test_failure_stops_then_next_invocation_retries_only_failed_tail(self):
        calls = []
        fail_once = {"b": True}

        def execute(task_id, attempt):
            calls.append((task_id, attempt))
            if task_id == "b" and fail_once.pop("b", False):
                raise RuntimeError("synthetic failure")
            return {"artifact": task_id}

        def validate(task_id, result):
            if result["artifact"] != task_id:
                raise AssertionError("bad artifact")

        with self.assertRaises(RuntimeError):
            SEQ.run_task_plan_for_test(["a", "b", "c"], self.state, execute, validate)
        first = json.loads(self.state.read_text())
        self.assertEqual(first["tasks"]["a"]["status"], "completed")
        self.assertEqual(first["tasks"]["b"]["status"], "failed")
        self.assertNotIn("c", first["tasks"])
        final = SEQ.run_task_plan_for_test(["a", "b", "c"], self.state, execute, validate)
        self.assertEqual(calls, [("a", 1), ("b", 1), ("b", 2), ("c", 1)])
        self.assertEqual(final["status"], "completed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
