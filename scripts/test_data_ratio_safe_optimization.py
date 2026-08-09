#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from backup_data_ratio_models_to_hf import selected_files
from run_data_ratio_ablation_optimized_tail import OptimizedTailRunner
from run_data_ratio_ablation_sequence import Task
from run_data_ratio_rule_rl_throughput_smoke import candidate_command


REPO = Path(__file__).resolve().parents[1]


class SafeOptimizationTests(unittest.TestCase):
    def test_gpu_zero3_has_no_offload(self):
        payload = json.loads(
            (REPO / "configs/deepspeed/ds_z3_gpu_torch_adamw.json").read_text()
        )
        zero = payload["zero_optimization"]
        self.assertEqual(zero["stage"], 3)
        self.assertNotIn("offload_optimizer", zero)
        self.assertNotIn("offload_param", zero)

    def test_smoke_keeps_scientific_batch_contract(self):
        command = candidate_command(
            repo=REPO,
            install=Path("/install"),
            output=Path("/output"),
            deepspeed=REPO / "configs/deepspeed/ds_z3_gpu_torch_adamw.json",
            gradient_checkpointing=False,
            max_steps=20,
        )
        joined = " ".join(command)
        for fragment in (
            "--num_generations 4",
            "--per_device_train_batch_size 1",
            "--gradient_accumulation_steps 1",
            "--reward_funcs accuracy format",
            "--max_steps 20",
            "--save_strategy no",
        ):
            self.assertIn(fragment, joined)

    def test_tail_checkpoint_policy_and_stage3_match(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = OptimizedTailRunner.__new__(OptimizedTailRunner)
            runner.repo = REPO
            runner.performance_report_path = Path(directory) / "performance.json"
            runner.performance_report_path.write_text("{}\n")
            runner.performance_report = {
                "selected": {
                    "name": "baseline_system_fallback",
                    "deepspeed": str(
                        REPO / "configs/deepspeed/ds_z3_optimizer_offload_torch_adamw.json"
                    ),
                    "deepspeed_sha256": __import__(
                        "run_data_ratio_ablation_sequence"
                    ).sha256_file(
                        REPO / "configs/deepspeed/ds_z3_optimizer_offload_torch_adamw.json"
                    ),
                    "gradient_checkpointing": True,
                    "attn_implementation": "sdpa",
                    "fallback": True,
                }
            }
            stage2 = runner.system_contract(
                Task("stage2_continue_rule_rl1000", "rule_rl", max_steps=1500)
            )
            gate = runner.system_contract(
                Task("base_rule_rl4000_gate50", "rule_rl", max_steps=50, gate_only=True)
            )
            full = runner.system_contract(
                Task("base_rule_rl4000", "rule_rl", max_steps=6000)
            )
            self.assertTrue(stage2["gradient_checkpointing"])
            self.assertEqual(stage2["attn_implementation"], "sdpa")
            self.assertEqual(stage2["save_steps"], 250)
            self.assertEqual(gate["save_steps"], 50)
            self.assertEqual(full["save_steps"], 500)

    def test_model_backup_excludes_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "config.json", "model.safetensors.index.json", "preprocessor_config.json",
                "tokenizer_config.json", "model-00001-of-00001.safetensors",
                "optimizer.pt", "scheduler.pt",
            ):
                (root / name).write_bytes(b"x")
            names = {path.name for path in selected_files(root)}
            self.assertIn("model-00001-of-00001.safetensors", names)
            self.assertNotIn("optimizer.pt", names)
            self.assertNotIn("scheduler.pt", names)


if __name__ == "__main__":
    unittest.main()
