#!/usr/bin/env python3

import unittest
from pathlib import Path

from backup_critical_models_to_hf import CRITICAL_SPECS, model_card, resolved_tasks


class CriticalBackupPolicyTests(unittest.TestCase):
    def test_allowlist_is_exact(self):
        self.assertEqual(
            [item["task_id"] for item in CRITICAL_SPECS],
            ["grok43_stage3", "stage2_continue_rule_rl1000", "base_rule_rl4000"],
        )
        self.assertFalse(any("sft0" in item["task_id"] for item in CRITICAL_SPECS))

    def test_visibility_policy(self):
        by_id = {item["task_id"]: item for item in CRITICAL_SPECS}
        self.assertTrue(by_id["grok43_stage3"]["private"])
        self.assertFalse(by_id["grok43_stage3"]["gated"])
        for task_id in ("stage2_continue_rule_rl1000", "base_rule_rl4000"):
            self.assertFalse(by_id[task_id]["private"])
            self.assertEqual(by_id[task_id]["gated"], "manual")

    def test_public_cards_declare_gate_and_base(self):
        for task in CRITICAL_SPECS[1:]:
            card = model_card(dict(task))
            self.assertIn("gated: true", card)
            self.assertIn("base_model: Qwen/Qwen2.5-VL-7B-Instruct", card)
            self.assertIn("manual approval", card)

    def test_paths_resolve_without_changing_allowlist(self):
        tasks = resolved_tasks(Path("/formal"), Path("/grok/output"))
        self.assertEqual(tasks[0]["source"], Path("/grok/output"))
        self.assertEqual(
            tasks[1]["source"],
            Path("/formal/tasks/stage2_continue_rule_rl1000/output"),
        )


if __name__ == "__main__":
    unittest.main()
