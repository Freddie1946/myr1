#!/usr/bin/env python3
from __future__ import annotations

import unittest

from prepare_visual_fidelity_panel import record_sha256, select_panel, target_choice


def record(choice: str, index: int) -> dict[str, str]:
    return {
        "image": f"/image/{index}.png",
        "problem": f"question {index}",
        "solution": f"<think>reason</think><answer>{choice}) answer</answer>",
    }


class VisualFidelityPanelTests(unittest.TestCase):
    def test_target_choice_uses_last_answer_tag(self) -> None:
        self.assertEqual(target_choice("<answer>A</answer> x <answer>D) y</answer>"), "D")

    def test_selection_is_balanced_deterministic_and_outcome_blind(self) -> None:
        records = [record(choice, index) for index, choice in enumerate("ABCD" * 8)]
        first = select_panel(records, per_choice=3, salt="fixed")
        second = select_panel(records, per_choice=3, salt="fixed")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual(
            {choice: sum(row["target_choice"] == choice for row in first) for choice in "ABCD"},
            {choice: 3 for choice in "ABCD"},
        )
        for row in first:
            self.assertEqual(row["source_record_sha256"], record_sha256(records[row["index"]]))
            self.assertNotIn("prediction", row)
        self.assertEqual(len({row["image"] for row in first}), len(first))

    def test_duplicate_images_are_never_selected_twice(self) -> None:
        records = [record(choice, index) for index, choice in enumerate("ABCD" * 8)]
        records[1]["image"] = records[0]["image"]
        selected = select_panel(records, per_choice=3, salt="fixed")
        self.assertEqual(len({row["image"] for row in selected}), 12)

    def test_schema_and_capacity_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unexpected validation schema"):
            select_panel([{"image": "x", "problem": "q", "solution": "<answer>A</answer>", "extra": 1}], per_choice=1, salt="x")
        with self.assertRaisesRegex(ValueError, "cannot fill"):
            select_panel([record("A", 0)], per_choice=1, salt="x")


if __name__ == "__main__":
    unittest.main()
