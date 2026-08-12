import unittest

from merge_pathmmu_split_passk import summarize
from run_pathmmu_split_passk_shard import selected_records


class SplitPassKTests(unittest.TestCase):
    def test_panel_selection_and_split_labels(self):
        sft = [{"image": "a", "problem": "p", "solution": "<answer>A</answer>"}] * 3
        rl = [{"image": "b", "problem": "q", "solution": "<answer>B</answer>"}] * 2
        rows = selected_records(sft, rl, 1)
        self.assertEqual([row["split"] for row in rows], ["sft3000", "rl1000"])

    def test_summary_bins(self):
        def row(correct):
            return {
                "accuracy_rewards": [1.0] * correct + [0.0] * (8 - correct),
                "format_rewards": [1.0] * 8,
                "generation_cap_hits": [False] * 8,
            }
        metrics = summarize([row(0), row(4), row(8)])
        self.assertAlmostEqual(metrics["mean_rollout_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["mixed_fraction"], 1 / 3)
        self.assertEqual(metrics["correct_rollout_histogram"]["4"], 1)


if __name__ == "__main__":
    unittest.main()
