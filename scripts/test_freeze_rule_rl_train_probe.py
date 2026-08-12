import unittest

from freeze_rule_rl_train_probe import freeze


class FreezeRuleRlTrainProbeTests(unittest.TestCase):
    def test_balanced_and_image_unique(self):
        rows = []
        for letter in "ABCD":
            for index in range(4):
                rows.append({
                    "image": f"{letter}-{index}.png",
                    "problem": f"question {letter} {index}",
                    "solution": f"<answer>{letter}</answer>",
                })
        frozen = freeze(rows, 8)
        self.assertEqual(len(frozen), 8)
        self.assertEqual(len({row["image"] for row in frozen}), 8)
        self.assertEqual(
            {letter: sum(f"<answer>{letter}</answer>" == row["solution"] for row in frozen) for letter in "ABCD"},
            {letter: 2 for letter in "ABCD"},
        )

    def test_requires_multiple_of_four(self):
        with self.assertRaises(ValueError):
            freeze([], 6)


if __name__ == "__main__":
    unittest.main()
