import unittest

from freeze_visual_adaptation_followup_data import answer_letter, select_records


class FollowupDataTests(unittest.TestCase):
    def test_balanced_and_image_unique(self):
        records = []
        for letter in "ABCD":
            for index in range(4):
                records.append({
                    "image": f"{letter}-{index}.png",
                    "problem": "q",
                    "solution": f"<think>x</think><answer>{letter}) answer</answer>",
                })
        selected = select_records(records, 8, 7)
        self.assertEqual(len({row["image"] for row in selected}), 8)
        self.assertEqual({letter: sum(answer_letter(row) == letter for row in selected) for letter in "ABCD"},
                         {letter: 2 for letter in "ABCD"})


if __name__ == "__main__":
    unittest.main()
