import unittest

from run_mmmu_retention_diagnostic import extract_choice


class ExtractChoiceTests(unittest.TestCase):
    def test_tagged_choice(self):
        self.assertEqual(extract_choice("<think>x</think><answer>E</answer>", "ABCDE"), "E")

    def test_rejects_invalid_choice(self):
        self.assertIsNone(extract_choice("<answer>E</answer>", "ABCD"))

    def test_explicit_trailing_answer(self):
        self.assertEqual(extract_choice("reason\nAnswer: C", "ABCD"), "C")

    def test_boxed_choice(self):
        self.assertEqual(extract_choice(r"The closest option is \(\boxed{C}\)", "ABCD"), "C")

    def test_incomplete_closing_tag_at_generation_cap(self):
        self.assertEqual(extract_choice("reason<answer>D</answer", "ABCD"), "D")


if __name__ == "__main__":
    unittest.main()
