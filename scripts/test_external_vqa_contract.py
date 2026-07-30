#!/usr/bin/env python3

import unittest

from external_vqa_contract import (
    normalize_short_answer,
    omnimed_prompt,
    omnimed_score,
    pathvqa_score,
)


OMNI = {
    "question": "What is shown?",
    "gt_answer": "Optical Coherence Tomography (OCT)",
    "option_A": "X-ray",
    "option_B": "Optical Coherence Tomography (OCT)",
    "option_C": "CT",
    "option_D": "MRI",
}


class ExternalVQAContractTests(unittest.TestCase):
    def test_normalize_is_conservative(self):
        self.assertEqual(normalize_short_answer('  "Positively   Charged." '), "positively charged")
        self.assertNotEqual(normalize_short_answer("the liver"), normalize_short_answer("liver"))

    def test_pathvqa_exact(self):
        self.assertTrue(pathvqa_score(" Yes. ", "yes")["exact_match"])
        self.assertFalse(pathvqa_score("The answer is yes.", "yes")["exact_match"])

    def test_omni_official_mapping(self):
        score = omnimed_score("Optical Coherence Tomography", OMNI)
        self.assertEqual(score["official_predicted_choice"], "B")
        self.assertTrue(score["official_most_similar_correct"])
        self.assertFalse(score["strict_text_correct"])

    def test_omni_prompt_uses_official_semantics(self):
        prompt = omnimed_prompt(OMNI)
        self.assertIn("Here are 4 candidate answers:", prompt)
        self.assertIn("Only return what you think is the correct answer", prompt)

    def test_empty_omni_completion_is_incorrect(self):
        score = omnimed_score("", OMNI)
        self.assertIsNone(score["official_predicted_choice"])
        self.assertFalse(score["official_most_similar_correct"])

    def test_two_option_omni_record(self):
        record = {
            "question": "Is the finding present?",
            "gt_answer": "Yes.",
            "option_A": "No",
            "option_B": "Yes.",
            "option_C": None,
            "option_D": None,
        }
        score = omnimed_score("Yes.", record)
        self.assertEqual(score["official_predicted_choice"], "B")
        self.assertTrue(score["official_most_similar_correct"])


if __name__ == "__main__":
    unittest.main()
