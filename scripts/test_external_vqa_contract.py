#!/usr/bin/env python3

import unittest

from external_vqa_contract import (
    extract_omnimed_answer,
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
        exact = pathvqa_score(" Yes. ", "yes")
        self.assertTrue(exact["exact_match"])
        sentence = pathvqa_score("The answer is yes.", "yes")
        self.assertFalse(sentence["exact_match"])
        self.assertTrue(sentence["contract_aligned_exact_match"])
        self.assertEqual(sentence["official_token_overlap_score"], 0.25)

    def test_pathvqa_official_free_form_scores_full_sentence(self):
        score = pathvqa_score(
            "Each histone subunit is positively charged.", "positively charged"
        )
        self.assertFalse(score["strict_exact_match"])
        self.assertAlmostEqual(score["official_token_overlap_score"], 2 / 6)
        self.assertAlmostEqual(score["official_token_f1_score"], 0.5)
        self.assertAlmostEqual(score["sentence_bleu_1"], 2 / 6)
        self.assertAlmostEqual(score["sentence_bleu_2"], (2 / 6 * 1 / 5) ** 0.5)
        self.assertEqual(score["sentence_bleu_3"], 0.0)

    def test_omni_official_mapping(self):
        score = omnimed_score("Optical Coherence Tomography", OMNI)
        self.assertEqual(score["official_predicted_choice"], "B")
        self.assertTrue(score["official_most_similar_correct"])
        self.assertFalse(score["strict_text_correct"])
        self.assertTrue(score["contract_aligned_correct"])

    def test_omni_answer_tag_restores_official_input_assumption(self):
        completion = (
            "<think>CT is the appropriate modality; MRI is a distractor.</think>"
            "<answer>B) Optical Coherence Tomography (OCT)</answer>"
        )
        score = omnimed_score(completion, OMNI)
        self.assertEqual(score["contract_aligned_answer_source"], "answer_tag")
        self.assertEqual(score["contract_aligned_predicted_choice"], "B")
        self.assertTrue(score["contract_aligned_correct"])

    def test_incomplete_answer_tag_is_extractable(self):
        answer, source = extract_omnimed_answer("<think>x</think><answer>C) CT</")
        self.assertEqual((answer, source), ("C) CT", "answer_tag"))

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
