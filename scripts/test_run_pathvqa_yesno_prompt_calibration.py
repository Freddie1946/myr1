#!/usr/bin/env python3

import unittest

from run_pathvqa_yesno_prompt_calibration import (
    extract_calibration_answer,
    strict_pathmmu_choice_format,
)


class RunPathvqaYesNoPromptCalibrationTests(unittest.TestCase):
    def test_yes_no_contracts_are_conservative(self):
        self.assertEqual(
            extract_calibration_answer("<answer>B) No</answer>", "xml_v1_192")[0],
            "no",
        )
        self.assertIsNone(
            extract_calibration_answer("It may be Yes or No", "short_v1_192")[0]
        )
        self.assertEqual(
            extract_calibration_answer(
                "<think>Image-grounded evidence.</think><answer>Yes</answer>",
                "domain_think_answer_v2_2048",
            )[0],
            "yes",
        )

    def test_choice_contract_maps_ab(self):
        self.assertEqual(extract_calibration_answer("A", "choice_v1_32")[0], "yes")
        self.assertEqual(extract_calibration_answer("B) No", "choice_v1_32")[0], "no")
        self.assertEqual(
            extract_calibration_answer("<answer>B) No</answer>", "choice_v1_192")[0],
            "no",
        )
        self.assertIsNone(extract_calibration_answer("C", "choice_v1_32")[0])

    def test_structured_choice_contract_maps_position_to_semantics(self):
        self.assertEqual(
            extract_calibration_answer(
                "<think>brief</think><answer>B) No</answer>",
                "pathmmu_ab_v1_2048",
            )[0],
            "no",
        )
        self.assertTrue(
            strict_pathmmu_choice_format(
                "<think>brief</think><answer>A) Yes</answer>",
                "pathmmu_ab_v1_2048",
            )
        )
        self.assertFalse(
            strict_pathmmu_choice_format(
                "<think>brief</think><answer>A</answer>",
                "pathmmu_ab_v1_2048",
            )
        )


if __name__ == "__main__":
    unittest.main()
