#!/usr/bin/env python3

import unittest

from run_pathvqa_yesno_prompt_calibration import extract_calibration_answer


class RunPathvqaYesNoPromptCalibrationTests(unittest.TestCase):
    def test_yes_no_contracts_are_conservative(self):
        self.assertEqual(
            extract_calibration_answer("<answer>B) No</answer>", "xml_v1_192")[0],
            "no",
        )
        self.assertIsNone(
            extract_calibration_answer("It may be Yes or No", "short_v1_192")[0]
        )

    def test_choice_contract_maps_ab(self):
        self.assertEqual(extract_calibration_answer("A", "choice_v1_32")[0], "yes")
        self.assertEqual(extract_calibration_answer("B) No", "choice_v1_32")[0], "no")
        self.assertEqual(
            extract_calibration_answer("<answer>B) No</answer>", "choice_v1_192")[0],
            "no",
        )
        self.assertIsNone(extract_calibration_answer("C", "choice_v1_32")[0])


if __name__ == "__main__":
    unittest.main()
