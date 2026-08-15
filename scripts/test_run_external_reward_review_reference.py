#!/usr/bin/env python3

import json
import unittest

from run_external_reward_review_reference import BOOLEAN_FIELDS, SCORE_FIELDS, parse_reference


class RewardReferenceTest(unittest.TestCase):
    def test_parse(self):
        value = {field: False for field in BOOLEAN_FIELDS}
        value.update({field: 3 for field in SCORE_FIELDS})
        value["overall_reason"] = "Concrete pathology rationale."
        self.assertEqual(parse_reference(json.dumps(value))["reviewer_confidence_1_to_5"], 3)

    def test_reject_bad_boolean(self):
        value = {field: False for field in BOOLEAN_FIELDS}
        value[BOOLEAN_FIELDS[0]] = "maybe"
        value.update({field: 3 for field in SCORE_FIELDS})
        value["overall_reason"] = "x"
        with self.assertRaises(ValueError):
            parse_reference(json.dumps(value))


if __name__ == "__main__":
    unittest.main()
