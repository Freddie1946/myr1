#!/usr/bin/env python3
from __future__ import annotations

import json
import unittest

from pathvqa_answer_intent_verifier import parse_verdict


class PathvqaIntentVerifierTests(unittest.TestCase):
    def test_accept(self) -> None:
        value = parse_verdict(json.dumps({"verdict": "accept", "contradiction_span": "", "reason_code": "consistent"}), "Clearly present.")
        self.assertEqual(value["verdict"], "accept")

    def test_reject_requires_exact_span(self) -> None:
        completion = "The finding is not present. <answer>Finding</answer>"
        value = parse_verdict(json.dumps({"verdict": "reject", "contradiction_span": "not present", "reason_code": "material_contradiction"}), completion)
        self.assertEqual(value["verdict"], "reject")
        with self.assertRaisesRegex(ValueError, "evidence"):
            parse_verdict(json.dumps({"verdict": "reject", "contradiction_span": "invented", "reason_code": "material_contradiction"}), completion)


if __name__ == "__main__":
    unittest.main()
