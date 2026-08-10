#!/usr/bin/env python3

import json
import unittest

from pathvqa_answer_intent_judge import (
    build_payload,
    parse_intent_judgment,
)


class PathvqaAnswerIntentJudgeTests(unittest.TestCase):
    def test_resolved_requires_exact_evidence(self):
        completion = "The finding is absent, so the final answer is No."
        value = parse_intent_judgment(json.dumps({
            "answer": "no",
            "evidence_span": "the final answer is No",
            "reason_code": "explicit_final",
        }), completion)
        self.assertEqual(value["answer"], "no")
        with self.assertRaisesRegex(ValueError, "exact completion substring"):
            parse_intent_judgment(json.dumps({
                "answer": "no", "evidence_span": "No evidence",
                "reason_code": "entailed_conclusion",
            }), completion)

    def test_unresolved_contract(self):
        value = parse_intent_judgment(json.dumps({
            "answer": "unresolved", "evidence_span": "",
            "reason_code": "no_answer",
        }), "truncated because")
        self.assertEqual(value["answer"], "unresolved")

    def test_payload_has_no_reference_or_model_identity(self):
        payload = build_payload("gpt-4.1-mini-2025-04-14", "question?", "completion")
        text = json.dumps(payload)
        self.assertNotIn("reference", text.lower())
        self.assertNotIn("model identity", text.lower())
        self.assertIn("completion", text.lower())


if __name__ == "__main__":
    unittest.main()
