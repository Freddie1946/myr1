#!/usr/bin/env python3

import json
import unittest
from unittest.mock import patch

from pathvqa_llm_judge import (
    JudgeRequestFailure,
    cache_key,
    parse_judgment,
    request_judgment,
    semantic_inputs,
    served_model_matches,
)


class FakeResponse:
    status = 200

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.body).encode()


class PathVQALLMJudgeTests(unittest.TestCase):
    def test_hosted_smoke_semantic_input_schema(self):
        row = {
            "completion": "positively",
            "score": {
                "answer_type": "free_form",
                "contract_aligned_answer_source": "raw_completion",
                "semantic_judge_input": {
                    "question": "how are the histone subunits charged?",
                    "reference": "positively charged",
                    "candidate": "positively",
                },
            },
        }
        self.assertEqual(
            semantic_inputs(row),
            ("how are the histone subunits charged?", "positively charged", "positively", "raw_completion"),
        )

    def test_valid_correct_and_error(self):
        self.assertTrue(parse_judgment(
            '{"correct":true,"reason":"Equivalent.","error_type":"correct"}'
        )["correct"])
        self.assertFalse(parse_judgment(
            '{"correct":false,"reason":"Does not answer.","error_type":"omission"}'
        )["correct"])

    def test_fenced_json_is_accepted_but_extra_fields_are_not(self):
        value = parse_judgment(
            '```json\n{"correct":false,"reason":"Related only.",'
            '"error_type":"related_but_not_answer"}\n```'
        )
        self.assertFalse(value["correct"])
        with self.assertRaises(ValueError):
            parse_judgment(
                '{"correct":true,"reason":"x","error_type":"correct","score":1}'
            )

    def test_boolean_and_error_type_must_agree(self):
        with self.assertRaises(ValueError):
            parse_judgment(
                '{"correct":true,"reason":"x","error_type":"unclear"}'
            )

    def test_bounded_reason_overrun_is_accepted(self):
        reason = "x" * 200
        value = parse_judgment(
            '{"correct":false,"reason":"' + reason + '","error_type":"unclear"}'
        )
        self.assertEqual(len(value["reason"]), 200)
        with self.assertRaises(ValueError):
            parse_judgment(
                '{"correct":false,"reason":"' + "x" * 241 + '","error_type":"unclear"}'
            )

    def test_cache_key_covers_semantic_inputs(self):
        base = cache_key("q", "a", "c", "m")
        self.assertEqual(base, cache_key("q", "a", "c", "m"))
        self.assertNotEqual(base, cache_key("q2", "a", "c", "m"))
        self.assertNotEqual(base, cache_key("q", "a2", "c", "m"))
        self.assertNotEqual(base, cache_key("q", "a", "c2", "m"))
        self.assertNotEqual(base, cache_key("q", "a", "c", "m2"))

    def test_frozen_gpt_alias_expansion_is_allowed(self):
        self.assertTrue(
            served_model_matches("gpt-4.1-mini", "gpt-4.1-mini-2025-04-14")
        )
        self.assertFalse(served_model_matches("gpt-4.1-mini", "gpt-4.1"))
        self.assertTrue(
            served_model_matches("gpt-5-mini", "gpt-5-mini-2025-08-07")
        )

    @patch("pathvqa_llm_judge.time.sleep", return_value=None)
    @patch("pathvqa_llm_judge.urllib.request.urlopen")
    def test_invalid_enum_is_retried_then_valid_response_is_used(self, urlopen, _sleep):
        invalid = {
            "id": "one",
            "model": "gpt-5-mini",
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
                "correct": False, "reason": "Wrong.", "error_type": "wrong_answer",
            })}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }
        valid = {
            "id": "two",
            "model": "gpt-5-mini",
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
                "correct": False, "reason": "Contradicts reference.",
                "error_type": "contradiction",
            })}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }
        urlopen.side_effect = [FakeResponse(invalid), FakeResponse(valid)]
        judgment, metadata = request_judgment(
            api_key="secret", model="gpt-5-mini", question="q",
            reference="a", candidate="c", timeout_seconds=1,
            max_attempts=3, retry_delays_seconds=(0, 0),
        )
        self.assertEqual(judgment["error_type"], "contradiction")
        self.assertEqual(len(metadata["attempts"]), 2)
        self.assertIn("error_type is invalid", metadata["attempts"][0]["error"])

    @patch("pathvqa_llm_judge.time.sleep", return_value=None)
    @patch("pathvqa_llm_judge.urllib.request.urlopen")
    def test_repeated_invalid_responses_fail_after_bound(self, urlopen, _sleep):
        invalid = {
            "id": "one",
            "model": "gpt-5-mini",
            "choices": [{"finish_reason": "stop", "message": {"content": "not-json"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10},
        }
        urlopen.side_effect = [FakeResponse(invalid), FakeResponse(invalid)]
        with self.assertRaises(JudgeRequestFailure) as raised:
            request_judgment(
                api_key="secret", model="gpt-5-mini", question="q",
                reference="a", candidate="c", timeout_seconds=1,
                max_attempts=2, retry_delays_seconds=(0,),
            )
        self.assertFalse(raised.exception.terminal)
        self.assertEqual(len(raised.exception.attempts), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
