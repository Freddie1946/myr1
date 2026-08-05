#!/usr/bin/env python3
"""Unit tests for the bounded AIGCBest GPT-4o smoke client."""

from __future__ import annotations

import json
import unittest

import run_aigcbest_stage3_gpt4o_smoke as smoke


class AigcBestStage3SmokeTest(unittest.TestCase):
    def test_payload_uses_frozen_contract(self) -> None:
        case = {
            "smoke_kind": "synthetic",
            "image": "/tmp/control.png",
            "problem": "problem",
            "solution": "solution",
            "completion": "completion",
        }
        payload = smoke.make_payload(case, "data:image/png;base64,AA==")
        self.assertEqual(payload["model"], "gpt-4o-2024-08-06")
        self.assertEqual(payload["response_format"]["json_schema"]["name"], "pathvlm_stage3_process_events_v1")
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["seed"], 42)
        self.assertNotIn("reasoning_effort", payload)

    def test_payload_supports_explicit_reasoning_effort(self) -> None:
        case = {
            "smoke_kind": "synthetic",
            "image": "/tmp/control.png",
            "problem": "problem",
            "solution": "solution",
            "completion": "completion",
        }
        payload = smoke.make_payload(
            case,
            "data:image/png;base64,AA==",
            model="gemini-3.1-pro",
            reasoning_effort="low",
        )
        self.assertEqual(payload["model"], "gemini-3.1-pro")
        self.assertEqual(payload["reasoning_effort"], "low")

    def test_parse_strict_success(self) -> None:
        events = {
            "image_feature_analysis_present": True,
            "option_elimination_present": True,
            "medical_knowledge_support_present": False,
            "histological_definition_error": False,
            "logical_contradiction": False,
            "outdated_or_incorrect_pathology_criterion": False,
            "evidence": {
                "image_feature_analysis_present": "feature",
                "option_elimination_present": "option",
                "medical_knowledge_support_present": "",
                "histological_definition_error": "",
                "logical_contradiction": "",
                "outdated_or_incorrect_pathology_criterion": "",
            },
        }
        body = {
            "model": smoke.MODEL,
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(events)}}],
        }
        parsed, scores = smoke.parse_success(body)
        self.assertEqual(parsed, events)
        self.assertEqual(scores["process"], 0.8)

    def test_payload_and_parser_support_explicit_candidate_identity(self) -> None:
        case = {
            "smoke_kind": "pathmmu_validation",
            "image": "/tmp/control.png",
            "problem": "problem",
            "solution": "solution",
            "completion": "completion",
        }
        payload = smoke.make_payload(
            case, "data:image/png;base64,AA==", model="claude-sonnet-4-6"
        )
        self.assertEqual(payload["model"], "claude-sonnet-4-6")
        events = {
            "image_feature_analysis_present": True,
            "option_elimination_present": True,
            "medical_knowledge_support_present": False,
            "histological_definition_error": False,
            "logical_contradiction": False,
            "outdated_or_incorrect_pathology_criterion": False,
            "evidence": {
                "image_feature_analysis_present": "feature",
                "option_elimination_present": "option",
                "medical_knowledge_support_present": "",
                "histological_definition_error": "",
                "logical_contradiction": "",
                "outdated_or_incorrect_pathology_criterion": "",
            },
        }
        body = {
            "model": "claude-sonnet-4-6",
            "choices": [
                {"finish_reason": "stop", "message": {"content": json.dumps(events)}}
            ],
        }
        parsed, _ = smoke.parse_success(body, model="claude-sonnet-4-6")
        self.assertEqual(parsed, events)

    def test_rejects_alias_or_extra_event(self) -> None:
        body = {
            "model": "gpt-4o",
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
        }
        with self.assertRaisesRegex(ValueError, "served model mismatch"):
            smoke.parse_success(body)


if __name__ == "__main__":
    unittest.main()
