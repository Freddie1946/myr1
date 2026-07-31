#!/usr/bin/env python3
"""CUDA-free tests for the AIGCBest Stage3 training Judge."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from stage3_aigcbest_judge import (
    AigcBestJudge,
    MODEL_ID,
    TransportResponse,
    make_cache_key,
    request_cost,
    score_events_at_penalty,
)


def events() -> dict:
    names = (
        "image_feature_analysis_present",
        "option_elimination_present",
        "medical_knowledge_support_present",
        "histological_definition_error",
        "logical_contradiction",
        "outdated_or_incorrect_pathology_criterion",
    )
    return {
        names[0]: True,
        names[1]: False,
        names[2]: False,
        names[3]: False,
        names[4]: True,
        names[5]: False,
        "evidence": {name: "e" if name in {names[0], names[4]} else "" for name in names},
    }


class ScoringTests(unittest.TestCase):
    def test_penalties_are_materially_distinct(self) -> None:
        self.assertAlmostEqual(score_events_at_penalty(events(), 0.3)["process"], 0.55)
        self.assertAlmostEqual(score_events_at_penalty(events(), 0.4)["process"], 0.4)
        self.assertAlmostEqual(score_events_at_penalty(events(), 0.5)["process"], 0.25)

    def test_cost_uses_frozen_rates(self) -> None:
        self.assertAlmostEqual(request_cost({"prompt_tokens": 1000, "completion_tokens": 100}), 0.0035)

    def test_cache_key_includes_penalty(self) -> None:
        values = dict(image_sha256="a", problem="p", solution="s", completion="c", max_judge_tokens=320)
        self.assertNotEqual(make_cache_key(**values, penalty=0.3), make_cache_key(**values, penalty=0.4))


class JudgeTests(unittest.TestCase):
    def test_one_success_commits_one_http_attempt_and_caches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "image.png"
            image.write_bytes(b"png")
            digest = __import__("hashlib").sha256(b"png").hexdigest()
            body = {
                "id": "response-1",
                "model": MODEL_ID,
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(events())}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
            }
            calls = []

            def transport(payload, key, timeout):
                calls.append(payload)
                return TransportResponse(status=200, headers={}, body=body, latency_seconds=0.1)

            with patch.dict(os.environ, {"PATHVLM_AIGCBEST_CACHE_NAMESPACE": "segment"}):
                judge = AigcBestJudge(
                    Path(directory) / "judge", penalty=0.4, transport=transport,
                    api_key="secret", limit_usd=1, reserve_usd=0.02,
                    max_http_attempts=2, minimum_request_interval_seconds=0,
                )
                first = judge.judge_one(
                    image_path=str(image), image_sha256=digest, problem="p", solution="s",
                    completion="c", source_metadata={"record_index": 1},
                )
                second = judge.judge_one(
                    image_path=str(image), image_sha256=digest, problem="p", solution="s",
                    completion="c", source_metadata={"record_index": 1},
                )
                self.assertEqual(first, second)
                self.assertEqual(len(calls), 1)
                ledger = judge.ledger.snapshot()
                self.assertEqual(ledger["completed_unique_requests"], 1)
                self.assertAlmostEqual(ledger["committed_spend_usd"], 0.0035)
                cache = next((Path(directory) / "judge" / "cache" / "segment").rglob("*.json"))
                record = json.loads(cache.read_text())
                self.assertIn("embedded-image-omitted", record["request"]["messages"][1]["content"][1]["image_url"]["url"])

    def test_invalid_penalty_and_token_ceiling_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"PATHVLM_AIGCBEST_CACHE_NAMESPACE": "segment"}
        ):
            with self.assertRaises(ValueError):
                AigcBestJudge(Path(directory), penalty=0.2)
            with self.assertRaises(ValueError):
                AigcBestJudge(Path(directory), penalty=0.4, max_judge_tokens=321)


if __name__ == "__main__":
    unittest.main()
