#!/usr/bin/env python3
"""CUDA-free regression tests for the approved Stage3 OpenRouter Judge."""

from __future__ import annotations

import hashlib
import http.client
import json
import tempfile
import unittest
from pathlib import Path

from stage3_openrouter_judge import (
    DEFAULT_MAX_UNIQUE_REQUESTS,
    EVENTS,
    MODEL_ID,
    BudgetError,
    BudgetLedger,
    OpenRouterJudge,
    TransportFailure,
    TransportResponse,
    _read_json_response_body,
    make_cache_key,
    response_schema,
    score_events,
    validate_events,
)


class _IncompleteBody:
    def __init__(self, partial: bytes):
        self.partial = partial

    def read(self):
        raise http.client.IncompleteRead(self.partial)


def event_payload(*, integrity: bool = True, errors: bool = False):
    return {
        **{name: integrity for name in EVENTS[:3]},
        **{name: errors for name in EVENTS[3:]},
        "evidence": {name: "" for name in EVENTS},
    }


def response_for(
    events,
    *,
    cost: float = 0.001,
    model: str = MODEL_ID,
    provider: str = "azure",
):
    return TransportResponse(
        status=200,
        headers={"x-generation-id": "gen-test"},
        body={
            "id": "gen-test",
            "model": model,
            "provider": provider,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(events),
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "total_tokens": 140,
                "cost": cost,
            },
        },
        latency_seconds=0.01,
    )


class SchemaAndScoringTests(unittest.TestCase):
    def test_complete_json_from_missing_chunk_trailer_is_accepted(self):
        payload = {"id": "gen-test", "choices": []}
        body = _IncompleteBody(json.dumps(payload).encode("utf-8"))
        self.assertEqual(_read_json_response_body(body), payload)

    def test_genuinely_truncated_json_still_fails_closed(self):
        body = _IncompleteBody(b'{"id":"gen-test","choices":[')
        with self.assertRaises(json.JSONDecodeError):
            _read_json_response_body(body)

    def test_schema_is_strict_and_complete(self):
        schema = response_schema()
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), {*EVENTS, "evidence"})
        self.assertEqual(
            set(schema["properties"]["evidence"]["required"]), set(EVENTS)
        )
        for name in EVENTS:
            self.assertEqual(
                schema["properties"]["evidence"]["properties"][name]["maxLength"],
                512,
            )

    def test_all_good_scores_one(self):
        scores = score_events(event_payload())
        self.assertEqual(scores["integrity"], 1.0)
        self.assertEqual(scores["knowledge"], 1.0)
        self.assertEqual(scores["process"], 1.0)

    def test_penalty_point_four_is_local_and_deterministic(self):
        value = event_payload()
        value[EVENTS[0]] = False
        value[EVENTS[1]] = False
        value[EVENTS[3]] = True
        scores = score_events(value)
        self.assertAlmostEqual(scores["integrity"], 0.2)
        self.assertAlmostEqual(scores["knowledge"], 0.6)
        self.assertAlmostEqual(scores["process"], 0.4)

    def test_three_missing_and_three_errors_floor_at_zero(self):
        scores = score_events(event_payload(integrity=False, errors=True))
        self.assertEqual(scores["integrity"], 0.0)
        self.assertEqual(scores["knowledge"], 0.0)
        self.assertEqual(scores["process"], 0.0)

    def test_extra_key_and_integer_boolean_are_rejected(self):
        extra = event_payload()
        extra["score"] = 1
        with self.assertRaises(ValueError):
            validate_events(extra)
        wrong_type = event_payload()
        wrong_type[EVENTS[0]] = 1
        with self.assertRaises(ValueError):
            validate_events(wrong_type)
        long_evidence = event_payload()
        long_evidence["evidence"][EVENTS[0]] = "x" * 513
        with self.assertRaises(ValueError):
            validate_events(long_evidence)

    def test_cache_key_changes_with_scientific_inputs(self):
        base = dict(
            image_sha256="a" * 64,
            problem="question",
            solution="A",
            completion="<think>x</think><answer>A</answer>",
        )
        first = make_cache_key(**base)
        self.assertEqual(first, make_cache_key(**base))
        for field in base:
            changed = dict(base)
            changed[field] += "x"
            self.assertNotEqual(first, make_cache_key(**changed))


class LedgerTests(unittest.TestCase):
    def test_reserve_commit_and_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(
                Path(directory) / "ledger.json",
                limit_usd=0.10,
                reserve_usd=0.05,
                max_unique_requests=3,
            )
            ledger.reserve("one", {})
            ledger.reserve("two", {})
            with self.assertRaises(BudgetError):
                ledger.reserve("three", {})
            ledger.commit("one", 0.01, {})
            with self.assertRaises(BudgetError):
                ledger.reserve("three", {})
            ledger.commit("two", 0.01, {})
            ledger.reserve("three", {})
            snapshot = ledger.snapshot()
            self.assertAlmostEqual(snapshot["committed_spend_usd"], 0.02)
            self.assertEqual(snapshot["completed_unique_requests"], 2)
            self.assertEqual(set(snapshot["reservations"]), {"three"})

    def test_failed_reservation_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(Path(directory) / "ledger.json")
            ledger.reserve("request", {})
            ledger.retain_failed_reservation(
                "request", error="transport ambiguous", attempts=3
            )
            reservation = ledger.snapshot()["reservations"]["request"]
            self.assertEqual(
                reservation["status"], "unresolved_potentially_billed_failure"
            )
            with self.assertRaises(BudgetError):
                ledger.reserve("request", {})

    def test_proven_unbilled_release_requires_complete_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(Path(directory) / "ledger.json")
            ledger.reserve("request", {})
            ledger.retain_failed_reservation("request", error="router 404", attempts=1)
            with self.assertRaises(BudgetError):
                ledger.release_proven_unbilled(
                    "request", evidence={"no_endpoint_selected": True}
                )
            ledger.release_proven_unbilled(
                "request",
                evidence={
                    "no_endpoint_selected": True,
                    "key_usage_unchanged": True,
                    "account_total_usage_unchanged": True,
                },
            )
            snapshot = ledger.snapshot()
            self.assertFalse(snapshot["reservations"])
            self.assertEqual(len(snapshot["unbilled_releases"]), 1)

    def test_unique_request_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            ledger = BudgetLedger(
                Path(directory) / "ledger.json",
                max_unique_requests=1,
            )
            ledger.reserve("one", {})
            ledger.commit("one", 0.001, {})
            with self.assertRaises(BudgetError):
                ledger.reserve("two", {})

    def test_frozen_default_includes_smoke_plus_training(self):
        self.assertEqual(DEFAULT_MAX_UNIQUE_REQUESTS, 401)


class JudgeTests(unittest.TestCase):
    def _source(self, directory: str):
        image = Path(directory) / "sample.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\nnot-decoded-by-the-client")
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        return image, digest

    def test_mocked_success_is_cached_and_charged_once(self):
        with tempfile.TemporaryDirectory() as directory:
            image, digest = self._source(directory)
            calls = []

            def transport(payload, api_key, timeout):
                calls.append((payload, api_key, timeout))
                return response_for(event_payload(), cost=0.002)

            root = Path(directory) / "judge"
            judge = OpenRouterJudge(
                root,
                transport=transport,
                api_key="test-only-not-real",
                retry_delays=(),
            )
            arguments = {
                "image_path": str(image),
                "image_sha256": digest,
                "problem": "What is shown?",
                "solution": "A",
                "completion": "<think>Image features support A.</think><answer>A</answer>",
                "source_metadata": {"record_index": 7},
            }
            self.assertEqual(judge.judge_one(**arguments), 1.0)
            self.assertEqual(judge.judge_one(**arguments), 1.0)
            self.assertEqual(len(calls), 1)
            request = calls[0][0]
            self.assertEqual(request["provider"]["only"], ["azure"])
            self.assertFalse(request["provider"]["allow_fallbacks"])
            self.assertTrue(request["provider"]["require_parameters"])
            self.assertTrue(request["provider"]["zdr"])
            self.assertEqual(request["provider"]["data_collection"], "deny")
            self.assertTrue(request["provider"]["enforce_distillable_text"])
            self.assertEqual(request["max_completion_tokens"], 320)
            self.assertNotIn("max_tokens", request)
            self.assertTrue(
                request["response_format"]["json_schema"]["strict"]
            )
            snapshot = judge.ledger.snapshot()
            self.assertEqual(snapshot["completed_unique_requests"], 1)
            self.assertAlmostEqual(snapshot["committed_spend_usd"], 0.002)
            self.assertFalse(snapshot["reservations"])
            self.assertEqual(len(list((root / "cache").rglob("*.json"))), 1)
            audit_rows = [
                json.loads(line)
                for line in (root / "audit" / "rank_00.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([row["cache_hit"] for row in audit_rows], [False, True])

    def test_hash_mismatch_refuses_transport_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            image, _ = self._source(directory)

            def forbidden(*args):
                raise AssertionError("transport must not be called")

            judge = OpenRouterJudge(
                Path(directory) / "judge",
                transport=forbidden,
                api_key="test-only-not-real",
                retry_delays=(),
            )
            with self.assertRaises(RuntimeError):
                judge.judge_one(
                    image_path=str(image),
                    image_sha256="0" * 64,
                    problem="q",
                    solution="A",
                    completion="c",
                )
            self.assertFalse((Path(directory) / "judge" / "budget_ledger.json").exists())

    def test_kimi_contract_is_distillable_zdr_and_provider_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            image, digest = self._source(directory)
            calls = []

            def transport(payload, api_key, timeout):
                calls.append(payload)
                return response_for(
                    event_payload(),
                    model="moonshotai/kimi-k2.6",
                    provider="Inceptron",
                )

            judge = OpenRouterJudge(
                Path(directory) / "judge",
                transport=transport,
                api_key="test-only-not-real",
                retry_delays=(),
                model_id="moonshotai/kimi-k2.6",
                provider_slugs=("inceptron",),
                max_token_parameter="max_tokens",
                max_judge_tokens=640,
                zdr_required=True,
                reasoning_enabled=False,
            )
            reward = judge.judge_one(
                image_path=str(image),
                image_sha256=digest,
                problem="q",
                solution="A",
                completion="c",
            )
            self.assertEqual(reward, 1.0)
            request = calls[0]
            self.assertEqual(request["model"], "moonshotai/kimi-k2.6")
            self.assertEqual(request["provider"]["only"], ["inceptron"])
            self.assertTrue(request["provider"]["zdr"])
            self.assertTrue(request["provider"]["enforce_distillable_text"])
            self.assertEqual(request["reasoning"], {"enabled": False})
            self.assertEqual(request["max_tokens"], 640)
            self.assertNotIn("max_completion_tokens", request)

    def test_persistent_transport_failure_retains_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            image, digest = self._source(directory)
            calls = []

            def transport(*args):
                calls.append(1)
                raise TransportFailure("rate limited", status=429)

            judge = OpenRouterJudge(
                Path(directory) / "judge",
                transport=transport,
                api_key="test-only-not-real",
                retry_delays=(0.0, 0.0),
            )
            with self.assertRaises(TransportFailure):
                judge.judge_one(
                    image_path=str(image),
                    image_sha256=digest,
                    problem="q",
                    solution="A",
                    completion="c",
                )
            self.assertEqual(len(calls), 3)
            reservations = judge.ledger.snapshot()["reservations"]
            self.assertEqual(len(reservations), 1)
            self.assertEqual(
                next(iter(reservations.values()))["status"],
                "unresolved_potentially_billed_failure",
            )

    def test_non_frozen_provider_response_is_rejected_and_billed(self):
        with tempfile.TemporaryDirectory() as directory:
            image, digest = self._source(directory)

            def transport(*args):
                response = response_for(event_payload())
                response.body["provider"] = "openai"
                return response

            judge = OpenRouterJudge(
                Path(directory) / "judge",
                transport=transport,
                api_key="test-only-not-real",
                retry_delays=(),
                metadata_lookup=lambda *args: {"provider_name": "OpenAI"},
            )
            with self.assertRaises(ValueError):
                judge.judge_one(
                    image_path=str(image),
                    image_sha256=digest,
                    problem="q",
                    solution="A",
                    completion="c",
                )
            snapshot = judge.ledger.snapshot()
            self.assertFalse(snapshot["reservations"])
            self.assertEqual(snapshot["completed_unique_requests"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
