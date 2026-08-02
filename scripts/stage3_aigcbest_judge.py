#!/usr/bin/env python3
"""Fail-closed AIGCBest GPT-4o process reward for matched Stage3 pilots."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable

from stage3_openrouter_judge import (
    BudgetLedger,
    InterprocessRateLimiter,
    InvalidJSONResponseBody,
    JudgeUnavailableForRuleFallback,
    RuleFallbackLimiter,
    SCHEMA_VERSION,
    SYSTEM_PROMPT,
    TRANSIENT_STATUS,
    TransportFailure,
    TransportResponse,
    _read_json_response_body,
    _aligned,
    _image_payload,
    atomic_json,
    canonical_json,
    completion_text,
    env_bool,
    now_iso,
    response_schema,
    sha256_bytes,
    structural_rule_fallback,
    validate_events,
)


API_URL = "https://api2.aigcbest.top/v1/chat/completions"
MODEL_ID = "gpt-4o-2024-08-06"
INPUT_USD_PER_MILLION = 2.5
OUTPUT_USD_PER_MILLION = 10.0
DEFAULT_LIMIT_USD = 6.0
DEFAULT_RESERVE_USD = 0.02
DEFAULT_MAX_HTTP_ATTEMPTS = 800
DEFAULT_MAX_JUDGE_TOKENS = 320
DEFAULT_RETRY_JUDGE_TOKEN_CAPS = (512, 768)
ALLOWED_PILOT_PENALTIES = {0.3, 0.4, 0.5}


Transport = Callable[[dict[str, Any], str, float], TransportResponse]


class RetryableJudgeResponseError(ValueError):
    """A billed response whose transient output can be retried safely."""


class TerminalJudgeResponseError(ValueError):
    """A response proving identity, refusal, or request-contract drift."""


def score_events_at_penalty(value: Any, penalty: float) -> dict[str, float | int]:
    events = validate_events(value)
    integrity_names = (
        "image_feature_analysis_present",
        "option_elimination_present",
        "medical_knowledge_support_present",
    )
    error_names = (
        "histological_definition_error",
        "logical_contradiction",
        "outdated_or_incorrect_pathology_criterion",
    )
    missing_integrity = sum(not events[name] for name in integrity_names)
    present_errors = sum(events[name] for name in error_names)
    integrity = max(0.0, 1.0 - penalty * missing_integrity)
    knowledge = max(0.0, 1.0 - penalty * present_errors)
    return {
        "penalty": penalty,
        "missing_integrity_count": missing_integrity,
        "present_knowledge_error_count": present_errors,
        "integrity": integrity,
        "knowledge": knowledge,
        "process": (integrity + knowledge) / 2.0,
    }


def request_cost(usage: Any) -> float:
    if not isinstance(usage, dict):
        raise ValueError("AIGCBest response lacks usage")
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if type(prompt) is not int or type(completion) is not int:
        raise ValueError("AIGCBest usage token counts are not integers")
    if prompt < 0 or completion < 0:
        raise ValueError("AIGCBest usage token counts are negative")
    return (
        prompt * INPUT_USD_PER_MILLION
        + completion * OUTPUT_USD_PER_MILLION
    ) / 1_000_000


def make_cache_key(
    *,
    image_sha256: str,
    problem: str,
    solution: str,
    completion: str,
    penalty: float,
    max_judge_tokens: int,
    retry_judge_token_caps: tuple[int, ...] = DEFAULT_RETRY_JUDGE_TOKEN_CAPS,
) -> str:
    contract = {
        "schema_version": SCHEMA_VERSION,
        "gateway": "aigcbest",
        "api_url": API_URL,
        "model": MODEL_ID,
        "penalty": penalty,
        "max_judge_tokens": max_judge_tokens,
        "retry_judge_token_caps": list(retry_judge_token_caps),
        "temperature": 0,
        "seed": 42,
        "system_prompt_sha256": sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
        "image_sha256": image_sha256,
        "problem": problem,
        "solution": solution,
        "completion": completion,
    }
    return sha256_bytes(canonical_json(contract).encode("utf-8"))


def default_transport(
    payload: dict[str, Any], api_key: str, timeout_seconds: float
) -> TransportResponse:
    request = urllib.request.Request(
        API_URL,
        data=canonical_json(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "PathVLM-R1 Stage3 GPT-4o pilot",
        },
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            try:
                body = _read_json_response_body(response)
            except InvalidJSONResponseBody as exc:
                raise TransportFailure(
                    f"AIGCBest returned invalid JSON: {exc}",
                    status=int(response.status),
                    headers={key.lower(): value for key, value in response.headers.items()},
                    body=exc.raw.decode("utf-8", errors="replace")[:4000],
                ) from exc
            return TransportResponse(
                status=int(response.status),
                headers={key.lower(): value for key, value in response.headers.items()},
                body=body,
                latency_seconds=time.monotonic() - started,
            )
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise TransportFailure(
            f"AIGCBest HTTP {exc.code}",
            status=int(exc.code),
            headers={key.lower(): value for key, value in exc.headers.items()},
            body=raw[:4000],
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TransportFailure(f"AIGCBest transport failure: {exc}") from exc


def sanitized_request(payload: dict[str, Any], image_sha256: str) -> dict[str, Any]:
    value = json.loads(json.dumps(payload))
    value["messages"][1]["content"][1]["image_url"]["url"] = (
        f"<embedded-image-omitted sha256={image_sha256}>"
    )
    return value


class AigcBestJudge:
    def __init__(
        self,
        root: Path,
        *,
        penalty: float,
        transport: Transport = default_transport,
        api_key: str | None = None,
        limit_usd: float = DEFAULT_LIMIT_USD,
        reserve_usd: float = DEFAULT_RESERVE_USD,
        max_http_attempts: int = DEFAULT_MAX_HTTP_ATTEMPTS,
        max_judge_tokens: int = DEFAULT_MAX_JUDGE_TOKENS,
        retry_judge_token_caps: tuple[int, ...] = DEFAULT_RETRY_JUDGE_TOKEN_CAPS,
        timeout_seconds: float = 180.0,
        retry_delays: tuple[float, ...] = (15.0, 45.0, 90.0),
        minimum_request_interval_seconds: float = 1.0,
    ):
        if penalty not in ALLOWED_PILOT_PENALTIES:
            raise ValueError(f"pilot penalty must be one of {sorted(ALLOWED_PILOT_PENALTIES)}")
        if max_judge_tokens != DEFAULT_MAX_JUDGE_TOKENS:
            raise ValueError(f"pilot max_judge_tokens must equal {DEFAULT_MAX_JUDGE_TOKENS}")
        if (
            not retry_judge_token_caps
            or any(type(cap) is not int or cap <= max_judge_tokens for cap in retry_judge_token_caps)
            or tuple(sorted(set(retry_judge_token_caps))) != retry_judge_token_caps
        ):
            raise ValueError("retry judge token caps must be unique increasing integers above 320")
        if any(delay < 0 for delay in retry_delays):
            raise ValueError("retry delays cannot be negative")
        namespace = os.getenv("PATHVLM_AIGCBEST_CACHE_NAMESPACE", "").strip()
        if not namespace or any(not (char.isalnum() or char in "_.-") for char in namespace):
            raise ValueError("PATHVLM_AIGCBEST_CACHE_NAMESPACE is required and must be safe")
        self.root = root
        self.cache_dir = root / "cache" / namespace
        self.audit_dir = root / "audit" / namespace
        self.penalty = penalty
        self.transport = transport
        self.last_result_source: str | None = None
        self.api_key = api_key if api_key is not None else os.getenv("AIGCBEST_API_KEY", "")
        self.timeout_seconds = timeout_seconds
        self.retry_delays = retry_delays
        self.max_judge_tokens = max_judge_tokens
        self.retry_judge_token_caps = retry_judge_token_caps
        self.ledger = BudgetLedger(
            root / "budget_ledger.json",
            limit_usd=limit_usd,
            reserve_usd=reserve_usd,
            max_unique_requests=max_http_attempts,
        )
        self.rate_limiter = InterprocessRateLimiter(
            root / "rate_limit.json", minimum_request_interval_seconds
        )

    def _request_payload(
        self, *, image_url: str, problem: str, solution: str, completion: str,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        text = (
            "QUESTION:\n"
            + problem
            + "\n\nREFERENCE ANSWER (context only; do not infer process quality from answer match):\n"
            + solution
            + "\n\nCANDIDATE COMPLETION TO AUDIT:\n"
            + completion
        )
        return {
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": SCHEMA_VERSION,
                    "strict": True,
                    "schema": response_schema(),
                },
            },
            "temperature": 0,
            "seed": 42,
            "stream": False,
            "max_tokens": self.max_judge_tokens if max_tokens is None else max_tokens,
        }

    def _append_audit(self, event: dict[str, Any]) -> None:
        rank = int(os.getenv("RANK", "0"))
        path = self.audit_dir / f"rank_{rank:02d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(".lock")
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _parse_response(
        self, response: TransportResponse
    ) -> tuple[dict[str, Any], dict[str, float | int], float, str]:
        body = response.body
        if body.get("error") or body.get("model") != MODEL_ID:
            raise TerminalJudgeResponseError(
                f"invalid or mismatched AIGCBest response model: {body.get('model')!r}"
            )
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise RetryableJudgeResponseError(
                "AIGCBest response must contain exactly one choice"
            )
        choice = choices[0]
        finish_reason = getattr(choice, "get", lambda *_: None)("finish_reason")
        if finish_reason == "length":
            raise RetryableJudgeResponseError("unexpected finish reason: 'length'")
        if not isinstance(choice, dict) or finish_reason != "stop":
            raise TerminalJudgeResponseError(
                f"unexpected finish reason: {finish_reason!r}"
            )
        message = choice.get("message")
        if isinstance(message, dict) and message.get("refusal"):
            raise TerminalJudgeResponseError("AIGCBest response contains a refusal")
        if not isinstance(message, dict):
            raise RetryableJudgeResponseError("AIGCBest response is missing its message")
        content = message.get("content")
        if not isinstance(content, str):
            raise RetryableJudgeResponseError("AIGCBest response content is not a string")
        try:
            events = validate_events(json.loads(content))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise RetryableJudgeResponseError(
                f"AIGCBest response event schema is invalid: {exc}"
            ) from exc
        scores = score_events_at_penalty(events, self.penalty)
        cost = request_cost(body.get("usage"))
        response_id = body.get("id")
        if not isinstance(response_id, str) or not response_id:
            raise TerminalJudgeResponseError("AIGCBest response lacks an id")
        return events, scores, cost, response_id

    def judge_one(
        self,
        *,
        image_path: str,
        image_sha256: str,
        problem: str,
        solution: str,
        completion: str,
        source_metadata: dict[str, Any] | None = None,
    ) -> float:
        self.last_result_source = None
        image = Path(image_path).resolve()
        cache_key = make_cache_key(
            image_sha256=image_sha256,
            problem=problem,
            solution=solution,
            completion=completion,
            penalty=self.penalty,
            max_judge_tokens=self.max_judge_tokens,
            retry_judge_token_caps=self.retry_judge_token_caps,
        )
        cache_path = self.cache_dir / cache_key[:2] / f"{cache_key}.json"
        lock_path = cache_path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("cache_key") != cache_key:
                    raise RuntimeError("cache content-address mismatch")
                scores = score_events_at_penalty(cached.get("events"), self.penalty)
                if abs(float(cached.get("scores", {}).get("process", -1)) - float(scores["process"])) > 1e-12:
                    raise RuntimeError("cached process score mismatch")
                self._append_audit({
                    "timestamp": now_iso(), "cache_key": cache_key, "cache_hit": True,
                    "process_reward": scores["process"], **(source_metadata or {}),
                })
                self.last_result_source = "cache"
                return float(scores["process"])
            if not self.api_key:
                raise RuntimeError("AIGCBEST_API_KEY is not set")
            image_url, image_bytes = _image_payload(image, image_sha256)
            attempts: list[dict[str, Any]] = []
            last_error: Exception | None = None
            response: TransportResponse | None = None
            events: dict[str, Any] | None = None
            scores: dict[str, float | int] | None = None
            cost = 0.0
            response_id = ""
            retryable_response_failures = 0
            for attempt_number, delay in enumerate((0.0, *self.retry_delays), 1):
                if delay:
                    time.sleep(delay)
                max_tokens = (
                    self.max_judge_tokens
                    if retryable_response_failures == 0
                    else self.retry_judge_token_caps[
                        min(retryable_response_failures - 1, len(self.retry_judge_token_caps) - 1)
                    ]
                )
                payload = self._request_payload(
                    image_url=image_url, problem=problem, solution=solution,
                    completion=completion, max_tokens=max_tokens,
                )
                request_sha256 = sha256_bytes(canonical_json(payload).encode("utf-8"))
                attempt_id = f"{cache_key}:http:{uuid.uuid4().hex}"
                self.ledger.reserve(attempt_id, {
                    "cache_key": cache_key,
                    "attempt_number": attempt_number,
                    "model": MODEL_ID,
                    "request_sha256": request_sha256,
                    "max_tokens": max_tokens,
                })
                try:
                    self.rate_limiter.wait()
                    response = self.transport(payload, self.api_key, self.timeout_seconds)
                    try:
                        parsed = self._parse_response(response)
                        events, scores, cost, response_id = parsed
                    except (RetryableJudgeResponseError, TerminalJudgeResponseError) as exc:
                        last_error = exc
                        try:
                            billed = request_cost(response.body.get("usage"))
                        except Exception:
                            billed = self.ledger.reserve_usd
                        self.ledger.commit(attempt_id, billed, {
                            "cache_key": cache_key,
                            "attempt_number": attempt_number,
                            "status": "billed_response_failed_validation",
                            "error": f"{type(exc).__name__}: {exc}",
                            "max_tokens": max_tokens,
                        })
                        attempts.append({
                            "attempt": attempt_number,
                            "http_status": response.status,
                            "latency_seconds": response.latency_seconds,
                            "status": "billed_response_failed_validation",
                            "cost_usd": billed,
                            "error": f"{type(exc).__name__}: {exc}",
                            "max_tokens": max_tokens,
                            "raw_response": response.body,
                        })
                        if isinstance(exc, RetryableJudgeResponseError):
                            retryable_response_failures += 1
                            if attempt_number < len((0.0, *self.retry_delays)):
                                continue
                            break
                        self._append_audit({
                            "timestamp": now_iso(),
                            "cache_key": cache_key,
                            "cache_hit": False,
                            "status": "terminal_billed_response_validation_failure",
                            "attempts": attempts,
                            "error": f"{type(exc).__name__}: {exc}",
                            **(source_metadata or {}),
                        })
                        # A model/provider/schema mismatch is not an outage.
                        # Retrying it could multiply spend and conceal contract
                        # drift, so fail the training segment immediately.
                        raise
                    self.ledger.commit(attempt_id, cost, {
                        "cache_key": cache_key,
                        "attempt_number": attempt_number,
                        "status": "completed",
                        "response_id": response_id,
                        "model": MODEL_ID,
                    })
                    attempts.append({
                        "attempt": attempt_number,
                        "http_status": response.status,
                        "latency_seconds": response.latency_seconds,
                        "status": "completed",
                        "response_id": response_id,
                        "cost_usd": cost,
                        "max_tokens": max_tokens,
                    })
                    break
                except TransportFailure as exc:
                    last_error = exc
                    self.ledger.commit(attempt_id, self.ledger.reserve_usd, {
                        "cache_key": cache_key,
                        "attempt_number": attempt_number,
                        "status": "ambiguous_failure_conservatively_billed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "max_tokens": max_tokens,
                    })
                    attempts.append({
                        "attempt": attempt_number,
                        "http_status": exc.status,
                        "status": "ambiguous_failure_conservatively_billed",
                        "reserved_cost_usd": self.ledger.reserve_usd,
                        "error": str(exc),
                        "body": exc.body,
                        "max_tokens": max_tokens,
                    })
                    retryable_transport = (
                        exc.status is None or exc.status == 200 or exc.status in TRANSIENT_STATUS
                    )
                    if retryable_transport:
                        continue
                    self._append_audit({
                        "timestamp": now_iso(), "cache_key": cache_key,
                        "cache_hit": False,
                        "status": "terminal_nontransient_transport_failure",
                        "attempts": attempts,
                        "error": f"{type(exc).__name__}: {exc}",
                        **(source_metadata or {}),
                    })
                    raise
            if events is None or scores is None or response is None:
                self._append_audit({
                    "timestamp": now_iso(), "cache_key": cache_key, "cache_hit": False,
                    "status": "judge_unavailable_after_bounded_attempts",
                    "attempts": attempts,
                    "error": f"{type(last_error).__name__}: {last_error}",
                    **(source_metadata or {}),
                })
                raise JudgeUnavailableForRuleFallback(str(last_error))
            record = {
                "schema_version": 1,
                "created_at": now_iso(),
                "cache_key": cache_key,
                "request_sha256": request_sha256,
                "request": sanitized_request(payload, image_sha256),
                "response": response.body,
                "response_headers": response.headers,
                "attempts": attempts,
                "served_model": response.body.get("model"),
                "served_provider": "aigcbest",
                "usage": response.body.get("usage"),
                "cost_usd": cost,
                "events": events,
                "scores": scores,
                "source": {
                    "image_path": str(image),
                    "image_sha256": image_sha256,
                    "image_bytes": image_bytes,
                    "problem_sha256": sha256_bytes(problem.encode("utf-8")),
                    "solution_sha256": sha256_bytes(solution.encode("utf-8")),
                    "completion_sha256": sha256_bytes(completion.encode("utf-8")),
                    **(source_metadata or {}),
                },
            }
            atomic_json(cache_path, record)
            self._append_audit({
                "timestamp": now_iso(), "cache_key": cache_key, "cache_hit": False,
                "status": "completed", "process_reward": scores["process"],
                "events": events, "scores": scores, "cost_usd": cost,
                "response_id": response_id, "served_model": response.body.get("model"),
                "usage": response.body.get("usage"), "attempts": attempts,
                "cache_path": str(cache_path), **(source_metadata or {}),
            })
            self.last_result_source = "remote"
            return float(scores["process"])


def process_reward(completions, solution, **kwargs):
    count = len(completions)
    solutions = _aligned(solution, count, "solution")
    image_paths = _aligned(kwargs.get("image_path"), count, "image_path")
    image_hashes = _aligned(kwargs.get("image_sha256"), count, "image_sha256")
    problems = _aligned(kwargs.get("problem"), count, "problem")
    record_indices = _aligned(kwargs.get("record_index"), count, "record_index")
    root_text = os.getenv("PATHVLM_STAGE3_JUDGE_ROOT")
    if not root_text:
        raise RuntimeError("PATHVLM_STAGE3_JUDGE_ROOT is required")
    penalty = float(os.getenv("PATHVLM_STAGE3_PENALTY", "nan"))
    retry_delays = tuple(
        float(value.strip())
        for value in os.getenv("PATHVLM_AIGCBEST_RETRY_DELAYS_SECONDS", "15,45,90").split(",")
        if value.strip()
    )
    retry_token_caps = tuple(
        int(value.strip())
        for value in os.getenv("PATHVLM_AIGCBEST_RETRY_JUDGE_TOKEN_CAPS", "512,768").split(",")
        if value.strip()
    )
    judge = AigcBestJudge(
        Path(root_text),
        penalty=penalty,
        limit_usd=float(os.getenv("PATHVLM_AIGCBEST_LIMIT_USD", "6")),
        reserve_usd=float(os.getenv("PATHVLM_AIGCBEST_RESERVE_USD", "0.02")),
        max_http_attempts=int(os.getenv("PATHVLM_AIGCBEST_MAX_HTTP_ATTEMPTS", "800")),
        max_judge_tokens=int(os.getenv("PATHVLM_AIGCBEST_MAX_JUDGE_TOKENS", "320")),
        retry_judge_token_caps=retry_token_caps,
        retry_delays=retry_delays,
        minimum_request_interval_seconds=float(
            os.getenv("PATHVLM_AIGCBEST_MIN_REQUEST_INTERVAL_SECONDS", "1")
        ),
    )
    fallback_enabled = env_bool("PATHVLM_STAGE3_RULE_FALLBACK_ENABLED", False)
    fallback_limiter = (
        RuleFallbackLimiter(
            Path(root_text) / "rule_fallback_ledger.json",
            total_limit=int(os.getenv("PATHVLM_STAGE3_RULE_FALLBACK_TOTAL_LIMIT", "24")),
            consecutive_limit=int(os.getenv("PATHVLM_STAGE3_RULE_FALLBACK_CONSECUTIVE_LIMIT", "4")),
        )
        if fallback_enabled
        else None
    )
    rewards: list[float] = []
    for index in range(count):
        completion = completion_text(completions[index])
        metadata = {
            "record_index": int(record_indices[index]),
            "rank": int(os.getenv("RANK", "0")),
            "local_rank": int(os.getenv("LOCAL_RANK", "0")),
            "training_segment": os.getenv("PATHVLM_TRAINING_SEGMENT", "unspecified"),
        }
        try:
            reward = judge.judge_one(
                image_path=str(image_paths[index]),
                image_sha256=str(image_hashes[index]),
                problem=str(problems[index]),
                solution=str(solutions[index]),
                completion=completion,
                source_metadata=metadata,
            )
            if (
                fallback_limiter is not None
                and judge.last_result_source == "remote"
            ):
                fallback_limiter.record_judge_success()
        except JudgeUnavailableForRuleFallback as exc:
            if fallback_limiter is None:
                raise
            reward, features = structural_rule_fallback(completion)
            state = fallback_limiter.record_fallback({
                **metadata,
                "reward": reward,
                "features": features,
                "judge_error": str(exc),
                "completion_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(),
            })
            judge._append_audit({
                "timestamp": now_iso(),
                "status": "bounded_structural_rule_fallback",
                "process_reward": reward,
                "features": features,
                "fallback_total_used": state["total_used"],
                "fallback_consecutive_used": state["consecutive_used"],
                **metadata,
            })
        rewards.append(float(reward))
    return rewards
