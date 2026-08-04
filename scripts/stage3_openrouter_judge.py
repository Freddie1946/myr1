#!/usr/bin/env python3
"""Fail-closed OpenRouter process reward for the approved Stage3 pilot."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import http.client
import json
import mimetypes
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = "pathvlm_stage3_process_events_v1"
MODEL_ID = "openai/gpt-4o-2024-08-06"
PROVIDER_SLUG = "azure"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
GENERATION_URL = "https://openrouter.ai/api/v1/generation"
PENALTY = 0.4
DEFAULT_LIMIT_USD = 15.0
DEFAULT_RESERVE_USD = 0.05
DEFAULT_MAX_UNIQUE_REQUESTS = 401  # one smoke plus at most 400 training items
EVIDENCE_TARGET_CHARACTERS = 160
EVIDENCE_MAX_CHARACTERS = 512
TRANSIENT_STATUS = {408, 429, 500, 502, 503, 520, 529}
AUDIT_RESPONSE_HEADERS = (
    "cf-ray",
    "retry-after",
    "x-generation-id",
    "x-openrouter-request-id",
    "x-request-id",
)
INTEGRITY_EVENTS = (
    "image_feature_analysis_present",
    "option_elimination_present",
    "medical_knowledge_support_present",
)
ERROR_EVENTS = (
    "histological_definition_error",
    "logical_contradiction",
    "outdated_or_incorrect_pathology_criterion",
)
EVENTS = INTEGRITY_EVENTS + ERROR_EVENTS

SYSTEM_PROMPT = """You are a pathology reasoning-process auditor.
Evaluate only whether the candidate reasoning is a medically sound, image-grounded process for
the supplied multiple-choice pathology question. The reference answer is context for checking
medical correctness; never award process credit merely because the candidate's final answer
matches it.

Return the six required booleans and one short evidence span for each event. For the first three
events, true means the component is genuinely present in the candidate reasoning. Option
elimination may be explicit or implicit and no fixed number of options must be eliminated. For
the last three events, true means the named error is present. Evidence must quote or tightly
point to the candidate reasoning; use an empty string when no supporting span exists. Keep each
evidence span within 160 characters whenever possible; the absolute schema ceiling is 512
characters. Do not add fields, scores, commentary, or a corrected answer."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_provider(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def audit_response_headers(headers: dict[str, str]) -> dict[str, str]:
    """Retain request-tracing headers without logging cookies or credentials."""

    return {
        name: str(headers[name])[:512]
        for name in AUDIT_RESPONSE_HEADERS
        if headers.get(name)
    }


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, found {raw!r}")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def completion_text(completion: Any) -> str:
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return str(completion[0].get("content", ""))
    return str(completion)


def response_schema() -> dict[str, Any]:
    event_properties = {
        name: {
            "type": "boolean",
            "description": (
                "True only when this event is present in the candidate reasoning."
            ),
        }
        for name in EVENTS
    }
    evidence_properties = {
        name: {
            "type": "string",
            "description": "A short verbatim or tightly localized evidence span; empty if absent.",
            "maxLength": EVIDENCE_MAX_CHARACTERS,
        }
        for name in EVENTS
    }
    return {
        "type": "object",
        "properties": {
            **event_properties,
            "evidence": {
                "type": "object",
                "properties": evidence_properties,
                "required": list(EVENTS),
                "additionalProperties": False,
            },
        },
        "required": [*EVENTS, "evidence"],
        "additionalProperties": False,
    }


def validate_events(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Judge output must be a JSON object")
    expected = {*EVENTS, "evidence"}
    if set(value) != expected:
        raise ValueError(
            f"Judge output keys differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )
    for name in EVENTS:
        if type(value[name]) is not bool:  # bool is intentionally exact, not int
            raise ValueError(f"Judge event is not boolean: {name}")
    evidence = value["evidence"]
    if not isinstance(evidence, dict) or set(evidence) != set(EVENTS):
        raise ValueError("Judge evidence keys differ from the frozen event set")
    for name in EVENTS:
        span = evidence[name]
        if not isinstance(span, str):
            raise ValueError(f"Judge evidence is not a string: {name}")
        if len(span) > EVIDENCE_MAX_CHARACTERS:
            raise ValueError(
                "Judge evidence exceeds the local "
                f"{EVIDENCE_MAX_CHARACTERS}-character ceiling: {name}"
            )
    return value


def score_events(value: Any) -> dict[str, float | int]:
    events = validate_events(value)
    missing_integrity = sum(not events[name] for name in INTEGRITY_EVENTS)
    present_errors = sum(events[name] for name in ERROR_EVENTS)
    integrity = max(0.0, 1.0 - PENALTY * missing_integrity)
    knowledge = max(0.0, 1.0 - PENALTY * present_errors)
    process = (integrity + knowledge) / 2.0
    return {
        "missing_integrity_count": missing_integrity,
        "present_knowledge_error_count": present_errors,
        "integrity": integrity,
        "knowledge": knowledge,
        "process": process,
    }


def make_cache_key(
    *,
    image_sha256: str,
    problem: str,
    solution: str,
    completion: str,
    model_id: str = MODEL_ID,
    provider_slugs: tuple[str, ...] = (PROVIDER_SLUG,),
    max_token_parameter: str = "max_completion_tokens",
    max_judge_tokens: int = 320,
    zdr_required: bool = True,
    data_collection: str = "deny",
    reasoning_enabled: bool | None = None,
) -> str:
    contract = {
        "schema_version": SCHEMA_VERSION,
        "model": model_id,
        "providers": list(provider_slugs),
        "max_token_parameter": max_token_parameter,
        "max_judge_tokens": max_judge_tokens,
        "zdr_required": zdr_required,
        "data_collection": data_collection,
        "reasoning_enabled": reasoning_enabled,
        "penalty": PENALTY,
        "system_prompt_sha256": sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
        "image_sha256": image_sha256,
        "problem": problem,
        "solution": solution,
        "completion": completion,
    }
    return sha256_bytes(canonical_json(contract).encode("utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class BudgetError(RuntimeError):
    """Raised when the client-side budget ledger refuses or cannot settle a call."""


class BudgetLedger:
    """Inter-process, fail-closed reservation ledger."""

    def __init__(
        self,
        path: Path,
        *,
        limit_usd: float = DEFAULT_LIMIT_USD,
        reserve_usd: float = DEFAULT_RESERVE_USD,
        max_unique_requests: int = DEFAULT_MAX_UNIQUE_REQUESTS,
    ):
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")
        self.limit_usd = float(limit_usd)
        self.reserve_usd = float(reserve_usd)
        self.max_unique_requests = int(max_unique_requests)
        if not 0 < self.reserve_usd <= self.limit_usd:
            raise ValueError("invalid per-request reserve")
        if self.max_unique_requests <= 0:
            raise ValueError("invalid unique-request limit")

    def _default(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "limit_usd": self.limit_usd,
            "reserve_usd": self.reserve_usd,
            "max_unique_requests": self.max_unique_requests,
            "committed_spend_usd": 0.0,
            "completed_unique_requests": 0,
            "reservations": {},
            "history": [],
            "budget_breached": False,
        }

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._default()
        value = json.loads(self.path.read_text(encoding="utf-8"))
        frozen = {
            "limit_usd": self.limit_usd,
            "reserve_usd": self.reserve_usd,
            "max_unique_requests": self.max_unique_requests,
        }
        mismatch = {
            key: {"expected": expected, "actual": value.get(key)}
            for key, expected in frozen.items()
            if value.get(key) != expected
        }
        if mismatch:
            raise BudgetError(f"budget ledger contract mismatch: {mismatch}")
        return value

    def _locked(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return handle

    def reserve(self, request_id: str, metadata: dict[str, Any]) -> None:
        handle = self._locked()
        try:
            ledger = self._read()
            reservations = ledger["reservations"]
            if request_id in reservations:
                raise BudgetError(
                    f"request has an unresolved prior reservation: {request_id}"
                )
            occupied = int(ledger["completed_unique_requests"]) + len(reservations)
            if occupied >= self.max_unique_requests:
                raise BudgetError("maximum unique OpenRouter request count reached")
            outstanding = sum(float(row["amount_usd"]) for row in reservations.values())
            projected = (
                float(ledger["committed_spend_usd"])
                + outstanding
                + self.reserve_usd
            )
            if projected > self.limit_usd + 1e-12:
                raise BudgetError(
                    f"USD {self.limit_usd:.2f} hard cap refuses next reservation; "
                    f"projected={projected:.8f}"
                )
            reservations[request_id] = {
                "amount_usd": self.reserve_usd,
                "reserved_at": now_iso(),
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                **metadata,
            }
            ledger["updated_at"] = now_iso()
            atomic_json(self.path, ledger)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def commit(self, request_id: str, cost_usd: float, metadata: dict[str, Any]) -> None:
        if cost_usd < 0:
            raise BudgetError("negative response cost")
        handle = self._locked()
        breached = False
        try:
            ledger = self._read()
            reservation = ledger["reservations"].pop(request_id, None)
            if reservation is None:
                raise BudgetError(f"missing reservation at commit: {request_id}")
            ledger["committed_spend_usd"] = (
                float(ledger["committed_spend_usd"]) + float(cost_usd)
            )
            ledger["completed_unique_requests"] = (
                int(ledger["completed_unique_requests"]) + 1
            )
            history = ledger.setdefault("history", [])
            history.append(
                {
                    "request_id": request_id,
                    "committed_at": now_iso(),
                    "cost_usd": float(cost_usd),
                    "reserved_usd": float(reservation["amount_usd"]),
                    **metadata,
                }
            )
            if float(cost_usd) > float(reservation["amount_usd"]) + 1e-12:
                ledger["budget_breached"] = True
                ledger["breach_reason"] = "actual_call_cost_exceeded_conservative_reserve"
                breached = True
            if float(ledger["committed_spend_usd"]) > self.limit_usd + 1e-12:
                ledger["budget_breached"] = True
                ledger["breach_reason"] = "committed_spend_exceeded_hard_cap"
                breached = True
            ledger["updated_at"] = now_iso()
            atomic_json(self.path, ledger)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        if breached:
            raise BudgetError("response cost breached the conservative budget reservation")

    def retain_failed_reservation(
        self, request_id: str, *, error: str, attempts: int
    ) -> None:
        handle = self._locked()
        try:
            ledger = self._read()
            reservation = ledger["reservations"].get(request_id)
            if reservation is None:
                raise BudgetError(f"missing failed reservation: {request_id}")
            reservation.update(
                {
                    "status": "unresolved_potentially_billed_failure",
                    "failure_recorded_at": now_iso(),
                    "error": error,
                    "attempts": attempts,
                }
            )
            ledger["updated_at"] = now_iso()
            atomic_json(self.path, ledger)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def release_proven_unbilled(
        self, request_id: str, *, evidence: dict[str, Any]
    ) -> None:
        """Release only a router-rejected request proven not to have been billed."""
        required = {
            "no_endpoint_selected": True,
            "key_usage_unchanged": True,
            "account_total_usage_unchanged": True,
        }
        if any(evidence.get(key) is not expected for key, expected in required.items()):
            raise BudgetError("unbilled-release evidence is incomplete")
        handle = self._locked()
        try:
            ledger = self._read()
            reservation = ledger["reservations"].pop(request_id, None)
            if reservation is None:
                raise BudgetError(f"missing reservation at unbilled release: {request_id}")
            ledger.setdefault("unbilled_releases", []).append(
                {
                    "request_id": request_id,
                    "released_at": now_iso(),
                    "prior_reservation": reservation,
                    "evidence": evidence,
                }
            )
            ledger["updated_at"] = now_iso()
            atomic_json(self.path, ledger)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def snapshot(self) -> dict[str, Any]:
        handle = self._locked()
        try:
            return self._read()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


class InterprocessRateLimiter:
    """Serialize request starts across ranks with a frozen minimum interval."""

    def __init__(self, path: Path, minimum_interval_seconds: float):
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")
        self.minimum_interval_seconds = float(minimum_interval_seconds)
        if self.minimum_interval_seconds < 0:
            raise ValueError("minimum OpenRouter request interval cannot be negative")

    def wait(self) -> None:
        if self.minimum_interval_seconds == 0:
            return
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            state: dict[str, Any] = {}
            if self.path.exists():
                state = json.loads(self.path.read_text(encoding="utf-8"))
                frozen = state.get("minimum_interval_seconds")
                if frozen != self.minimum_interval_seconds:
                    raise RuntimeError(
                        "OpenRouter rate-limit contract mismatch: "
                        f"expected {self.minimum_interval_seconds}, found {frozen}"
                    )
            now = time.time()
            previous = float(state.get("last_request_started_at_epoch", 0.0))
            delay = previous + self.minimum_interval_seconds - now
            if delay > 0:
                time.sleep(delay)
            started = time.time()
            atomic_json(
                self.path,
                {
                    "schema_version": 1,
                    "minimum_interval_seconds": self.minimum_interval_seconds,
                    "last_request_started_at_epoch": started,
                    "last_request_started_at": datetime.fromtimestamp(
                        started, timezone.utc
                    ).isoformat(),
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                },
            )
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


@dataclass
class TransportResponse:
    status: int
    headers: dict[str, str]
    body: dict[str, Any]
    latency_seconds: float


class TransportFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        headers: dict[str, str] | None = None,
        body: str = "",
        ambiguous_success_body: bool = False,
    ):
        super().__init__(message)
        self.status = status
        self.headers = headers or {}
        self.body = body
        self.ambiguous_success_body = bool(ambiguous_success_body)


class InvalidJSONResponseBody(ValueError):
    """A success-status response body that cannot be verified as complete JSON."""

    def __init__(self, message: str, *, raw: bytes, incomplete_read: bool):
        super().__init__(message)
        self.raw = raw
        self.incomplete_read = incomplete_read


class RetryableResponseValidationError(ValueError):
    """A billed response is malformed/truncated but does not violate frozen identity."""


class JudgeUnavailableForRuleFallback(RuntimeError):
    """An unavailable transport was conservatively billed and may use local fallback."""


class RuleFallbackLimitExceeded(RuntimeError):
    """The bounded local fallback contract refuses further rule rewards."""


Transport = Callable[[dict[str, Any], str, float], TransportResponse]


def _read_json_response_body(response: Any) -> dict[str, Any]:
    """Read a JSON body, accepting a complete body from a truncated chunk trailer.

    Some HTTP/1.1 relays deliver every JSON byte and then close the connection before
    sending the terminal zero-length chunk.  ``http.client`` raises ``IncompleteRead``
    in that case even though ``exc.partial`` is a complete, parseable response.  It is
    safe to use the partial bytes only when strict JSON parsing reaches the end of the
    supplied body; genuinely truncated JSON continues to fail closed.
    """

    incomplete_read = False
    try:
        raw = response.read()
    except http.client.IncompleteRead as exc:
        raw = exc.partial
        incomplete_read = True
    if not isinstance(raw, bytes):
        raise InvalidJSONResponseBody(
            "HTTP response body must be bytes",
            raw=b"",
            incomplete_read=incomplete_read,
        )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        qualifier = "incomplete" if incomplete_read else "invalid"
        raise InvalidJSONResponseBody(
            f"OpenRouter returned {qualifier} JSON: {exc}",
            raw=raw,
            incomplete_read=incomplete_read,
        ) from exc
    if not isinstance(value, dict):
        raise InvalidJSONResponseBody(
            "OpenRouter response body must be a JSON object",
            raw=raw,
            incomplete_read=incomplete_read,
        )
    return value


def default_transport(
    payload: dict[str, Any], api_key: str, timeout_seconds: float
) -> TransportResponse:
    encoded = canonical_json(payload).encode("utf-8")
    request = urllib.request.Request(
        API_URL,
        data=encoded,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Freddie1946/myr1",
            "X-OpenRouter-Title": "PathVLM-R1 Stage3 Pilot",
            "X-OpenRouter-Metadata": "enabled",
            "X-OpenRouter-Cache": "false",
        },
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = int(response.status)
            headers = {key.lower(): value for key, value in response.headers.items()}
            try:
                body = _read_json_response_body(response)
            except InvalidJSONResponseBody as exc:
                raise TransportFailure(
                    str(exc),
                    status=status,
                    headers=headers,
                    body=exc.raw.decode("utf-8", errors="replace")[:4000],
                    # A 2xx response proves that the provider accepted the request,
                    # but an incomplete/invalid body cannot prove its exact cost or
                    # safely identify a response to retry.  The caller therefore
                    # settles the reserve once and may use the bounded local fallback.
                    ambiguous_success_body=200 <= status < 300,
                ) from exc
            return TransportResponse(
                status=status,
                headers=headers,
                body=body,
                latency_seconds=time.monotonic() - started,
            )
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise TransportFailure(
            f"OpenRouter HTTP {exc.code}",
            status=int(exc.code),
            headers={key.lower(): value for key, value in exc.headers.items()},
            body=raw[:4000],
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise TransportFailure(f"OpenRouter transport failure: {exc}") from exc


def _generation_metadata(
    generation_id: str, api_key: str, timeout_seconds: float
) -> dict[str, Any]:
    url = GENERATION_URL + "?" + urllib.parse.urlencode({"id": generation_id})
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {api_key}"}, method="GET"
    )
    last_error: Exception | None = None
    for delay in (0.0, 1.0, 2.0):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
                data = payload.get("data")
                if not isinstance(data, dict):
                    raise ValueError("generation metadata has no data object")
                return data
        except Exception as exc:  # metadata lookup is bounded and fail-closed
            last_error = exc
    raise RuntimeError(f"generation metadata lookup failed: {last_error}")


def _image_payload(path: Path, expected_sha256: str) -> tuple[str, int]:
    raw = path.read_bytes()
    actual = sha256_bytes(raw)
    if actual != expected_sha256:
        raise RuntimeError(
            f"image SHA-256 mismatch for {path}: expected {expected_sha256}, found {actual}"
        )
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    if not mime.startswith("image/"):
        raise RuntimeError(f"unsupported image MIME type: {mime}")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}", len(raw)


class OpenRouterJudge:
    def __init__(
        self,
        root: Path,
        *,
        transport: Transport = default_transport,
        api_key: str | None = None,
        limit_usd: float = DEFAULT_LIMIT_USD,
        reserve_usd: float = DEFAULT_RESERVE_USD,
        max_unique_requests: int = DEFAULT_MAX_UNIQUE_REQUESTS,
        timeout_seconds: float = 90.0,
        retry_delays: tuple[float, ...] | None = None,
        minimum_request_interval_seconds: float | None = None,
        metadata_lookup: Callable[[str, str, float], dict[str, Any]] = _generation_metadata,
        model_id: str | None = None,
        provider_slugs: tuple[str, ...] | None = None,
        max_token_parameter: str | None = None,
        max_judge_tokens: int | None = None,
        zdr_required: bool | None = None,
        data_collection: str | None = None,
        reasoning_enabled: bool | None = None,
    ):
        self.root = root
        cache_namespace = os.getenv("PATHVLM_OPENROUTER_CACHE_NAMESPACE", "").strip()
        if cache_namespace and not re.fullmatch(r"[A-Za-z0-9_.-]+", cache_namespace):
            raise ValueError("invalid PATHVLM_OPENROUTER_CACHE_NAMESPACE")
        self.cache_dir = (
            root / "cache" / cache_namespace if cache_namespace else root / "cache"
        )
        self.audit_dir = root / "audit"
        self.ledger = BudgetLedger(
            root / "budget_ledger.json",
            limit_usd=limit_usd,
            reserve_usd=reserve_usd,
            max_unique_requests=max_unique_requests,
        )
        self.transport = transport
        # Exposed only so the process-level outage circuit breaker can
        # distinguish a real remote recovery from a local cache hit.  A cache
        # hit is valid reward evidence, but it does not prove that the Judge
        # endpoint is reachable again.
        self.last_result_source: str | None = None
        self.api_key = api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY", "")
        self.timeout_seconds = float(timeout_seconds)
        if retry_delays is None:
            retry_text = os.getenv(
                "PATHVLM_OPENROUTER_RETRY_DELAYS_SECONDS", "2,5"
            )
            retry_delays = tuple(
                float(value.strip())
                for value in retry_text.split(",")
                if value.strip()
            )
        if any(delay < 0 for delay in retry_delays):
            raise ValueError("OpenRouter retry delays cannot be negative")
        self.retry_delays = retry_delays
        if minimum_request_interval_seconds is None:
            minimum_request_interval_seconds = float(
                os.getenv("PATHVLM_OPENROUTER_MIN_REQUEST_INTERVAL_SECONDS", "0")
            )
        self.rate_limiter = InterprocessRateLimiter(
            root / "rate_limit.json", minimum_request_interval_seconds
        )
        self.metadata_lookup = metadata_lookup
        self.model_id = model_id or os.getenv(
            "PATHVLM_OPENROUTER_MODEL_ID", MODEL_ID
        ).strip()
        if not self.model_id:
            raise ValueError("OpenRouter model id must not be empty")
        if provider_slugs is None:
            provider_text = os.getenv(
                "PATHVLM_OPENROUTER_PROVIDER_ONLY", PROVIDER_SLUG
            )
            provider_slugs = tuple(
                value.strip() for value in provider_text.split(",") if value.strip()
            )
        self.provider_slugs = tuple(provider_slugs)
        if not self.provider_slugs:
            raise ValueError("at least one frozen OpenRouter provider is required")
        normalized_providers = {
            normalize_provider(value) for value in self.provider_slugs
        }
        if "" in normalized_providers or len(normalized_providers) != len(
            self.provider_slugs
        ):
            raise ValueError("invalid or duplicate normalized OpenRouter providers")
        self.max_token_parameter = max_token_parameter or os.getenv(
            "PATHVLM_OPENROUTER_MAX_TOKEN_PARAMETER", "max_completion_tokens"
        ).strip()
        if self.max_token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError(
                "PATHVLM_OPENROUTER_MAX_TOKEN_PARAMETER must be max_tokens "
                "or max_completion_tokens"
            )
        self.max_judge_tokens = int(
            max_judge_tokens
            if max_judge_tokens is not None
            else os.getenv("PATHVLM_OPENROUTER_MAX_JUDGE_TOKENS", "320")
        )
        if self.max_judge_tokens <= 0:
            raise ValueError("Judge token ceiling must be positive")
        self.zdr_required = (
            env_bool("PATHVLM_OPENROUTER_ZDR_REQUIRED", True)
            if zdr_required is None
            else bool(zdr_required)
        )
        self.data_collection = (
            os.getenv("PATHVLM_OPENROUTER_DATA_COLLECTION", "deny").strip()
            if data_collection is None
            else data_collection
        )
        if self.data_collection not in {"allow", "deny"}:
            raise ValueError("OpenRouter data_collection must be allow or deny")
        if reasoning_enabled is None:
            self.reasoning_enabled = (
                env_bool("PATHVLM_OPENROUTER_REASONING_ENABLED", False)
                if "PATHVLM_OPENROUTER_REASONING_ENABLED" in os.environ
                else None
            )
        else:
            self.reasoning_enabled = bool(reasoning_enabled)
        if not env_bool("PATHVLM_OPENROUTER_ENFORCE_DISTILLABLE_TEXT", True):
            raise ValueError(
                "Stage3 training requires enforce_distillable_text=true"
            )

    def _request_payload(
        self,
        *,
        image_url: str,
        problem: str,
        solution: str,
        completion: str,
    ) -> dict[str, Any]:
        text = (
            "QUESTION:\n"
            + problem
            + "\n\nREFERENCE ANSWER (context only; do not infer process quality from answer match):\n"
            + solution
            + "\n\nCANDIDATE COMPLETION TO AUDIT:\n"
            + completion
        )
        provider = {
            "only": list(self.provider_slugs),
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": self.data_collection,
            "enforce_distillable_text": True,
        }
        if self.zdr_required:
            provider["zdr"] = True
        payload = {
            "model": self.model_id,
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
            "provider": provider,
            "temperature": 0,
            "seed": 42,
            "stream": False,
        }
        if self.reasoning_enabled is not None:
            payload["reasoning"] = {"enabled": self.reasoning_enabled}
        payload[self.max_token_parameter] = self.max_judge_tokens
        return payload

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / cache_key[:2] / f"{cache_key}.json"

    def _append_audit(self, event: dict[str, Any]) -> None:
        rank = int(os.getenv("RANK", "0"))
        path = self.audit_dir / f"rank_{rank:02d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _parse_response(
        self, response: TransportResponse
    ) -> tuple[dict[str, Any], dict[str, float | int], float, str, str, dict[str, Any]]:
        body = response.body
        if not isinstance(body, dict) or body.get("error"):
            raise RetryableResponseValidationError(
                "OpenRouter success response contains an error"
            )
        if body.get("model") != self.model_id:
            raise ValueError(
                f"served model mismatch: expected {self.model_id}, "
                f"found {body.get('model')}"
            )
        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise RetryableResponseValidationError(
                "OpenRouter response must contain exactly one choice"
            )
        choice = choices[0]
        if choice.get("error") or choice.get("finish_reason") == "error":
            raise RetryableResponseValidationError(
                "OpenRouter choice contains a provider error"
            )
        if choice.get("finish_reason") not in {"stop", None}:
            raise RetryableResponseValidationError(
                f"unexpected finish reason: {choice.get('finish_reason')}"
            )
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("refusal"):
            raise ValueError("Judge response is missing content or contains a refusal")
        content = message.get("content")
        if not isinstance(content, str):
            raise RetryableResponseValidationError("Judge content is not a string")
        try:
            events = validate_events(json.loads(content))
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise RetryableResponseValidationError(
                f"Judge event response failed validation: {exc}"
            ) from exc
        scores = score_events(events)
        usage = body.get("usage")
        if not isinstance(usage, dict) or not isinstance(usage.get("cost"), (int, float)):
            raise ValueError("OpenRouter response is missing numeric usage.cost")
        cost = float(usage["cost"])
        if cost < 0:
            raise ValueError("OpenRouter response has negative usage.cost")
        generation_id = str(
            body.get("id")
            or response.headers.get("x-generation-id")
            or ""
        )
        if not generation_id:
            raise ValueError("OpenRouter response has no generation id")
        provider = str(body.get("provider") or "")
        metadata: dict[str, Any] = {}
        allowed = {normalize_provider(value) for value in self.provider_slugs}
        if normalize_provider(provider) not in allowed:
            metadata = self.metadata_lookup(
                generation_id, self.api_key, self.timeout_seconds
            )
            provider = str(metadata.get("provider_name") or "")
        if normalize_provider(provider) not in allowed:
            raise ValueError(
                f"served provider is not in frozen provider set "
                f"{self.provider_slugs!r}: {provider!r}"
            )
        return events, scores, cost, generation_id, provider, metadata

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
            model_id=self.model_id,
            provider_slugs=self.provider_slugs,
            max_token_parameter=self.max_token_parameter,
            max_judge_tokens=self.max_judge_tokens,
            zdr_required=self.zdr_required,
            data_collection=self.data_collection,
            reasoning_enabled=self.reasoning_enabled,
        )
        cache_path = self._cache_path(cache_key)
        lock_path = cache_path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = lock_path.open("a+")
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            if cache_path.is_file():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("cache_key") != cache_key:
                    raise RuntimeError("cache content-address mismatch")
                events = validate_events(cached.get("events"))
                scores = score_events(events)
                if abs(float(cached.get("scores", {}).get("process", -1)) - scores["process"]) > 1e-12:
                    raise RuntimeError("cached process score mismatch")
                self._append_audit(
                    {
                        "timestamp": now_iso(),
                        "cache_key": cache_key,
                        "cache_hit": True,
                        "process_reward": scores["process"],
                        "image_path": str(image),
                        "image_sha256": image_sha256,
                        "problem_sha256": sha256_bytes(problem.encode("utf-8")),
                        "solution_sha256": sha256_bytes(solution.encode("utf-8")),
                        "completion_sha256": sha256_bytes(completion.encode("utf-8")),
                        "cache_path": str(cache_path),
                        **(source_metadata or {}),
                    }
                )
                self.last_result_source = "cache"
                return float(scores["process"])

            if not self.api_key:
                raise RuntimeError("OPENROUTER_API_KEY is not set")
            image_url, image_bytes = _image_payload(image, image_sha256)
            request_payload = self._request_payload(
                image_url=image_url,
                problem=problem,
                solution=solution,
                completion=completion,
            )
            request_sha256 = sha256_bytes(canonical_json(request_payload).encode("utf-8"))
            attempts: list[dict[str, Any]] = []
            response: TransportResponse | None = None
            parsed: tuple[
                dict[str, Any],
                dict[str, float | int],
                float,
                str,
                str,
                dict[str, Any],
            ] | None = None
            terminal_exception: Exception | None = None
            fallback_exception: JudgeUnavailableForRuleFallback | None = None
            delays = (0.0, *self.retry_delays)
            for attempt_number, delay in enumerate(delays, 1):
                if delay:
                    time.sleep(delay)
                attempt_id = f"{cache_key}:http:{uuid.uuid4().hex}"
                self.ledger.reserve(
                    attempt_id,
                    {
                        "logical_cache_key": cache_key,
                        "attempt_number": attempt_number,
                        "model": self.model_id,
                        "providers": list(self.provider_slugs),
                        "request_sha256": request_sha256,
                    },
                )
                response = None
                try:
                    self.rate_limiter.wait()
                    response = self.transport(
                        request_payload, self.api_key, self.timeout_seconds
                    )
                except TransportFailure as exc:
                    retryable = exc.status is None or exc.status in TRANSIENT_STATUS
                    attempt = {
                        "attempt": attempt_number,
                        "status": exc.status,
                        "error": str(exc),
                        "body": exc.body,
                        "response_headers": audit_response_headers(exc.headers),
                        "retryable": retryable,
                    }
                    attempts.append(attempt)
                    if exc.ambiguous_success_body:
                        self.ledger.commit(
                            attempt_id,
                            self.ledger.reserve_usd,
                            {
                                "logical_cache_key": cache_key,
                                "attempt_number": attempt_number,
                                "generation_id": str(
                                    exc.headers.get("x-generation-id") or ""
                                ),
                                "model": self.model_id,
                                "provider": "",
                                "request_sha256": request_sha256,
                                "status": "ambiguous_success_body_conservatively_billed",
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                        )
                        attempt.update(
                            {
                                "retryable": False,
                                "fallback_eligible": True,
                                "settlement": "conservative_reserve_committed",
                            }
                        )
                        if env_bool(
                            "PATHVLM_STAGE3_RULE_FALLBACK_ENABLED", False
                        ):
                            fallback_exception = JudgeUnavailableForRuleFallback(
                                str(exc)
                            )
                        else:
                            terminal_exception = exc
                        break
                    if retryable:
                        self.ledger.commit(
                            attempt_id,
                            self.ledger.reserve_usd,
                            {
                                "logical_cache_key": cache_key,
                                "attempt_number": attempt_number,
                                "generation_id": "",
                                "model": self.model_id,
                                "provider": "",
                                "request_sha256": request_sha256,
                                "status": "retryable_transport_conservatively_billed",
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                        )
                        attempt["settlement"] = "conservative_reserve_committed"
                        if attempt_number < len(delays):
                            continue
                        if env_bool("PATHVLM_STAGE3_RULE_FALLBACK_ENABLED", False):
                            fallback_exception = JudgeUnavailableForRuleFallback(str(exc))
                        else:
                            terminal_exception = exc
                        break
                    self.ledger.retain_failed_reservation(
                        attempt_id,
                        error=f"{type(exc).__name__}: {exc}",
                        attempts=1,
                    )
                    attempt["settlement"] = "reservation_retained"
                    terminal_exception = exc
                    break

                attempts.append(
                    {
                        "attempt": attempt_number,
                        "status": response.status,
                        "latency_seconds": response.latency_seconds,
                        "generation_id_header": response.headers.get("x-generation-id"),
                    }
                )
                try:
                    parsed = self._parse_response(response)
                except Exception as exc:
                    failure_cost = (
                        response.body.get("usage", {}).get("cost")
                        if isinstance(response.body, dict)
                        and isinstance(response.body.get("usage"), dict)
                        else None
                    )
                    if not isinstance(failure_cost, (int, float)) or failure_cost < 0:
                        self.ledger.retain_failed_reservation(
                            attempt_id,
                            error=f"{type(exc).__name__}: {exc}",
                            attempts=1,
                        )
                        attempts[-1].update(
                            {
                                "validation_error": f"{type(exc).__name__}: {exc}",
                                "retryable": False,
                                "settlement": "reservation_retained",
                            }
                        )
                        terminal_exception = exc
                        break
                    self.ledger.commit(
                        attempt_id,
                        float(failure_cost),
                        {
                            "logical_cache_key": cache_key,
                            "attempt_number": attempt_number,
                            "generation_id": str(response.body.get("id") or ""),
                            "model": self.model_id,
                            "provider": str(response.body.get("provider") or ""),
                            "request_sha256": request_sha256,
                            "status": "billed_response_failed_validation",
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    retryable = isinstance(exc, RetryableResponseValidationError)
                    attempts[-1].update(
                        {
                            "validation_error": f"{type(exc).__name__}: {exc}",
                            "retryable": retryable,
                            "settlement": "reported_cost_committed",
                            "cost_usd": float(failure_cost),
                        }
                    )
                    if retryable and attempt_number < len(delays):
                        continue
                    if retryable and env_bool(
                        "PATHVLM_STAGE3_RULE_FALLBACK_ENABLED", False
                    ):
                        fallback_exception = JudgeUnavailableForRuleFallback(str(exc))
                    else:
                        terminal_exception = exc
                    break

                events, scores, cost, generation_id, provider, generation_metadata = parsed
                self.ledger.commit(
                    attempt_id,
                    cost,
                    {
                        "logical_cache_key": cache_key,
                        "attempt_number": attempt_number,
                        "generation_id": generation_id,
                        "model": self.model_id,
                        "provider": provider,
                        "request_sha256": request_sha256,
                        "status": "completed",
                    },
                )
                attempts[-1].update(
                    {
                        "retryable": False,
                        "settlement": "reported_cost_committed",
                        "cost_usd": cost,
                    }
                )
                break

            if parsed is None:
                failure_status = (
                    "judge_unavailable_after_bounded_attempts"
                    if fallback_exception is not None
                    else "terminal_judge_failure"
                )
                self._append_audit(
                    {
                        "timestamp": now_iso(),
                        "cache_key": cache_key,
                        "cache_hit": False,
                        "status": failure_status,
                        "error": f"{type(fallback_exception or terminal_exception).__name__}: "
                        f"{fallback_exception or terminal_exception}",
                        "attempts": attempts,
                        "request_sha256": request_sha256,
                        **(source_metadata or {}),
                    }
                )
                if fallback_exception is not None:
                    raise fallback_exception
                if terminal_exception is None:
                    raise RuntimeError("Judge attempts ended without a classified result")
                raise terminal_exception
            cache_record = {
                "schema_version": 1,
                "created_at": now_iso(),
                "cache_key": cache_key,
                "request_sha256": request_sha256,
                "request": request_payload,
                "response": response.body,
                "response_headers": response.headers,
                "attempts": attempts,
                "generation_id": generation_id,
                "generation_metadata": generation_metadata,
                "served_model": response.body.get("model"),
                "served_provider": provider,
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
            atomic_json(cache_path, cache_record)
            self._append_audit(
                {
                    "timestamp": now_iso(),
                    "cache_key": cache_key,
                    "cache_hit": False,
                    "status": "completed",
                    "process_reward": scores["process"],
                    "events": events,
                    "scores": scores,
                    "cost_usd": cost,
                    "generation_id": generation_id,
                    "served_model": response.body.get("model"),
                    "served_provider": provider,
                    "usage": response.body.get("usage"),
                    "attempts": attempts,
                    "request_sha256": request_sha256,
                    "cache_path": str(cache_path),
                    "image_path": str(image),
                    "image_sha256": image_sha256,
                    "problem_sha256": sha256_bytes(problem.encode("utf-8")),
                    "solution_sha256": sha256_bytes(solution.encode("utf-8")),
                    "completion_sha256": sha256_bytes(completion.encode("utf-8")),
                    **(source_metadata or {}),
                }
            )
            self.last_result_source = "remote"
            return float(scores["process"])
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()


def _aligned(values: Any, count: int, name: str) -> list[Any]:
    if values is None:
        raise RuntimeError(f"Stage3 process reward is missing {name}")
    result = list(values)[:count]
    if len(result) != count:
        raise RuntimeError(f"Stage3 process reward length mismatch for {name}")
    return result


def structural_rule_fallback(completion: str) -> tuple[float, dict[str, bool]]:
    """A target-blind, capped structural reward used only during brief judge outages."""

    text = completion_text(completion)
    think_match = re.search(r"<think\b[^>]*>(.*?)(?:</think\s*>|$)", text, re.I | re.S)
    answer_match = re.search(r"<answer\b[^>]*>(.*?)(?:</answer\s*>|$)", text, re.I | re.S)
    reasoning = think_match.group(1).strip() if think_match else ""
    lower = reasoning.lower()
    features = {
        "nonempty_reasoning_and_answer": bool(reasoning and answer_match and answer_match.group(1).strip()),
        "image_grounding_language": bool(
            re.search(
                r"\b(image|slide|section|field|cells?|nuclei|nuclear|cytoplasm|stroma|"
                r"gland|architecture|lesion|tumou?r|tissue|histolog|morpholog|stain)\w*\b",
                lower,
            )
        ),
        "comparison_or_elimination": bool(
            re.search(
                r"\b(option|choice|eliminat|exclude|unlikely|less likely|rather than|"
                r"whereas|in contrast|does not fit|not consistent)\w*\b",
                lower,
            )
        ),
        "medical_support_language": bool(
            re.search(
                r"\b(because|therefore|consistent with|suggest|indicat|characteristic|"
                r"diagnos|criterion|feature|support)\w*\b",
                lower,
            )
        ),
    }
    reward = (
        0.15 * features["nonempty_reasoning_and_answer"]
        + 0.15 * features["image_grounding_language"]
        + 0.10 * features["comparison_or_elimination"]
        + 0.10 * features["medical_support_language"]
    )
    return reward, features


class RuleFallbackLimiter:
    """Inter-process total and consecutive caps for structural fallback rewards."""

    def __init__(self, path: Path, *, total_limit: int, consecutive_limit: int):
        if total_limit <= 0 or consecutive_limit <= 0 or consecutive_limit > total_limit:
            raise ValueError("invalid Stage3 rule fallback limits")
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")
        self.total_limit = total_limit
        self.consecutive_limit = consecutive_limit

    def _default(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "total_limit": self.total_limit,
            "consecutive_limit": self.consecutive_limit,
            "total_used": 0,
            "consecutive_used": 0,
            "events": [],
            "updated_at": now_iso(),
        }

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._default()
        state = json.loads(self.path.read_text(encoding="utf-8"))
        expected = {
            "total_limit": self.total_limit,
            "consecutive_limit": self.consecutive_limit,
        }
        mismatch = {
            key: {"expected": value, "actual": state.get(key)}
            for key, value in expected.items()
            if state.get(key) != value
        }
        if mismatch:
            raise RuleFallbackLimitExceeded(f"rule fallback contract mismatch: {mismatch}")
        return state

    def _lock(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return handle

    def record_fallback(self, metadata: dict[str, Any]) -> dict[str, Any]:
        handle = self._lock()
        try:
            state = self._read()
            next_total = int(state["total_used"]) + 1
            next_consecutive = int(state["consecutive_used"]) + 1
            if next_total > self.total_limit:
                raise RuleFallbackLimitExceeded(
                    f"rule fallback total limit reached: {self.total_limit}"
                )
            if next_consecutive > self.consecutive_limit:
                raise RuleFallbackLimitExceeded(
                    f"rule fallback consecutive limit reached: {self.consecutive_limit}"
                )
            state["total_used"] = next_total
            state["consecutive_used"] = next_consecutive
            state["events"].append(
                {"timestamp": now_iso(), "event": "fallback", **metadata}
            )
            state["updated_at"] = now_iso()
            atomic_json(self.path, state)
            return state
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def record_judge_success(self) -> None:
        handle = self._lock()
        try:
            state = self._read()
            if int(state["consecutive_used"]) == 0:
                return
            state["events"].append(
                {
                    "timestamp": now_iso(),
                    "event": "judge_success_resets_consecutive_failures",
                    "prior_consecutive_used": int(state["consecutive_used"]),
                }
            )
            state["consecutive_used"] = 0
            state["updated_at"] = now_iso()
            atomic_json(self.path, state)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


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
    judge = OpenRouterJudge(
        Path(root_text),
        limit_usd=float(os.getenv("PATHVLM_OPENROUTER_LIMIT_USD", "15")),
        reserve_usd=float(os.getenv("PATHVLM_OPENROUTER_RESERVE_USD", "0.05")),
        max_unique_requests=int(
            os.getenv("PATHVLM_OPENROUTER_MAX_UNIQUE_REQUESTS", "401")
        ),
    )
    fallback_enabled = env_bool("PATHVLM_STAGE3_RULE_FALLBACK_ENABLED", False)
    fallback_limiter = (
        RuleFallbackLimiter(
            Path(root_text) / "rule_fallback_ledger.json",
            total_limit=int(os.getenv("PATHVLM_STAGE3_RULE_FALLBACK_TOTAL_LIMIT", "24")),
            consecutive_limit=int(
                os.getenv("PATHVLM_STAGE3_RULE_FALLBACK_CONSECUTIVE_LIMIT", "4")
            ),
        )
        if fallback_enabled
        else None
    )
    rewards = []
    for index in range(count):
        completion = completion_text(completions[index])
        source_metadata = {
            "record_index": int(record_indices[index]),
            "rank": int(os.getenv("RANK", "0")),
            "local_rank": int(os.getenv("LOCAL_RANK", "0")),
            "training_segment": os.getenv(
                "PATHVLM_TRAINING_SEGMENT", "unspecified"
            ),
        }
        try:
            reward = judge.judge_one(
                image_path=str(image_paths[index]),
                image_sha256=str(image_hashes[index]),
                problem=str(problems[index]),
                solution=str(solutions[index]),
                completion=completion,
                source_metadata=source_metadata,
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
            state = fallback_limiter.record_fallback(
                {
                    **source_metadata,
                    "reward": reward,
                    "features": features,
                    "judge_error": str(exc),
                    "completion_sha256": sha256_bytes(completion.encode("utf-8")),
                }
            )
            judge._append_audit(
                {
                    "timestamp": now_iso(),
                    "status": "bounded_structural_rule_fallback",
                    "process_reward": reward,
                    "features": features,
                    "fallback_total_used": state["total_used"],
                    "fallback_consecutive_used": state["consecutive_used"],
                    **source_metadata,
                }
            )
        rewards.append(reward)
    return rewards
