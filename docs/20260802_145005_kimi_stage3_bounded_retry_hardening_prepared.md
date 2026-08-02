# Kimi Stage3 bounded-retry hardening prepared

Timestamp: `2026-08-02 14:50:05 CST`

## Outcome

The fresh Kimi 2.6 Stage3 arm remains **unlaunched** and is still ordered after GPT-4o Stage3
validation and the runnable GPU-local baselines.  Its OpenRouter judge path now implements the
user-approved recovery contract: retry an ambiguous transport, truncation, malformed JSON or
retryable event-schema failure at most three times after the first request, then enter the
existing structural-rule fallback only through the unchanged total-24/consecutive-4 circuit
breaker.

Every physical attempt has a unique ledger identity.  A retryable response with reported usage is
committed at the reported cost; an ambiguous/transient transport is conservatively committed at
the configured reserve.  Exact model/provider identity mismatches, refusals, missing accounting
and non-transient failures remain terminal and are not retried.  Successful cache records retain
the complete physical-attempt history.

The fresh arm allows 12,000 logical training judgments and 360 retry attempts.  Because its paid
synthetic contract smoke shares the same ledger, the ledger ceiling is 12,361 physical requests.
The old two Kimi attempts remain immutable failures with combined audited cost `$3.28217016`; the
fresh arm's retained `$30` authorization therefore has `$26.71782984` remaining.

## Verification

- Python compilation and Bash syntax checks passed.
- `scripts.test_stage3_openrouter_judge` and `scripts.test_stage3_checkpoint_recovery`: 38 tests
  passed.
- Six additional Stage3 contract/sensitivity/selection suites: 26 tests passed.
- New regressions prove bounded ambiguous retries, conservative settlement, malformed-response
  retry followed by success, and terminal no-retry behavior for model-identity mismatch.
- No network request, paid request, training, inference, validation or test access occurred.

This is engineering preparation (`formal_result: false`), not a Kimi efficacy result.

