# Stage3 and PathVQA bounded-retry recovery

Recorded at: `2026-08-02T13:49:47+08:00`

## Failure being corrected

The formal GPT-4o Stage3 arm reached optimizer step 820 and stopped after one billed Judge response
returned `finish_reason=length` at the frozen 320-token first-attempt cap.  The prior client treated
every billed response-validation failure as terminal, so it did not retry that transient truncation.
The complete eight-rank `checkpoint-800` was independently validated and is the recovery point;
steps 801--820 are not claimed as saved training progress.

The two hosted PathVQA prediction files are complete with small recorded failure sets, but their
GPT-5-mini semantic-judge continuations also stopped: one run attempted to read `completion` from a
failed prediction row, and one stopped after an otherwise valid verdict used an out-of-contract
`error_type`.  Failed prediction rows must not be invented or silently scored.

## Newly authorized bounded behavior

The user explicitly requested retry behavior and continuation.  The AIGCBest Stage3 client now:

- permits at most four physical attempts under the existing global physical-attempt ledger;
- retries ambiguous connection/timeout and HTTP-200 invalid-body failures, conservatively charging
  every ambiguous attempt at the pre-existing reserve amount;
- retries response truncation and event-schema/JSON validation failures;
- keeps 320 tokens for the first response, then raises only response-validation retries to 512 and
  768 tokens (the final bounded attempt remains at 768);
- preserves each failed response, error, token cap, latency and billed amount in the audit trail;
- emits `JudgeUnavailableForRuleFallback` only after bounded retry exhaustion, so the existing
  total-24/consecutive-4 structural fallback limiter remains the final guard;
- continues to stop immediately on served-model mismatch, refusal, non-transient HTTP failure,
  missing response identity or other proven contract drift.

The PathVQA semantic Judge now retries network, JSON, finish-reason and strict verdict-schema
failures up to three physical attempts.  It records exhausted failures and continues to the next
case.  A failed hosted-model prediction is explicitly skipped and recorded in a deduplicated audit
file; it is never passed to the semantic Judge and never counted as semantically correct.
Served-model mismatch remains terminal.

## Verification and continuation boundary

- 51 CUDA-free Stage3/recovery/contract tests passed.
- 11 PathVQA/hosted-evaluation tests passed.
- Python compilation, shell syntax and `git diff --check` passed.
- No paid request, model inference or GPU training occurred while applying this correction.
- After a clean commit, resume GPT-4o Stage3 from the validated `checkpoint-800` in a fresh cache
  segment.  Resume the existing Qwen-VL-Plus and Claude Haiku 4.5 PathVQA GPT-5-mini semantic Judge
  outputs rather than discarding their already completed judgments.

This changes the earlier 2026-08-01 fail-closed transport policy by explicit later user approval.
It does not change the Stage3 parent, seed, penalty 0.4, reward events, rule fallback limits,
maximum logical judgments, checkpoint cadence or model identity.

