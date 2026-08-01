# Stage3 fault-tolerance hardening and coefficient interpretation

Recorded at: `2026-08-01T18:14:14+08:00`

## Scope and current authorization

The user cancelled the proposed 300-step coefficient extension and requested stronger failure
handling.  This change made no network request, paid API call, model inference, training run or
test access.  No complete 1,500-step Stage3 arm is authorized by this record.

## What the 100-step coefficient pilots establish

The matched GPT-4o pilots scored 234/385 (60.78%), 237/385 (61.56%) and 240/385 (62.34%) for
coefficients 0.3, 0.4 and 0.5.  All paired confidence intervals cross zero and all exact two-sided
McNemar p-values exceed 0.30.  The result therefore supports local robustness over 0.3--0.5; it
does not establish that 0.4, or any tested coefficient, is statistically optimal.

The defensible use of 0.4 is consequently not a post-hoc performance selection.  It is a fixed
historical/mechanistic setting whose sensitivity has been checked.  For either three-event
subscale, the deterministic reward levels are:

| error/missing-event count | c=0.3 | c=0.4 | c=0.5 |
| ---: | ---: | ---: | ---: |
| 0 | 1.0 | 1.0 | 1.0 |
| 1 | 0.7 | 0.6 | 0.5 |
| 2 | 0.4 | 0.2 | 0.0 |
| 3 | 0.1 | 0.0 | 0.0 |

Among the three tested candidates, 0.4 preserves four ordered severity levels while reaching zero
at the maximum count.  Coefficient 0.3 never reaches the floor even when all three components fail;
0.5 collapses two and three failures to the same value.  This reward-geometry argument, the prior
0.4 contract and the nonsignificant sensitivity result can justify fixing 0.4 without claiming it
was empirically best.  The user has not yet accepted this revised justification, so the formal
coefficient remains explicitly unfrozen in this record.

## Fault-tolerance correction

The prior supervisor restarted after every nonzero distributed-launch exit.  Since torchrun often
maps unrelated root causes to exit status 1, that behavior could restart OOM, budget exhaustion,
model/provider mismatch, source drift, user interrupt or an unknown new failure without diagnosis.

The corrected implementation is fail-closed and uses a recovery whitelist:

- only a recognized transient Judge transport failure or the consecutive Judge-outage circuit
  breaker may enter automatic checkpoint recovery;
- user interrupt, budget/request limit, model/provider/schema mismatch, source/contract drift,
  OOM/disk/numeric failure, total fallback exhaustion and every unknown failure stop without an
  automatic relaunch;
- failure classification, the matched pattern and the SHA-256 of the inspected log tail are added
  to the supervisor audit;
- unresolved budget reservations are conservatively settled before either recovery or terminal
  stop;
- automatic recoveries are hard-limited to 0--3, matching the approved protocol;
- a cache hit no longer resets the consecutive remote-outage counter.  Only a newly completed
  remote Judge response proves endpoint recovery;
- the fallback ledger remains inter-process locked and was exercised concurrently in a new test.

Historical-log replay classified Kimi attempt 01's `http.client.IncompleteRead` as recoverable and
attempt 02's `KeyboardInterrupt` as terminal.  This directly reproduces the intended distinction.

## Verification

- 41 CUDA-free Stage3 tests passed, including 36 Judge/recovery-specific tests.
- Both Kimi launcher and supervisor pass `bash -n`.
- `git diff --check` passes.
- No current Stage3 training process exists and neither historical Kimi attempt contains a usable
  formal checkpoint.

Exact code hashes are recorded in
`protocol/stage3_fault_tolerance_hardening_20260801_181414.json`.
