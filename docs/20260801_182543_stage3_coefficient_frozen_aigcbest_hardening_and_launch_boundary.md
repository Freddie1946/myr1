# Stage3 coefficient freeze, AIGCBest hardening and launch boundary

Recorded at: `2026-08-01T18:25:43+08:00`

## Corrected interpretation of the 100-step pilots

The user correctly challenged the earlier shorthand that the 100-step experiment demonstrated
coefficient insensitivity.  The point estimates are monotonic: 60.78%, 61.56% and 62.34% at 0.3,
0.4 and 0.5.  The 0.3-to-0.5 paired difference is +1.56 percentage points, with a 95% paired
bootstrap interval of [-0.78, 4.16] points and exact two-sided McNemar p=0.307.  This short,
single-seed experiment has limited power.  It did not detect a statistically distinguishable
difference, but it neither proves insensitivity nor rules out a beneficial monotonic trend.

The user confirmed that the original 0.4 design was based on the three-event subscale geometry.
For one subscale, 0.4 yields reward levels 1.0, 0.6, 0.2 and 0.0 for zero through three missing/error
events.  It therefore preserves all four ordered severity levels and reaches zero when all three
components fail.  In contrast, 0.3 leaves a positive 0.1 reward at complete failure, while 0.5
collapses two and three failures to the same zero floor.

The formal coefficient is now frozen at 0.4 as an a-priori mechanistic/historical setting, not as
the statistically best pilot arm.  The 0.3/0.5 runs are reported as bounded sensitivity evidence.
The user cancelled the proposed 300-step extension.

## GPT-4o/AIGCBest fault-tolerance correction

The AIGCBest training client inherited two pilot-only behaviors unsuitable for a formal long run:
it could retry a connection failure with ambiguous billing state, and it could retry a billed
model/schema validation failure.  It also treated a cache hit as evidence that the remote endpoint
had recovered.

The client now:

- retries only explicit transient HTTP 408/429/500/502/503/529 responses;
- never resends a connection-level or HTTP-200 invalid/truncated response in the same process;
- accepts an incomplete chunked body only when the received bytes form complete valid JSON;
- stops immediately after a billed model/schema/identity validation failure;
- resets the consecutive-outage counter only after a fresh remote success, never after a cache hit;
- conservatively records every physical attempt and preserves the bounded rule fallback.

Forty-four CUDA-free Stage3 tests pass, including new AIGCBest tests for ambiguous connection
failure, explicit transient retry, terminal model mismatch and remote-versus-cache source state.

## Launch and baseline boundary

- All eight A100 80GB GPUs are idle; approximately 985 GiB host memory and 5.1 TiB filesystem space
  are available.
- The user declined a proposed 4+4-GPU concurrency smoke.  Formal Stage3 therefore retains the
  existing eight-GPU launcher; local GPU baselines must not contend with it and will be queued.
- Kimi attempts 01/02 consumed a conservatively accounted USD 3.28217016 of the previously approved
  USD 30 aggregate ceiling.  The remaining ceiling is USD 26.71782984, with zero unresolved
  reservations.  A fresh run/cache namespace is required because neither attempt has a checkpoint.
- The complete GPT-4o arm remains outside the old USD 18 pilot authorization.  At the measured
  0.4-pilot rate, 12,000 logical calls project approximately USD 61.96 before retry contingency;
  its separate formal ceiling remains pending.
- PathMMU test999 is complete for every currently runnable local baseline.  PathVQA/OmniMedVQA have
  adapter smokes and rescored prior outputs but not a complete uniform rerun.  The user reauthorized
  baseline testing; local GPU work will be queued after Stage3, while hosted API baselines still
  require explicit per-run paid ceilings.

No paid call, model inference, training launch or new test access occurred in this correction.

