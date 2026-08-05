# Kimi balance continuation and HTTP 429 recovery verified

Recorded at: `2026-08-05 19:53:00 CST`

## Approved continuation contract

The user accepted the 18 rule fallbacks already present in the audit ledger and approved
continuing from the latest complete checkpoint with up to the remaining USD 20 OpenRouter
balance. Before the first continuation launch, immutable copies of the budget and fallback
ledgers were retained. The runtime contract was amended as follows:

- budget ceiling: `26.71782984` to `46.70863776` USD, exactly the settled
  `26.70863776` USD spend plus the newly approved `20` USD;
- physical-request ceiling: `12976` to `13500`;
- total rule-fallback ceiling: `24` to `36`;
- consecutive rule-fallback ceiling initially remained `4`.

The paid smoke marker was not rewritten. It remains evidence of the original contract, while the
runtime ledgers carry explicit timestamped amendments.

## Segment04 result and HTTP 429 diagnosis

The structurally validated `checkpoint-1100` contained four model shards, eight optimizer
states, eight model states and eight RNG states. Segment04 restored all optimizer and RNG states
and completed step 1101 with loss `0.0011`, reward `2.3499999046325684` and KL
`0.027099609375`. It then stopped during step 1102 after four consecutive rule fallbacks caused
by bounded OpenRouter HTTP 429 failures. The supervisor correctly classified this as a
recoverable Judge outage and settled all reservations. No newer checkpoint was created, so the
next valid resume remained `checkpoint-1100`.

A read-only OpenRouter credit observation showed total credits of `36` USD and actual total usage
of `15.432634495` USD. The account therefore was not exhausted; the failure was provider-side
rate limiting or temporary capacity at the frozen Inceptron endpoint.

Because eight distributed ranks form one synchronized request wave, a consecutive limit of four
could stop halfway through one wave. The fallback ledger was backed up and its consecutive limit
was atomically raised from `4` to `8`, while preserving `total_used=23`,
`consecutive_used=4`, the total limit of `36`, and all prior events. The global request-start
interval was also changed from 4 seconds to 8 seconds.

## Segment05 pre-transport failure and correction

Segment05 passed the original static preflight and restored the checkpoint, but the first Judge
call found the existing shared `rate_limit.json` still frozen at 4 seconds. All ranks failed
locally with `OpenRouter rate-limit contract mismatch: expected 8.0, found 4.0` before any network
transport. There was no OpenRouter request, optimizer update or model-state change. The
supervisor conservatively settled eight already-created reservations at `0.05` USD each, so the
internal budget ledger includes `0.40` USD that was not observed as an actual OpenRouter charge.

The correction adds an atomic rate-limit migration command, validates the shared rate-limit
contract during static preflight before GPU loading, and performs local rate limiting before
creating a budget reservation. The 4-second rate state was backed up and atomically migrated to
8 seconds without changing its last-request timestamp.

## Segment06 verified resume

Segment06 launched from the same validated `checkpoint-1100` under shared-GPU admission:

- supervisor PID: `771517`;
- launcher PID: `771525`;
- torch distributed PID: `771542`;
- worker PIDs: `771620` through `771627`;
- master port: `29647`;
- supervisor log: `supervisor_resume_segment06_20260805_194600.log`;
- training log: `train_segment06.log`.

All optimizer and RNG states loaded successfully. The first resumed Judge wave contained five
successful remote Inceptron results and three bounded structural fallbacks after HTTP 429. The
first remote result succeeded on its first attempt, proving the 8-second rate contract was live;
that success reset the retained consecutive count from four to zero. Step 1101 then completed:

- loss: `0.0011`;
- gradient norm: `4.756123065948486`;
- reward: `2.0875000953674316`;
- process reward: `0.46250003576278687`;
- KL: `0.027099609375`;
- epoch: `2.2`.

Immediately after that step, the fallback ledger contained 26 total uses and three consecutive
uses, both below their 36/8 limits. The mutable budget observation contained 10,030 completed
physical attempts and `30.323539110000233` USD conservatively committed, with the next step's
first reservation already in flight. These are runtime observations, not final settlement.

The independent AFI-VLA processes were not stopped or modified. Under the fully resident shared
load, the most occupied GPUs retained about 14.3 GiB each. No OOM or process collision was
observed.

## Immutable evidence

- pre-balance-amendment budget ledger SHA-256:
  `711fe2d4fe69461629ff6dcdbf041f872cb6c8cd13ce53eff310ef7392e639ae`;
- pre-total-cap-amendment fallback ledger SHA-256:
  `4daf729f1826a01e13f9eafa035fe17c20703ab86bdf8ddf72e8a89c16f8463b`;
- pre-consecutive-amendment fallback ledger SHA-256:
  `dff74cbe88be958d36a798df6718b257839112369f982ae2503f5bd5410b1d94`;
- pre-interval-amendment rate-limit state SHA-256:
  `d2d8c59e05b08a41c816557e34bff72de8455f4be7e1e0daa66246a235ca480d`.

## Terminal status

Segment06 subsequently completed steps 1102 through 1105. During step 1106, continued HTTP 429
failures exhausted the approved total fallback limit. Rank 5 raised
`rule fallback total limit reached: 36`, and the supervisor stopped fail-closed at
`2026-08-05 20:13:38 CST`. The final settled runtime state was:

- latest completed optimizer step: `1105`;
- attempted but uncommitted step: `1106`;
- fallback total: `36/36`;
- fallback consecutive count: `2/8`;
- completed physical attempts: `10136/13500`;
- conservatively committed budget: `34.41095351000011/46.70863776` USD;
- unresolved reservations: `0`;
- latest complete checkpoint: `checkpoint-1100`.

The run is stopped and is not complete. Because checkpoints are saved every 100 steps, steps
1101 through 1105 are metrics-only evidence and are not recoverable model state. Another restart
would recompute from step 1101. Increasing the total fallback cap or changing the provider is a
scientific contract decision requiring new approval; neither was performed by this record.

Code commit at the verified launch: `cb88f6544ddf37af0d801f8fdb6d844dd10c8b91`.
