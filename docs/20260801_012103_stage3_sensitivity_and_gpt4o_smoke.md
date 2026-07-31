# Stage 3 penalty sensitivity and GPT-4o AIGCBest smoke

Timestamp: 2026-08-01 01:21:03 +0800

## Scope and authorization boundary

This execution implements only the first paid-cost gate in the approved future plan:

1. freeze and publish the approved plan;
2. rescore existing six-event Judge caches offline at penalties 0.2--0.6;
3. make exactly two non-training GPT-4o AIGCBest smoke requests;
4. measure cost and stop before any 100-step or formal Stage 3 training.

No training, checkpoint creation, validation selection, test inference, or formal result was run in
this step.  No additional paid request is authorized by this record.

## Reproducible offline sensitivity

The new script `scripts/analyze_stage3_penalty_sensitivity.py` reads cached six-event judgments,
excludes records marked `source.smoke=true`, recomputes the deterministic process reward and makes
zero network calls.  Its unit test covers clipping, tied ranks, smoke exclusion and group-level
agreement.  The immutable full output is stored outside Git because it is a generated report:

- path: `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/reports/stage3_penalty_sensitivity_20260801/offline_sensitivity_20260801_011622.json`
- SHA-256: `a13db8cb2b1663f6e1a22769c8c049ee8d440f9a06563ca9209287db61bda3c7`
- non-smoke judgments: 499 pilot + 2,197 attempt 01 + 88 attempt 02 = 2,784

The table reports mean process reward, population standard deviation, fraction clipped to zero,
sample-level Spearman correlation with 0.4, and within-generation-group centered-advantage sign
agreement with 0.4.

| Cache segment | penalty | mean | SD | zero | Spearman vs 0.4 | group sign agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pilot50, n=499 | 0.2 | 0.8108 | 0.1728 | 0.00% | 0.9962 | 95.16% |
|  | 0.3 | 0.7162 | 0.2592 | 0.00% | 0.9962 | 95.16% |
|  | 0.4 | 0.6481 | 0.3145 | 0.40% | 1.0000 | 100.00% |
|  | 0.5 | 0.5932 | 0.3589 | 12.02% | 0.9935 | 95.77% |
|  | 0.6 | 0.5701 | 0.3678 | 12.02% | 0.9910 | 94.15% |
| Kimi attempt 01, n=2,197 | 0.2 | 0.8080 | 0.1745 | 0.00% | 0.9945 | 93.86% |
|  | 0.3 | 0.7120 | 0.2617 | 0.00% | 0.9945 | 93.86% |
|  | 0.4 | 0.6470 | 0.3144 | 1.09% | 1.0000 | 100.00% |
|  | 0.5 | 0.5975 | 0.3550 | 10.51% | 0.9923 | 91.94% |
|  | 0.6 | 0.5771 | 0.3649 | 10.51% | 0.9891 | 91.12% |
| Kimi attempt 02, n=88 | 0.2 | 0.8330 | 0.1697 | 0.00% | 0.9971 | 97.73% |
|  | 0.3 | 0.7494 | 0.2546 | 0.00% | 0.9971 | 97.73% |
|  | 0.4 | 0.6886 | 0.3102 | 0.00% | 1.0000 | 100.00% |
|  | 0.5 | 0.6392 | 0.3553 | 10.23% | 0.9952 | 96.59% |
|  | 0.6 | 0.6159 | 0.3668 | 10.23% | 0.9903 | 95.45% |

Interpretation is deliberately limited.  The overall ordering is highly correlated across the
tested coefficients, but clipping and group-level advantages are not identical.  In particular,
0.5/0.6 collapse roughly 10--12% of cached judgments to zero, while 0.2/0.3 never reach zero in
these caches.  This diagnostic therefore supports the predeclared online comparison of
0.3/0.4/0.5; it does not establish a winning coefficient and cannot substitute for matched
training pilots.

## GPT-4o AIGCBest feasibility

The authenticated model catalog and public price table were checked before the paid calls.  The
exact model ID `gpt-4o-2024-08-06` was present, with the expected default rates of $2.50 per million
input tokens and $10 per million output tokens.  The new bounded client:

- uses the same frozen pathology auditor prompt and strict six-event JSON schema as Stage 3;
- accepts a vision input and requires the served model to equal the requested version exactly;
- sets temperature 0, seed 42 and maximum output 320 tokens;
- makes at most one paid request per invocation, has no retry path and refuses overwrite;
- stores request provenance, full response, event parsing, tokens, price snapshot and cost; and
- marks both calls as non-training and non-formal.

Both authorized calls passed:

| Smoke | Input | Output | Process reward at 0.4 | Estimated cost | Result SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| synthetic blue/red control | 870 | 115 | 0.8 | $0.0033250 | `c8c264f34b4e1b46bf50b557b43963fb89c56a6a0c577fd9e1abbacfcb97dbe4` |
| PathMMU validation index 0 with existing pilot completion | 1,465 | 137 | 1.0 | $0.0050325 | `d6d1d148b712044be874e950a462a0d87ce6a30e15ca541addbebd175adfc148` |

Total observed smoke cost was **$0.0083575**, with zero retries.  The account usage endpoint changed
its quota-style counters in a way that does not behave like a reliable dollar ledger; request
token usage and the frozen public rate are therefore the auditable cost basis.

Full records remain at:

- `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/reports/aigcbest_gpt4o_stage3_smoke_20260801/synthetic_result.json`
- `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/reports/aigcbest_gpt4o_stage3_smoke_20260801/pathmmu_validation_result.json`

## Cost gate for the next step

A single validation smoke is too small to be the only budget estimate.  Applying the verified
rates to the existing Kimi cache mean (1,437.674 input and 226.185 output tokens) gives
$0.005856035 per call:

| Scope | Calls | Point estimate | 20% planning reserve |
| --- | ---: | ---: | ---: |
| one 100-step pilot | 800 | $4.6848 | $5.6218 |
| three matched 100-step pilots | 2,400 | $14.0545 | $16.8654 |
| later formal 1,500-step run | 12,000 | $70.2724 | $84.3269 |

The next requested authorization is therefore a **separate $18 total cap** for exactly three
matched 100-step GPT-4o pilots (penalties 0.3/0.4/0.5), with no more than 800 unique Judge calls and
$6 per pilot.  This request does not include a formal 1,500-step run.  Training must remain stopped
until that new budget and the final launch preflight are explicitly approved.

## Verification

Executed successfully:

```text
PYTHONPATH=scripts python3 -m unittest \
  scripts/test_analyze_stage3_penalty_sensitivity.py \
  scripts/test_run_aigcbest_stage3_gpt4o_smoke.py

Ran 6 tests ... OK
```
