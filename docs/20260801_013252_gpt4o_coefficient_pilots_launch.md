# GPT-4o Stage 3 coefficient pilots: paid launch authorization

Timestamp: 2026-08-01 01:32:52 +0800

## Authorization

The user explicitly approved the paid gate proposed in
`20260801_012103_stage3_sensitivity_and_gpt4o_smoke.md`:

- three matched 100-step Stage 3 pilots at penalties 0.3, 0.4 and 0.5;
- no more than $6 per pilot and $18 total;
- no more than 800 actual outbound Judge HTTP attempts per pilot;
- no authorization for a 1,500-step formal run or new test inference.

The runs execute sequentially in coefficient order 0.3, 0.4, 0.5.  Every run starts independently
from the same immutable Stage2 seed-42 checkpoint.  No coefficient arm may resume from or reuse the
Judge cache of another arm.

## Fixed training contract

| Field | Value |
| --- | --- |
| Parent | `outcome_grpo_n1000_seed42_epoch02_step1000` |
| Parent manifest SHA-256 | `83df2570a33bd760bebb6ef8afca175b71c33bb585e107cdb2f3889edebc04e0` |
| Training records | frozen RL1000 |
| Judge gateway | AIGCBest |
| Judge model | exact `gpt-4o-2024-08-06` |
| Coefficients | 0.3, 0.4, 0.5 |
| Steps per arm | 100 |
| Judge completions per step | 8 globally |
| Maximum actual HTTP attempts | 800 per arm, including retries |
| USD cap | $6 per arm, $18 total |
| Maximum Judge output | 320 tokens |
| Retry delays | 15, 45, 90 seconds |
| Rule fallback | max reward 0.5, max 24 total and 4 consecutive per arm |
| Learning rate | 1e-6 |
| Beta | 0.04 |
| Generations per prompt | 4 |
| Model completion maximum | 192 tokens |
| Save interval | step 100 |
| Save limit | 2 |
| Seed/data seed | 42/42 |
| Vision tower | frozen |
| Automatic restart after failure | none |
| Selection | PathMMU validation385 only |
| Test999 | not accessed and never used for selection |

The request-count interpretation is deliberately strict.  Each physical outbound attempt reserves
one entry in the 800-attempt ledger.  A retry is not hidden behind one logical cache key.  Successful
responses are billed from returned input/output tokens at the preflighted public rates; ambiguous
failures are conservatively charged the full $0.02 reservation.  Eight concurrent in-flight calls
are also covered by reservations before they start.  A budget or attempt-cap refusal terminates the
run rather than silently relaxing the contract.

## New implementation

- `scripts/stage3_aigcbest_judge.py` implements exact-model structured vision judging, a
  coefficient-dependent deterministic reward, content-addressed per-segment cache, response and
  event audit, actual-attempt accounting, cost accounting, bounded retry, and bounded structural
  fallback.
- `scripts/grpo_pathmmu.py` selects that implementation only when the launch environment explicitly
  sets `PATHVLM_STAGE3_JUDGE_BACKEND=aigcbest`; historical OpenRouter behavior remains the default.
- `scripts/launch_stage3_gpt4o_penalty_pilot.sh` accepts only 0.3, 0.4 or 0.5.  It verifies every
  parent manifest file, every frozen RL1000 image, the GPT-4o validation smoke, current authenticated
  catalog identity, current public price ratio, clean Git state, idle eight-GPU state, disk space and
  master port before creating a run directory.

The launch directories are fixed as:

- `/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/stage3_process_grpo/gpt4o_penalty_0p3_100step_seed42_20260801`
- `/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/stage3_process_grpo/gpt4o_penalty_0p4_100step_seed42_20260801`
- `/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/stage3_process_grpo/gpt4o_penalty_0p5_100step_seed42_20260801`

Each directory is single-use and the launcher refuses overwrite or implicit resume.  If a pilot
fails, the failure, latest complete checkpoint, ledger and root cause must be audited before the user
is asked whether to authorize any recovery.  There is no immediate automatic rerun.

## Pre-launch tests

The following CPU-only suite passed before launch:

```text
PYTHONPATH=scripts python3 -m unittest \
  scripts/test_stage3_aigcbest_judge.py \
  scripts/test_stage3_openrouter_judge.py \
  scripts/test_stage3_checkpoint_recovery.py \
  scripts/test_analyze_stage3_penalty_sensitivity.py \
  scripts/test_run_aigcbest_stage3_gpt4o_smoke.py

Ran 41 tests ... OK
```

Training had not started at the timestamp of this record.  Each run writes its own static preflight,
training log, reward audit, Judge cache, attempt/budget ledger, fallback ledger and checkpoint.
