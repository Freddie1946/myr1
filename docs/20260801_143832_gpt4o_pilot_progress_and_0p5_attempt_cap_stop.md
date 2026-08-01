# GPT-4o coefficient pilots: progress and penalty-0.5 attempt-cap stop

Timestamp: 2026-08-01 14:38:32 +0800

## Current state

The sequential supervisor behaved as frozen: it accepted two complete arms, stopped after the
third arm failed, and did not restart it.

| Penalty | Optimizer steps | Checkpoint | HTTP attempts | Valid Judge caches | Fallbacks | Committed cost | State |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0.3 | 100 | complete step 100 | 800 | 800 | 0 | $4.122075 | passed supervision |
| 0.4 | 100 | complete step 100 | 800 | 799 | 0 | $4.130780 | passed supervision |
| 0.5 | 99 | none | 794 committed + 6 unresolved reservations | 792 | 0 | $4.122265 committed | stopped |

The 0.4 arm contains one billed response that failed strict validation and was retried.  One exact
logical duplicate was served from its within-arm cache, so the arm still completed within 800
physical HTTP attempts.  Both completed arms have final model weights, complete distributed
`checkpoint-100`, train-state audit, empty reservation set, exact served-model audit and no budget
breach.

## Exact 0.5 failure

The 0.5 arm completed and logged optimizer step 99.  Those 99 steps produced 792 logical process
rewards.  Before the final step it had also consumed two extra physical attempts: one billed
response failed strict validation, and one ambiguous transport failure was conservatively billed.
The ledger therefore contained 794 committed attempts.

At step 100, six ranks obtained reservations, bringing committed plus outstanding attempts to the
frozen maximum of 800.  The remaining two ranks were refused by `BudgetLedger.reserve` with:

```text
BudgetError: maximum unique OpenRouter request count reached
```

The inherited error text says OpenRouter, but the ledger was the AIGCBest per-attempt ledger.  The
distributed launcher terminated the other ranks after the two fail-closed refusals.  The six
in-flight reservations have no validated local response and remain unresolved.  No rule fallback
was used because an exhausted paid-attempt authorization is a hard stop, not a Judge outage.

This was a deterministic consequence of defining 800 as the physical-attempt ceiling while also
allowing retries.  A complete 100-step run needs 800 logical judgments before any retry overhead.
The cap did exactly what was documented; it was the wrong quantity if bounded retries were expected
to coexist with 100 complete steps.

## Recovery and budget boundary

Saving was frozen at step 100, so the failed arm has no checkpoint and cannot resume from step 99.
No process is running and all GPUs are idle.  No files have been deleted, no cache has been reused,
and no rerun has started.

Conservative pilot expenditure is:

- completed 0.3: $4.122075;
- completed 0.4: $4.130780;
- committed 0.5: $4.122265;
- six unresolved 0.5 reservations at $0.02 each: $0.120000;
- conservative total: $12.495120;
- remaining under the approved $18 aggregate ceiling: $5.504880.

A scientifically clean recovery would start 0.5 again from the fixed Stage2 parent in a new run
directory, with no cache reuse.  It requires explicit permission to reinterpret the request gate as
800 logical judgments plus a separately bounded retry allowance.  A proposed recovery cap is 824
physical attempts (800 base plus at most 24 retry attempts) and $5.50 total, which remains inside
the original $18 aggregate budget after conservatively settling the six unresolved reservations.
This proposal has **not** been authorized or executed.

PathMMU validation385 comparison has not started.  PathMMU test999 was not accessed.
