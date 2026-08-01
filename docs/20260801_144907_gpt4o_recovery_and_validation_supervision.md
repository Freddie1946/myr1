# GPT-4o recovery and validation supervision

Timestamp: 2026-08-01 14:49:07 +0800

The fresh penalty-0.5 recovery passed its static preflight and started from repository commit
`25a25f7`.  The preflight records a new Judge root, `cache_reuse=false`, 824 physical attempts,
$5.50, exact `gpt-4o-2024-08-06`, the fixed parent and no automatic restart.

`scripts/supervise_gpt4o_0p5_recovery_and_validation.sh` waits for that active run.  It does not
restart it.  A successful run must have a complete distributed checkpoint-100, a fresh 100-step
train-state audit, exactly 800 process reward records, the exact Judge and 0.5 coefficient, no more
than 824 attempts, no more than $5.50, no unresolved reservations and at most 24 fallbacks.

Only after those gates pass does the supervisor run the frozen, deterministic PathMMU validation385
protocol for the completed 0.3, 0.4 and recovered 0.5 model outputs.  The existing validation script
now permits explicit model and destination paths while retaining its historical defaults and all
data/hash/scoring checks.  Each validation output is single-use and test999 remains excluded.
