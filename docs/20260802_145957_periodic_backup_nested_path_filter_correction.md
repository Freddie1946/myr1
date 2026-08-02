# Periodic backup nested-path filter correction

Timestamp: `2026-08-02 14:59:57 CST`

## Correction

Remote-tree verification of the first rolling backup at HF revision
`0f0fb0561194284e6f7b40a201b74881e1628d9b` found that it contains the active PathVQA files and
Stage3 top-level logs, but not the intended nested `stage3_gpt4o/reward_audit/` records or
`stage3_gpt4o/judge/` ledgers.  A GNU `find -maxdepth 1` predicate had global traversal scope even
though it appeared inside the selection expression.

The auditor now performs three explicit traversals: Stage3 top-level JSON/JSONL/log files, all
rank JSONL files below `reward_audit/`, and top-level JSON ledgers below `judge/`.  Locks, weights
and credentials remain excluded.  Bash syntax and a no-network dry run passed; the corrected staged
aggregate SHA-256 is `0c03f728b7e9388935e3fb32b97dd7f7031e339d274610e7693bc726beac4b33`.

The immutable 241-file completed-results backup was never affected.  The first live revision is
retained as a truthful partial disaster-recovery snapshot; the next content-changed cron/manual
run will add the missing nested Stage3 evidence and will be verified separately.

This is an operational correction (`formal_result: false`), not an efficacy result.

