# GPT-4o penalty-0.5 fresh recovery approval

Timestamp: 2026-08-01 14:44:36 +0800

The user approved the recovery proposed after the physical-attempt-cap stop:

1. conservatively settle the six unresolved reservations from the stopped arm at $0.02 each;
2. start penalty 0.5 again from the immutable Stage2 seed-42 parent;
3. use a new run directory and new Judge root, with no cache reuse;
4. train the same 100 steps and produce the same 800 logical process judgments;
5. allow no more than 824 physical HTTP attempts, including up to 24 retry attempts;
6. enforce a $5.50 recovery-run hard cap, preserving the original $18 aggregate gate; and
7. perform no automatic rerun if this recovery fails.

All model/training settings remain identical to the completed 0.3 and 0.4 arms.  Save interval
remains step 100 for a matched comparison.  The sole changes are the separately authorized retry
allowance and corresponding recovery budget cap.

The fixed recovery directory is:

`/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/stage3_process_grpo/gpt4o_penalty_0p5_100step_seed42_20260801_recovery_fresh01`

The generic launcher accepts this mode only when both conditions hold:

```text
PATHVLM_STAGE3_RECOVERY_ID=fresh01
penalty=0.5
```

Any other recovery ID or coefficient pairing fails before secret loading, filesystem mutation or
paid requests.  The recovery preflight records `cache_reuse=false`, the exact budget, attempt limit,
repository commit and fixed parent identity.  PathMMU validation385 remains post-training; test999
remains excluded.
