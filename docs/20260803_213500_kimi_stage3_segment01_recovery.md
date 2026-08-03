# Kimi Stage3 segment01 recovery

Recorded at 2026-08-03 21:35 CST.

## Observed terminal failure

- The original `segment00` stopped after step 19/1500 and produced no checkpoint.
- The failure was not CUDA OOM, a budget breach, or a model/provider identity mismatch.
- OpenRouter returned HTTP 200 with an incomplete JSON body on rank 7. The old fail-closed path retained the ambiguous reservation and terminated all ranks.
- The supervisor conservatively settled all eight in-flight reservations. The cumulative ledger then contained 161 completed requests and USD 0.59327564 committed spend, with no remaining reservation or budget breach.
- The incomplete output was preserved under `failed_precheckpoint_outputs/segment01-prelaunch-output`; it was not deleted or overwritten.

## Narrow recovery hardening

Commit `c914fd3` makes only the HTTP-2xx/incomplete-body case eligible for the already frozen bounded structural fallback:

- the remote request is not resent;
- the USD 0.05 reserve is committed once conservatively;
- the local target-blind structural fallback remains limited to 24 total and 4 consecutive uses;
- identity, provider, source, schema, and budget failures remain terminal;
- a manually approved recovery can begin at a new launch index and therefore receives a fresh cache namespace.

The Stage3 Judge and checkpoint-recovery regression suite passed 46 tests before launch.

## Restart boundary

- Run directory: `/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/stage3_process_grpo/kimi26_full3epoch_seed42_fresh_20260803`
- Recovery segment: `segment01`
- Parent: frozen Stage2 outcome-GRPO epoch 2 step 1000
- Resume checkpoint: none (none existed)
- Cache namespace: `segment01`
- Save interval: 100 steps
- Formal ledger hard cap: USD 26.71782984 (unchanged)
- Supervisor launch indices remaining: 1 through 3

The first new training step completed successfully in 78.54 seconds with finite loss/gradient and audited accuracy, format, and process rewards. This confirms execution but is not a performance result.

