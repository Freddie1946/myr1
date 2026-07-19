# Validation-curve v2 preflight completed

Timestamp: `2026-07-19T18:23:37+08:00`

## Scope

After commit `543bb96` froze `pathmmu_image_disjoint_v2`, the validation-curve runner was executed
with `--preflight-only`. It did not load the test split, generate an answer, create a validation run
directory, update a model, or start training.

## Result

The gate returned `passed: true` with:

- exactly 21 planned jobs: frozen base plus n=2000 and n=3000 seed-42 epochs 1--10;
- both completed formal SFT parent manifests and their expected hashes;
- all 20 epoch snapshot manifests/files/hashes accepted by the runner;
- v2 rewritten validation SHA-256
  `6434da3e89e81c4e6a01736a1eda885858b56c28f8bfef284bd730693f37a2ca`;
- pinned chat-template SHA-256
  `ad60d90252ed0b0705ba14e2d0ad0fec0beac1ea955642b54059b36052d8bc96`;
- parser regression tests and both code-hash manifests passing;
- physical GPUs 0--7 each at 18 MiB and 0% utilization, with no compute process;
- `test_accessed: false`.

Persistent evidence:

`/home/wjy/pathvlm_r1_v1_formal/reports/sft_validation_curve_v2_preflight_20260719_181800.json`

SHA-256:

`a32451d546ef210f135a140bac15edceadfabe9e329690bb33684a84b488a419`

## State after the gate

No validation inference is running and no new run directory was created. Attempt02 is ready but has
not launched. A launch must create a fresh directory, run all 21 jobs without reusing Attempt01, save
8,085 raw predictions plus job/aggregate audits, and continue to prohibit test access. Outcome GRPO
and all training remain stopped.
