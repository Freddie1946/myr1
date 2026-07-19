# Formal n=3000/n=2000 SFT completion and validation-curve plan

Timestamp: `2026-07-19T15:27:42+08:00`

## Completed formal training

The exact fail-closed handoff completed without retry or ambiguity.

- n=3000 seed-42 completed at `2026-07-17T14:02:45+08:00`, with 3,750 optimizer steps,
  ten scientific snapshots at steps 375 through 3,750, and all 14 formal gates true.
- After all eight GPUs returned to the 18-MiB idle baseline, the watcher launched one fresh
  base-model n=2000 seed-42 run at `2026-07-17T14:03:36+08:00`.
- n=2000 completed at `2026-07-18T03:05:16+08:00`, with 2,500 optimizer steps, ten scientific
  snapshots at steps 250 through 2,500, and all 14 formal gates true.
- Both runs reloaded their final checkpoints, changed language parameters, preserved exact visual
  tensor equality, retained the latest two full resume checkpoints, maintained the 550-GiB disk
  reserve, and did not access test.
- The handoff state became terminal `completed` at `2026-07-18T03:05:38+08:00`. Its subordinate
  `parent_status` and `target_status` fields remain stale heartbeat values (`running`); the terminal
  top-level state, completed events, and both authoritative run manifests are unambiguous. The event
  wording `completed_and_validated` means manifest/gate validation, not validation-split inference.

No performance inference was included in the deliberately training-only handoff. Therefore n=2000
and n=3000 are completed formal training results but do not yet have validation accuracy curves.

## Frozen validation scope

The user authorized filling this gap. The new curve contains exactly 21 deterministic jobs:

1. the exact frozen base model revision;
2. n=2000 epochs 1 through 10; and
3. n=3000 epochs 1 through 10.

Every job uses only the frozen `validation_0385` rewritten records with SHA-256
`6434da3e89e81c4e6a01736a1eda885858b56c28f8bfef284bd730693f37a2ca`, deterministic decoding,
`max_new_tokens: 192`, the fixed prompt already used by the historical final-checkpoint validation,
and parser v2. Test is forbidden and remains untouched.

The model-selection rule is frozen before observing results: separately for n=2000 and n=3000,
maximize validation accuracy; exact tie, maximize format reward; exact tie, choose the earliest epoch.
The base result is a reference and is not an epoch candidate. No test result may alter this rule.

## Integrity and output gates

Before inference, the runner requires a clean Git tree, its frozen code manifest, passing formal
preflight, passing parser regression tests, exact parent paths/config hashes/gates, all validation
image paths, and idle physical GPUs 0--7. It verifies the size and SHA-256 of every file in all 20
scientific snapshot manifests and the frozen base config/index/tokenizer hashes.

Jobs are assigned deterministically across GPUs 0--7, with staggered model loading and at most one
job per GPU. Each job writes a separate manifest, command, inference log, metrics JSON, and 385-record
raw prediction JSONL. After inference, every record is re-parsed offline; index, image, question,
solution, predicted/target choice, accuracy reward, format reward, aggregate count, and aggregate
means must exactly agree. A failure stops jobs that have not started and makes the curve non-formal;
there is no automatic retry.

Seven isolated curve-runner tests pass. They cover valid/corrupt/wrong-step snapshots, exact offline
rescore, prediction-index mismatch, the frozen tie-break rule, and missing-epoch rejection. Existing
parser regression tests also pass. A full hash/GPU preflight-only run must pass after commit before
the 21-job curve launches.

At audit time all eight GPUs were idle. `/home` had 998,876,680,192 free bytes. Curve outputs are
small relative to the checkpoints, but another two-tier formal training run is not permitted until
free space again exceeds its 1,157,493,686,272-byte start gate.
