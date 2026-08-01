# GPT-4o formal launcher and execution order frozen

Recorded at: `2026-08-01T18:48:26+08:00`

The user authorized the following order: complete GPT-4o Stage3 first while running compatible
non-GPU hosted baselines; then run GPU-occupying local baselines; run Kimi Stage3 last.  The user
declined a 4+4-GPU concurrency smoke and requested no user-defined USD budget ceiling.

The formal GPT-4o seed-42 launcher and bounded supervisor are now implemented but have not yet
made a network request or started training.  The scientific contract is unchanged: fixed Stage2
epoch-2/step-1000 parent, RL1000, coefficient 0.4, 1,500 optimizer steps, four generations,
global completion batch eight, learning rate 1e-6, beta 0.04, seed/data-seed 42, 192 completion
tokens, eight A100 GPUs, full language-model training and frozen vision/projector modules.

Full resume checkpoints are saved every 100 steps with the newest two retained.  Model-only epoch
snapshots are retained at 500/1000/1500.  The supervisor resumes only structurally complete
eight-rank checkpoints and performs at most three classified recoveries.

There is no user USD stopping threshold.  Infinite calling remains forbidden: the formal run has
12,000 logical judgments and at most 12,360 physical HTTP attempts.  The existing ledger requires
a finite numeric capacity, so it uses USD 247.20, exactly 12,360 times the conservative USD 0.02
reservation.  This is a technical accounting capacity, not an expected spend, model-selection
criterion or user budget.  Expected spend from the matched pilot remains approximately USD 61.96.

The formal launcher rechecks the exact parent/data/image/smoke hashes, authenticated model catalog,
public price row, clean Git state, port, eight idle GPUs and 500-GiB storage reserve before launch.
Four static contract tests and shell syntax checks pass.  A read-only preflight-only mode is
available and stops before creating a run directory or issuing a paid request.

Every baseline must pass fixed smoke cases before full evaluation.  Smoke checks adapter behavior,
served identity, image acceptance, output termination/extraction and scoring, never comparative
accuracy or prompt selection.

