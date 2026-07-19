# Stage-2 priority pilot and formal n=1000 protocol frozen

Timestamp: `2026-07-20T03:30:24+08:00`

## User authorization and boundary

The user approved beginning with one 50-step engineering pilot and, only after every predeclared
gate passes, automatically starting the highest-priority formal combination: selected SFT n=3000
epoch 3 followed by Outcome-GRPO n=1000, seed 42, for three epochs. No RL 250/500 arm, alternative
SFT/RL combination, seed 43/44, Stage 3 or formal test inference is authorized in this pipeline.

The formal run always restarts from the exact validation-selected SFT snapshot at step 1125. The
pilot output can never become its parent.

## Pilot contract

The pilot uses all 1,000 frozen v2 RL records, eight GPUs, four generations, accuracy plus strict
format reward, learning rate `1e-6`, beta `0.04`, and the prompt-v2 contract. It is always marked
`formal_result: false`.

1. Train to step 25 and save a full resumable ZeRO checkpoint.
2. Stop only after checkpoint save completes.
3. Start a fresh distributed process and resume the saved checkpoint to step 50.
4. Preserve 800 reward events representing 400 generated completions. Each event records the exact
   frozen record index, resolved image path and SHA-256, question, structured prompt and contract.
5. Recompute every online accuracy/format reward offline with the same parser.
6. Require finite training metrics, nonzero gradient and positive reward standard deviation in each
   segment, at least one positive-variance four-generation group in each segment, at least 98%
   online strict-format compliance, a changed language tensor, exactly unchanged visual tensor and
   a loadable final save.
7. Run deterministic inference on all 385 validation records and save raw predictions. Require at
   least 222/385 correct (no more than three percentage points below the 233/385 parent), at least
   98% format and choice extraction, no empty completion, and no significant paired degradation by
   exact two-sided McNemar test at alpha 0.05.

Any failed criterion preserves artifacts and stops. There is no automatic retry, prompt tuning,
threshold change or favorable rerun.

## Formal n=1000 contract

After a passing pilot and a new idle-GPU/port/storage check, the formal run starts from the original
SFT parent with the same optimizer/reward/prompt settings. Three explicit passes through the 1,000
record RL adapter yield 500 optimizer steps per epoch and 1,500 total steps.

- Save full resume state every 100 steps and retain the newest one.
- Retain model-only snapshots at steps 500, 1000 and 1500.
- Preserve 24,000 online reward events and exact source metadata.
- Require online/offline reward consistency, positive reward variance, finite/nonzero learning,
  language change, visual equality, loadability and at least 98% online format compliance.
- Validate all three epoch snapshots on the complete validation split. Select maximum accuracy;
  exact tie, maximum format; exact tie, earliest epoch.
- Keep at least 550 GiB free. The post-pilot measured checkpoint size is used in a conservative
  transient-storage projection before formal launch.

The corrected 999-QA test remains unopened and cannot influence the pilot or formal selection.

## Implementation and reviewer mapping

The fail-closed runner is `formal_machine/run_stage2_priority_pipeline.py`, launched only through
`scripts/launch_stage2_priority_pipeline.sh`. The frozen scientific values are in
`protocol/stage2_priority_n1000_seed42_manifest_20260720_033024.json`; executable hashes are frozen
separately before execution.

This plan advances Outcome-GRPO evidence for R2-4 and an outcome-only arm for R3-3. It prioritizes
one point toward R3-5 under the user's time constraint, but does not complete the RL scale curve.
It preserves raw paired predictions and seed provenance for R3-6 but does not replace the pending
multi-seed analysis.
