# Stage-2 formal n=1000 seed-42 continuation frozen

Timestamp: `2026-07-20T06:52:00+08:00`

## Authorization and exact lineage

The user confirmed the highest-priority formal line and authorized its launch:

`SFT n=3000, seed 42, validation-selected epoch 3 / step 1125` ->
`Outcome GRPO n=1000, seed 42, three epochs / 1500 optimizer steps`.

The exact parent remains:

`/home/wjy/pathvlm_r1_v1_formal/runs/stage1_sft/n3000_seed0042/formal_sft_n3000_seed0042_20260717_014544/epoch_snapshots/checkpoint-1125`

Its snapshot-manifest SHA-256 is
`6c8708dd677f5ab1563e77317d86cd1b53cb9b969ebda09e4a312581a2b9f1f1`.
The engineering pilot output is not the parent and the passing pilot will not be rerun.

## Frozen continuation gates

The formal-only launcher is fail closed and binds the exact source-pipeline, passing-pilot,
checkpoint-pruning and pre-pruning inventory hashes. Before creating the formal directory it must
also pass:

- clean Git worktree and exact committed code manifests;
- exact selected SFT parent identity, files and loadability;
- frozen `pathmmu_image_disjoint_v2` data and exact RL-1000/validation adapters;
- parser-v2 and reward-audit regression tests;
- at least 897,060,103,530 free bytes, including the frozen 550-GiB reserve;
- all eight authorized GPUs idle and TCP port 29820 free.

The original pipeline manifest remains truthfully failed at its pre-formal storage gate. A separate
`formal_only_continuation_manifest.json` records this approved continuation without rewriting that
history.

## Training and validation contract

- Eight GPUs; global batch size 8; four generations per prompt.
- Learning rate `1e-6`, beta `0.04`, maximum completion length 192.
- Full language-model parameters trainable; vision tower and multimodal projector frozen.
- Full rolling resume checkpoint every 100 steps with one retained.
- Model-only snapshots at steps 500, 1000 and 1500.
- Preserve and audit exactly 24,000 online reward events.
- Require finite/nonzero gradients, positive reward variance, exact online/offline parser results,
  exact source/image/prompt provenance, model reload, language-tensor change and visual equality.
- Run deterministic 385-record validation on all three epoch snapshots, preserving raw generations.
- Select maximum validation accuracy, then maximum format rate, then earliest epoch.

Test remains sealed. No RL 250/500, other SFT/RL combinations, additional seeds, Stage 3, retries or
hyperparameter tuning are authorized by this continuation. Any failure is preserved and stops the
chain.

## Reviewer mapping

This run is the prioritized first formal Outcome-GRPO point for R2-1, R2-4, R3-3, R3-5 and R3-6.
It does not by itself complete the requested RL scale curve, multiple-seed uncertainty, Process
Reward ablation, expert evaluation, or final test evaluation.
