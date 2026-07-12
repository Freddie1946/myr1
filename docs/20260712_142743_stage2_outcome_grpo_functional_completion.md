# Stage 2 outcome GRPO functional completion

Timestamp: 2026-07-12 14:27:43 Asia/Shanghai

## Outcome

Attempt 02 completed with exit status 0 and exercised the intended online path:

1. Loaded the newly trained SFT policy checkpoint.
2. Loaded and ZeRO-3-partitioned a reference model for KL.
3. Sampled two image-conditioned completions during training.
4. Computed PathMMU accuracy and format rewards online.
5. Entered GRPO loss/backward/optimizer framework code.
6. Gathered and saved a two-shard checkpoint.

Vision modules were frozen. Policy language-model parameters remained configured as trainable. Physical GPUs 1 and 2 peaked at approximately 15.7 GiB each and were released normally.

## Zero-variance caveat

Both completions reached the 64-token cap before closing `</think><answer>...`, so both accuracy and format rewards were 0. Group reward standard deviation was therefore 0, producing normalized advantages of 0, `loss=0`, and `grad_norm=0`.

This proves functional online generation/reward/checkpoint plumbing, but it does not prove a nonzero policy parameter update. It must not be presented as evidence of GRPO effectiveness.

For the formal run, use a completion cap large enough for the required format and monitor reward variance. A later debug retry may use more generations and a longer completion cap, but it is not required to validate the remaining inference/scoring/migration plumbing.

## Metrics

- Online reward events: 4 (2 completions × 2 reward functions).
- Completion length: 64 tokens.
- Mean accuracy reward: 0.0.
- Mean format reward: 0.0.
- Reward standard deviation: 0.0.
- KL: 0.0.
- Training runtime: 39.2518 seconds.
- Total wall time including model/reference load and save: 105.21 seconds.
- Saved output size: 7.1 GiB.

## Checkpoint

`/home/wjy/revision_runs/10_training/pathvlm_r1_v1/debug_e2e/qwen2_5_vl_3b/stage2_outcome_grpo/n0008/seed0042/attempt02/output`

Because the effective gradient was zero, it should be numerically equivalent to its SFT parent aside from serialization/config metadata. It remains useful for testing downstream checkpoint handoff.

## Next action

Run validation-only inference from this stage output, save raw completions, and apply the same `pathmmu_rewards.py` module offline. Stage 3 process reward remains awaiting an agreed final definition and will not be silently invented.

