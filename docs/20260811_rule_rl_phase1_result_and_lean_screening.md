# Rule-RL Phase 1 result and lean screening decision

Phase 1 completed without accessing PathMMU test999. The automatic, absolute
sampled-format gate returned `r2_required`; its original output is preserved.
That implementation did not match the previously agreed relative-format
criterion: R2 was to be added only if R1 degraded format relative to R0.

At step 50, greedy format was 100% for both arms. Sampled format was 90.0% for
R0 and 90.6% for R1. R1 produced four additional correct train-probe answers
and five additional correct PathMMU-validation answers, but the paired
differences were not statistically significant.

> R1 showed a consistent but non-significant advantage over R0 without format
> degradation; therefore R1 was provisionally selected for downstream
> mechanism screening.

This is not a claim that R1 has won or that format reward has been proven to be
the bottleneck. R2 is skipped for the current screening. A downstream arm is
flagged for possible R2 only if sampled format falls by more than 5 percentage
points relative to the matched R1 baseline, or greedy format falls below 98%.

The lean mandatory sequence is:

1. Evaluate all six L-r32 SFT checkpoints on PathMMU validation and inspect the
   training-fit curve. Run full OOD retention only for shortlisted checkpoints.
2. Run R1-G20-LR3 with a 50-step scheduler and stop safely at step 25.
3. Run R1-G2-LR1 with a 500-step scheduler and stop safely at step 250.
4. Compare all three arms at exactly 500 prompt exposures, then stop for joint
   analysis. G20-LR10, RL-r32 and Judge remain conditional.
