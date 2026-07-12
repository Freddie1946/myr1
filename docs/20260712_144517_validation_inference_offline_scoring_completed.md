# Validation inference and offline scoring completed

Timestamp: 2026-07-12 14:45:17 Asia/Shanghai

## Outcome

The Stage 2 checkpoint loaded successfully on physical GPU 1 and completed deterministic multimodal generation for two frozen validation records. Raw predictions and offline scores were persisted. Exit status was 0 and the GPU returned to baseline afterward.

- Samples: 2 validation QA.
- Test accessed: no.
- Generation: greedy, `do_sample=false`, maximum 128 new tokens.
- Accuracy reward: 1/2, mean 0.5.
- Strict think/answer format reward: 2/2, mean 1.0.
- Wall time including checkpoint load: 17.27 seconds.
- Maximum resident host memory: 5,828,372 KiB.

These two examples are a pipeline check only and must not be reported as a performance estimate.

## Saved artifacts

- Raw predictions: `debug_e2e/qwen2_5_vl_3b/evaluation/validation_n0002/stage2_checkpoint/results/predictions.jsonl`
- Metrics: `debug_e2e/qwen2_5_vl_3b/evaluation/validation_n0002/stage2_checkpoint/results/metrics.json`
- Full log: `debug_e2e/qwen2_5_vl_3b/evaluation/validation_n0002/stage2_checkpoint/inference.log`

## Online versus offline confirmation

The same `pathmmu_rewards.py` functions were used in both contexts. In GRPO they were called before the optimizer step and determined advantages. Here they were called after deterministic completions were generated and only produced evaluation records. Offline scores cannot retroactively change training.

## Important generation finding

The 128-token validation outputs closed both required tags, while the 64-token online GRPO outputs were truncated before `</think><answer>`. Formal GRPO should therefore avoid a 64-token completion cap for this prompt format; the exact cap should be chosen on validation/runtime constraints, not test.

