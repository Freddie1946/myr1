# Validation inference and offline scoring plan

Timestamp: 2026-07-12 14:29:14 Asia/Shanghai

## Scope

Load the Stage 2 gathered checkpoint on one free GPU and run deterministic generation on the first two records of the frozen validation split. Test remains sealed.

## Outputs

- One JSONL row per sample containing image path, problem, solution, raw completion, parsed choices, accuracy reward, and format reward.
- A JSON metrics summary.
- Full command, log, run manifest, environment identifiers, and source adapter hash.

## Consistency check

Offline scoring imports the exact same `pathmmu_rewards.py` used inside online GRPO. This isolates the difference between when scoring occurs:

- Online: completions are generated and rewarded inside the training step; rewards immediately determine GRPO advantages.
- Offline: already saved completions are read/scored after generation; scores cannot alter the completed training step.

## Inference settings

- Physical GPU 1 only.
- SDPA and bfloat16.
- Greedy deterministic generation (`do_sample=false`).
- Maximum 128 new tokens to allow the required tags to close more often than the 64-token GRPO smoke cap.

These two samples are only a pipeline check, not an accuracy estimate.

