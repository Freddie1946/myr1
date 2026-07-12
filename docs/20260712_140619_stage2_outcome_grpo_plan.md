# Stage 2 outcome-reward GRPO debug plan

Timestamp: 2026-07-12 14:06:19 Asia/Shanghai

## Parent and data

- Parent: newly completed E1 3B SFT checkpoint, not the base model and not a historical checkpoint.
- Data: first 8 deterministic records of the frozen RL split, recorded by SHA-256.
- Validation/test: not accessed during GRPO training.

## Online behavior being tested

The policy generates two sampled completions per prompt during the training step. The reward functions execute immediately inside the trainer:

- `accuracy_reward`: extracts the final `<answer>` when present and compares multiple-choice letters A-D; falls back to normalized exact text.
- `format_reward`: checks for the complete `<think>...</think><answer>...</answer>` structure.

Each per-completion online reward event is appended as JSONL. The same reward module will later score saved validation completions offline, allowing online/offline parser consistency to be checked.

## Engineering adaptations

- A wrapper injects PathMMU rewards into the existing repository `grpo_rec` entry without editing the dirty repository.
- A stable symlink whose name contains `Qwen2.5-VL` points to the completed SFT output. This is required because the historical trainer selects multimodal model and processor classes using a model-path substring.
- SDPA is used instead of the historical FlashAttention monkey patch.
- ZeRO-3 offloads optimizer state only; policy and reference-model parameter partitions remain on GPUs.
- Vision modules, including the projector under `visual`, are frozen.

## One-step settings

- Two GPUs, per-device batch 1, global batch 2.
- Two generations, the minimum valid group size for global batch 2.
- Completion cap 64 tokens.
- KL beta 0.04 so the reference path is exercised.
- One GRPO optimizer step; full policy language-model parameters remain trainable.

## Preconditions

1. Reward unit tests pass.
2. Every adapter image exists and required fields are non-empty.
3. SFT symlink resolves to the completed checkpoint.
4. GPUs 1 and 2 remain sufficiently free.
5. At least 20 GiB disk headroom remains before saving another gathered checkpoint.

