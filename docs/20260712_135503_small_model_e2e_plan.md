# Small-model end-to-end revision pipeline plan

Timestamp: 2026-07-12 13:55:03 Asia/Shanghai

## Decision

Use `Qwen/Qwen2.5-VL-3B-Instruct@66285546d2b821cf421d4f5eb2576359d3770cd3` as the debug proxy. The formal main model remains the pinned Qwen2.5-VL-7B base. Outputs from this pipeline are engineering evidence only and must not be reported as final experimental results.

The prepared 7B Attempt 04 is superseded before launch because shared-GPU availability makes repeated 7B debugging impractical. It is retained for provenance.

## Why 3B

- It is already cached locally and uses the same Qwen2.5-VL architecture and interfaces as the formal 7B model.
- The repository has historical evidence that this exact model family can complete full language-model SFT.
- No sub-billion Qwen2.5-VL checkpoint is cached, and changing architecture would validate less of the real pipeline.

## End-to-end stages

### E0. Data contracts

- Reuse the frozen image-disjoint v1 split.
- Build tiny, deterministic debug adapters from SFT, RL, and validation only.
- Keep test sealed.
- Validate paths, images, required fields, answer parsing, and exact sample counts.

### E1. SFT

- Run one full-parameter language-model optimizer step on the 8-sample SFT adapter.
- Freeze vision tower and multimodal projector.
- Save a loadable checkpoint and record resource peaks.

### E2. Outcome-reward online GRPO

- Adapt the frozen RL samples to the existing VLM-R1 online-generation data contract.
- Start from the new debug SFT checkpoint.
- Generate completions during training, compute answer-correctness and format rewards online, and complete one optimizer step.
- Use a small generation count and short completion length while preserving the same code path as formal training.

### E3. Process-reward completion

- First make the reward function and data schema executable on a tiny reconstruction set.
- Run unit/contract tests and one online training step only after the user and assistant agree on the final process-reward definition.
- Label all reconstructed supervision explicitly; do not present it as recovered historical data.

### E4. Validation inference and offline scoring

- Load each produced checkpoint and generate on a tiny validation-only adapter.
- Persist raw completions.
- Score saved completions offline using the same answer/format parsers and compare with online reward logs.
- Never use test in this debug loop.

### E5. Migration package

- Produce formal 7B configs, launch commands, environment lock, data manifests, expected directory tree, preflight checks, resume checks, and result templates.
- Parameterize model path, GPU count, DeepSpeed config, sample size, and seed; do not hard-code debug-host assumptions into formal configs.

## Current execution order

1. Complete E1 on physical GPUs 1 and 2.
2. Audit and adapt the existing `grpo_rec.py` path for PathMMU outcome reward.
3. Complete E2.
4. Pause for process-reward definition if needed, while still completing E4 and migration work that does not depend on it.

## Safety and storage

- Do not use or disturb GPU 0 or GPU 3 while occupied by other users.
- Monitor the 37 GiB current disk headroom. Keep only checkpoints needed to prove load/save and downstream handoff.
- Any checkpoint deletion requires exact-path verification and user approval.

