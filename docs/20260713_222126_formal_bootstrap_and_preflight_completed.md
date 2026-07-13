# Formal bootstrap and strengthened preflight completed

Timestamp: `2026-07-13 22:21:26 Asia/Shanghai`

## Result

The formal-machine bootstrap completed successfully without launching training. The authoritative
report is `/home/wjy/pathvlm_r1_v1_formal/reports/preflight_report.json`; its overall `passed` value and
all individual gates are `true`.

## Prepared environment

- SFT: Python 3.10, Torch 2.6.0+cu124, CUDA runtime 12.4, Transformers 4.49.0,
  Tokenizers 0.21.0, Accelerate 1.2.1, Datasets 3.2.0, DeepSpeed 0.15.4, TRL 0.9.6,
  LLaMA-Factory 0.9.2.
- GRPO: Python 3.10, Torch 2.6.0+cu124, CUDA runtime 12.4, Transformers 4.49.0,
  Tokenizers 0.21.0, Accelerate 1.4.0, Datasets 3.3.2, DeepSpeed 0.15.4, TRL 0.15.2,
  and the vendored Open-R1 source.
- Both environments see CUDA and eight devices. LLaMA-Factory and Open-R1 import checks pass, and
  parser/reward regression tests pass.

## Fixed source, model, and data

- LLaMA-Factory official v0.9.2 commit:
  `e2299e261be852304bb1d370515078193ab12bd8`.
- LLaMA-Factory launcher SHA-256:
  `8f16bb782a6da2122b5accd50ce1a01fd99284dffa20ff340850ec3b06927b77`.
- Base model: `Qwen/Qwen2.5-VL-7B-Instruct` revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- Model shape: `qwen2_5_vl`, hidden size 3,584, 28 layers, five shards totaling
  16,584,414,560 bytes.
- Config, model-index, and tokenizer-config hashes exactly match `protocol/base_model_manifest.json`.
- PathMMU resolved revision: `054e64e56e599e9636024f1471d49ecae4a2784f`.
- PathMMU archive: 1,931,564,255 bytes, SHA-256
  `9799d35d7d5dace0ccb34d71aea1c13900287fca50e8b9b51d282ea23b80eb36`.
- Required images: 3,809; valid: 3,809; missing: 0.
- SFT/RL/validation/test image counts are 2,121/708/272/708 and all six pairwise overlaps are zero.
- All exact adapter source counts pass; `picked.json` was not used; test remains evaluation-only.

Machine-resolved SFT and Outcome-GRPO configs are under
`/home/wjy/pathvlm_r1_v1_formal/generated_configs`, and persistent paths are in
`/home/wjy/pathvlm_r1_v1_formal/FORMAL_PATHS.env`. The installation occupies approximately 32 GiB.

## Remaining execution gate

At the final preflight snapshot, GPU 0 still contained the collaborator's Python process using about
11.6 GiB. GPUs 1-7 were idle except for display baseline allocations. No process was terminated and no
training was started. The next action is a fresh occupancy check after GPU 0 is released, followed by
the formal SFT one-step/save/reload/resume smoke only. Formal long training remains unauthorized until
that smoke is reviewed and accepted.
