# Stage 2 nonzero-gradient retry plan

Timestamp: 2026-07-12 15:04:50 Asia/Shanghai

## Correction to the previous interpretation

The two Attempt 02 completions each reached exactly 64 generated tokens and stopped inside `<think>`, before producing `<answer>`. This directly explains `format_reward=0` and why the answer parser returned `predicted_choice=null`.

It does not prove that the completions would have become correct with more tokens. The first reasoning trace leaned toward fibroblasts (choice D), and the second leaned toward melanocytes (choice B), while the correct answer was keratinocytes (choice A). Accuracy failure was therefore not safely attributable to truncation alone.

## Retry design

- Reuse exactly that failed RL record: frozen RL source index 6.
- Parent: completed 3B SFT checkpoint, not the zero-gradient GRPO serialization.
- Four generations instead of two.
- Maximum 192 completion tokens instead of 64.
- Two GPUs, per-device batch 2, global batch 4.
- Same real accuracy and strict format rewards.
- One optimizer step.
- Test remains sealed.

## Required evidence

The retry is successful only if:

1. Online reward events contain four completions and two reward components per completion.
2. At least one reward component varies within the four-completion group.
3. `reward_std > 0`.
4. `grad_norm > 0`.
5. The run exits successfully and saves a loadable checkpoint.
6. A language-model tensor differs from its SFT parent.
7. A frozen visual tensor is identical to its SFT parent.

An exit-0 run with zero reward variance will be recorded but will not satisfy this gate.

## Memory and fallback

Attempt 02 used approximately 15.7 GiB per GPU at batch 1. Batch 2 plus longer completions may still fit within 24 GiB but must be monitored. If it OOMs, do not change to LoRA or unfreeze/freeze different components. Retry with lower debug image pixels or wait for more free GPUs, documenting the change.

## Storage

Attempt 02's online reward JSONL, log, run manifest, configs, tokenizer/config metadata, and result documentation are retained. Its two large model shards are eligible for deletion because the run had zero gradient and the saved model is not a meaningful training result. Exact deletion requires approval.

