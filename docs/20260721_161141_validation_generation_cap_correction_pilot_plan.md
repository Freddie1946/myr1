# Validation generation-cap correction pilot

Timestamp: `2026-07-21T16:11:41+08:00`

## Reason

The historical deterministic validation curve used `max_new_tokens=192`. The pinned Base model hit
that exact cap on 231/385 records, and 169/385 records had no parseable final choice. A
validation-only diagnostic with the same Base model, data, image pipeline, prompt, chat template,
parser and greedy decoding, changing only the cap to 512, produced 195/385 correct rather than
100/385 and did not hit the new cap. The diagnostic Base output maximum was 417 tokens. Therefore
the 192-token Base score is not a comparable estimate of multiple-choice capability.

## Authorized pilot

Before rerunning every checkpoint-selection curve, evaluate these three frozen checkpoints on the
385-record validation split only:

1. pinned Qwen2.5-VL-7B Base;
2. selected formal SFT n=3000 seed42 epoch 3 / step 1125;
3. selected formal Outcome-GRPO n=1000 seed42 epoch 2 / step 1000.

The pilot uses deterministic decoding and `max_new_tokens=768`. It preserves raw predictions and
is an evaluation-protocol diagnostic (`formal_result: false`), not a replacement for the historical
curve. Test remains forbidden and is not accessed. No model is trained or modified.

## Gates

- exact validation count is 385 for every model;
- data, chat-template, model/checkpoint and code identities match the frozen manifest;
- raw predictions, metrics and logs are preserved;
- online-saved rewards and choices exactly match offline parser v2 rescoring;
- every completion is non-empty and has a recorded token count;
- no completion reaches the 768-token cap;
- results are compared pairwise by the exact `(image, problem)` identity;
- any job failure is preserved as a failed attempt; there is no automatic retry;
- `test_accessed: false`.

If this pilot passes, a separate documented run will apply the corrected generation protocol to
all validation candidates that participate in checkpoint or data-scale selection.

