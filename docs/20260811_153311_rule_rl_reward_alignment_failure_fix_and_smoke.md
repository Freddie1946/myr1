# Rule-RL reward-alignment failure, correction and eight-GPU smoke

Timestamp: 2026-08-11 15:33:11 Asia/Shanghai

## Scientific status correction

The completed run at
`/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/formal_selected_rule_rl1000_20260811/formal_z2_gc0_mbs10_len384`
is scientifically invalid. Its step-100/150 weights and all downstream evaluation results must be
excluded from formal tables and claims. They remain on disk only for failure audit and must not be
used as a parent.

The failure is not attributed to LoRA rank, the rule reward definition or the selected L-SFT3000
parent. The vendored multimodal trainer's sampler had already repeated every source row four times,
but the custom-reward path repeated every non-prompt input column four additional times. This was
harmless only for the historical per-device batch of one; at the new per-device batch of ten it
silently paired most local completions with another prompt's solution.

## Complete invalid-run audit

All 12,000 accuracy-reward events were remapped from their actual prompt text to the frozen
1,000-row RL dataset and recomputed with the same offline parser.

| Gate | Invalid run result |
|---|---:|
| Exact prompt/problem/solution row pairing | 3,600/12,000 = 30.00% |
| Target choice letter aligned, including coincidence | 5,706/12,000 = 47.55% |
| Online binary reward equal to the true offline reward | 8,197/12,000 = 68.31% |
| Incorrect positive rewards | 966 |
| Missed positive rewards | 2,837 |
| Total incorrect binary rewards | 3,803 |
| Prompt/problem mismatches across accuracy + format events | 16,800/24,000 |
| Generation groups with inconsistent logged solutions | 1,200/3,000 |

The corrupted audit showed 38.23% all-zero groups versus 22.13% after true-answer recomputation,
and 55.67% mixed groups versus a true 62.73%. The defect therefore removed useful within-group
GRPO contrast and sometimes reinforced wrong answers.

## Correction

The trainer now copies reward kwargs one-for-one from the already repeated local batch and rejects:

- any input/completion length mismatch;
- any custom reward output with the wrong length;
- any formal solution or source-metadata length mismatch;
- any formal prompt/problem mismatch.

The same correction is applied to the non-vLLM and vLLM multimodal trainers. A CUDA-independent
per-device-batch-10 regression test reproduces the former boundary-crossing local shape. A new
post-run verifier maps every event back to the frozen dataset, recomputes accuracy/format rewards,
checks paired event identity, and reconstructs four-generation groups across rank boundaries.
The formal launcher now invokes this verifier after training and exits nonzero if the post-run
alignment gate does not pass.

## Corrected real smoke

Run:
`/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/reward_alignment_fix_smoke_20260811_1532_mbs10`

| Item | Value |
|---|---:|
| GPUs | 8 x A100 80 GB |
| Per-device batch | 10 |
| Generations per prompt | 4 |
| Completion cap | 384 |
| Optimizer steps | 2 |
| Accuracy events | 160 |
| Format events | 160 |
| Reconstructed groups | 40 |
| Alignment/offline-reward errors | **0** |
| Step-1 / step-2 accuracy reward | 0.4375 / 0.4625 |
| Step-1 / step-2 grad norm | 0.11996 / 0.13955 |
| Runtime | 70.50 seconds |
| Trainability/freeze gate | PASS |

This is an engineering smoke (`formal_result: false`). It does not authorize or claim a corrected
formal RL result. Before a formal rerun, the large-batch optimizer-update compression must be
calibrated on training/validation-only evidence; test results from the invalid run cannot select
the learning rate or schedule.

## Artifacts

- Machine-readable record:
  `protocol/rule_rl_reward_alignment_fix_smoke_20260811_153311.json`
- Corrected smoke verifier:
  `/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/reward_alignment_fix_smoke_20260811_1532_mbs10/reward_alignment_verification.json`
- Corrected smoke verifier SHA-256:
  `cd80d64b2eef274338ff05b6d907ce58b2f3d315f4ec869a3943df817afe42d0`
- Corrected smoke reward-event aggregate SHA-256:
  `88af3e9504617476ac042dbf109e2ee249b71a5979b24668bc479a48c7679695`
- Frozen RL JSON SHA-256:
  `5a818b3b4f3c9e5e1a17579111ff3e21008cf8124c4a05cb74eb2dcd326bbd73`
