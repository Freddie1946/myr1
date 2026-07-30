# PathMMU SFT3000, SFT4000 and Stage2 diagnostics completed

Completed: `2026-07-29T22:37:42+08:00`

The fixed SFT3000 parent, exposure-matched SFT4000 control and selected Stage2 Outcome-GRPO
checkpoint have now completed the same PathMMU v2 test999 development-diagnostic protocol.
Stage3 remains unstarted.

| checkpoint | correct | accuracy | choice extracted | format correct |
|---|---:|---:|---:|---:|
| SFT3000 parent | 582/999 | 58.26% | 997/999 | 997/999 |
| SFT4000 control | 595/999 | 59.56% | 995/999 | 991/999 |
| Stage2 Outcome-GRPO | 609/999 | 60.96% | 999/999 | 999/999 |

All three runs used the same test file and parser, BF16 without quantization, greedy decoding,
batch size 8 and a 1,024-token cap. None produced an empty completion or hit the generation cap.

The paired comparisons are:

| comparison | left only correct | right only correct | right-minus-left | exact two-sided McNemar |
|---|---:|---:|---:|---:|
| SFT3000 vs SFT4000 | 145 | 158 | +13 / +1.30 pp | 0.490646 |
| SFT3000 vs Stage2 RL | 59 | 86 | +27 / +2.70 pp | 0.030482 |
| SFT4000 vs Stage2 RL | 142 | 156 | +14 / +1.40 pp | 0.451463 |

The main scientific reading is limited but useful: ordinary additional SFT explains part of the
raw improvement over the SFT3000 parent, while Stage2 RL has the highest point estimate. On this
single seed and development-classified split, only the unadjusted SFT3000-versus-Stage2 paired
comparison is below 0.05; it does not remain below 0.05 after a three-comparison Holm correction.
These diagnostics are therefore inputs to bad-case analysis and Stage3 planning, not final
multi-seed performance claims.

A twelve-model joint report was generated at:

`/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/reports/pathmmu_test999_badcases_available12_20260729_2241`

It contains 23 cases missed by all twelve models, 11 cases answered by all twelve, and 988 cases
with prediction disagreement. The report summary SHA-256 is
`1104acb5cd44ed0de8eb7458591586155efc3792171107e00a8539753d48b5a9`.

Two SFT3000 full-run launch attempts exited before model loading or data access: the first used a
mistyped environment directory, and the second used obsolete argument names plus an invalid
split-role label. They wrote no predictions. The corrected third launch completed all 999 records.
The immutable completion record is
`protocol/pathmmu_test999_sft3000_sft4000_stage2_completion_manifest_20260729_223742.json`.
