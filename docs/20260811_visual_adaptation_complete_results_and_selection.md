# Visual-adaptation validation: complete results and architecture selection (2026-08-11)

## Outcome

The pre-registered architecture gate selects **L**: frozen vision encoder, frozen multimodal
projector, and rank-16 LoRA on the language model. Within L, checkpoint step 16 (epoch 1.0) is the
selected checkpoint from the 1,500-sample screening run.

The main reason is not that projector adaptation A failed to learn. A reached the highest PathMMU
point estimate, but its advantage over L was only 1.04 percentage points and was statistically
unresolved, while its PathVQA retention fell below the pre-registered 64.02% threshold. Updating the
last vision blocks was already rejected by the preceding 500-sample pilot, so it was not expanded in
the 1,500-sample confirmation.

This is an architecture-selection experiment, not a paper test result. It used one seed (42), and no
PathMMU, PathVQA, or OmniMedVQA test split was accessed for selection.

## Architecture definitions

| Arm | Vision encoder | Projector | Language model |
|---|---|---|---|
| Base | unchanged | unchanged | unchanged |
| C0 | frozen | frozen | full-parameter training |
| L | frozen | frozen | LoRA r16 |
| A | frozen | fully trained | LoRA r16 |
| B2 | LoRA on blocks 30--31 | fully trained | LoRA r16 |
| B4 | LoRA on blocks 28--31 | fully trained | LoRA r16 |

## Phase 1: 500-sample rapid screen, all arms

All trained arms used the same 500 examples for 10 epochs, seed 42, eight A100 GPUs, per-device
batch 12, and global batch 96. `Train500` is accuracy on the fixed training probe. PathVQA uses a
balanced, image-unique 512-case validation panel and forced `Yes`/`No` next-token scoring, so its
coverage is 100% and it is unaffected by free-generation parsing. MMMU is the fixed non-medical
116-case dev retention panel, rescored with the corrected parser and a 4,096-token limit.

| Arm | Train500 | PathMMU val385 | PathVQA normal | PathVQA shuffle | PathVQA blank | Delta normal-shuffle | Delta normal-blank | MMMU dev116 | MMMU parsed | MMMU cap hits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 258/500 (51.60%) | 196/385 (50.91%) | 338/512 (66.02%) | 259/512 (50.59%) | 261/512 (50.98%) | 15.43 pp | 15.04 pp | 66/116 (56.90%) | 116/116 | 0/116 |
| C0 | 495/500 (99.00%) | 209/385 (54.29%) | 287/512 (56.05%) | 258/512 (50.39%) | 261/512 (50.98%) | 5.66 pp | 5.08 pp | 56/116 (48.28%) | 116/116 | 0/116 |
| L | 297/500 (59.40%) | 214/385 (55.58%) | 329/512 (64.26%) | 261/512 (50.98%) | 256/512 (50.00%) | 13.28 pp | 14.26 pp | 58/116 (50.00%) | 109/116 | 5/116 |
| A | 320/500 (64.00%) | 219/385 (56.88%) | 325/512 (63.48%) | 258/512 (50.39%) | 253/512 (49.41%) | 13.09 pp | 14.06 pp | 63/116 (54.31%) | 113/116 | 1/116 |
| B2 | 313/500 (62.60%) | 219/385 (56.88%) | 326/512 (63.67%) | 258/512 (50.39%) | 252/512 (49.22%) | 13.28 pp | 14.45 pp | 60/116 (51.72%) | 113/116 | 3/116 |
| B4 | 311/500 (62.20%) | 220/385 (57.14%) | 327/512 (63.87%) | 259/512 (50.59%) | 253/512 (49.41%) | 13.28 pp | 14.45 pp | 62/116 (53.45%) | 113/116 | 2/116 |

### Phase-1 training throughput

| Arm | Runtime | Samples/s | Final train loss |
|---|---:|---:|---:|
| C0 | 147.25 s | 33.955 | 0.2899 |
| L | 120.77 s | 41.401 | 1.1275 |
| A | 120.99 s | 41.325 | 1.1045 |
| B2 | 124.91 s | 40.030 | 1.1041 |
| B4 | 126.40 s | 39.558 | 1.1035 |

### Phase-1 free-generation visual-dependence diagnostic

This supplementary diagnostic used free generation rather than forced binary scoring. The Base
normal run was rerun with a 1,024-token limit after the earlier 192-token configuration proved too
short; the intervention runs used 4,096 tokens for Base and 1,024 for C0/L. None of the rows below
hit its generation cap.

| Arm | Normal accuracy | Shuffle accuracy | Blank accuracy | Parsed normal/shuffle/blank |
|---|---:|---:|---:|---:|
| Base | 328/512 (64.06%) | 270/512 (52.73%) | 287/512 (56.05%) | 512 / 512 / 512 |
| C0 | 300/512 (58.59%) | 266/512 (51.95%) | 287/512 (56.05%) | 510 / 512 / 509 |
| L | 324/512 (63.28%) | 262/512 (51.17%) | 257/512 (50.20%) | 512 / 512 / 512 |

The 500-sample screen therefore rejects C0 because it nearly memorizes the training probe while
losing PathVQA, MMMU, and most of the normal-minus-perturbed visual signal. B2/B4 add trainable
vision components without a resolved advantage over A. Only L and A were retained for the larger
follow-up.

## Phase 2: 1,500-sample L-versus-A follow-up

The follow-up used 1,500 fixed, image-unique SFT examples stratified to 375 examples per answer
letter, three epochs, seed 42, eight A100 GPUs, per-device batch 12, global batch 96, BF16, SDPA,
and a 1,024-token training cutoff. Checkpoints were saved every eight optimization steps, or about
half an epoch.

| Arm | Runtime | Samples/s | Steps/s | Final train loss |
|---|---:|---:|---:|---:|
| L | 112.24 s | 40.094 | 0.428 | 1.169709 |
| A | 111.28 s | 40.437 | 0.431 | 1.158356 |

### Complete PathMMU validation curve

All 12 checkpoints were evaluated with greedy free generation and a 1,024-token limit. No output
hit the limit. Only L step 8 had one non-conforming response; every other row had 385/385 correct
format.

| Arm | Step | Epoch | Correct | Accuracy | Format | Cap hits | Within 1 pp of arm best |
|---|---:|---:|---:|---:|---:|---:|---|
| L | 8 | 0.5 | 202/385 | 52.47% | 384/385 | 0 | No |
| L | 16 | 1.0 | 218/385 | 56.62% | 385/385 | 0 | Yes |
| L | 24 | 1.5 | 205/385 | 53.25% | 385/385 | 0 | No |
| L | 32 | 2.0 | 217/385 | 56.36% | 385/385 | 0 | Yes |
| L | 40 | 2.5 | 208/385 | 54.03% | 385/385 | 0 | No |
| L | 48 | 3.0 | 213/385 | 55.32% | 385/385 | 0 | No |
| A | 8 | 0.5 | 203/385 | 52.73% | 385/385 | 0 | No |
| A | 16 | 1.0 | 202/385 | 52.47% | 385/385 | 0 | No |
| A | 24 | 1.5 | 207/385 | 53.77% | 385/385 | 0 | No |
| A | 32 | 2.0 | 215/385 | 55.84% | 385/385 | 0 | No |
| A | 40 | 2.5 | 222/385 | 57.66% | 385/385 | 0 | Yes |
| A | 48 | 3.0 | 218/385 | 56.62% | 385/385 | 0 | No |

### Complete shortlisted-checkpoint retention table

`Train-Val gap` is Train500 minus PathMMU validation accuracy. Negative values here indicate that
the fixed training probe was harder than this validation panel; they are not evidence of leakage.
PathVQA deltas use format-neutral forced-binary scoring. MMMU used greedy free generation with the
4,096-token safety limit requested for truncation robustness.

| Checkpoint | Train500 | PathMMU val385 | Train-Val gap | PathVQA normal | Shuffle | Blank | Delta vision | Delta blank | MMMU dev116 | MMMU parsed | Cap hits | Mean/median tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L16 (epoch 1.0) | 255/500 (51.00%) | 218/385 (56.62%) | -5.62 pp | 335/512 (65.43%) | 261/512 (50.98%) | 259/512 (50.59%) | 14.45 pp | 14.84 pp | 63/116 (54.31%) | 115/116 (99.14%) | 1/116 | 203.18 / 89.5 |
| L32 (epoch 2.0) | 252/500 (50.40%) | 217/385 (56.36%) | -5.96 pp | 334/512 (65.23%) | 260/512 (50.78%) | 254/512 (49.61%) | 14.45 pp | 15.63 pp | 63/116 (54.31%) | 108/116 (93.10%) | 3/116 | 324.40 / 87.0 |
| A40 (epoch 2.5) | 272/500 (54.40%) | 222/385 (57.66%) | -3.26 pp | 326/512 (63.67%) | 259/512 (50.59%) | 257/512 (50.20%) | 13.09 pp | 13.48 pp | 63/116 (54.31%) | 112/116 (96.55%) | 1/116 | 229.85 / 88.5 |

A40's shuffle, blank, and MMMU cells are post-filter diagnostics run for a complete symmetric
table. They do not retroactively alter the pre-registered gate: A40 had already failed the 64.02%
PathVQA-normal threshold.

## Paired comparisons and final decision

| Comparison | Difference | 95% confidence interval | Exact McNemar p |
|---|---:|---:|---:|
| A40 minus L16, PathMMU val385 | +1.04 pp | [-2.84, +4.81] pp (image-cluster bootstrap) | 0.694 |
| L16 minus A40, PathVQA val512 | +1.76 pp | [-0.98, +4.30] pp (paired bootstrap) | 0.243 |
| L32 minus L16, PathVQA val512 | -0.20 pp | [-1.37, +0.98] pp (paired bootstrap) | 1.000 |

The pre-registered rule required A to exceed L by at least 2.0 pp on PathMMU while not losing more
than 1.0 pp on PathVQA, MMMU, or visual-dependence delta. A does not meet the PathMMU margin and is
1.76 pp below L16 on PathVQA. Its PathVQA point estimate is also 0.35 pp below the absolute 64.02%
retention threshold. All three shortlisted checkpoints tie on MMMU accuracy, while L16 has the best
MMMU parsing rate and the fewest cap hits among the two eligible L checkpoints.

Therefore:

1. Select architecture **L** for the formal 3,000-sample SFT.
2. Select **L16** as the best checkpoint of this screening run.
3. Keep vision and projector frozen during the initial rule-based RL run; train language LoRA only.
4. Re-run the same generalization and RL-readiness funnel on the formal SFT checkpoints before
   choosing the 1,000-sample rule-RL initialization.

## Reproducibility pointers

- Frozen selection contract: `protocol/visual_adaptation_followup_selection_contract_20260811.json`
- 1,500-sample training root:
  `/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/visual_adaptation_followup_n1500_e3_2seed_20260811`
- Complete follow-up evaluation root:
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/visual_adaptation_followup_n1500_e3_2seed_20260811`
- Machine-readable decision: `architecture_decision.json` under the evaluation root
- Frozen PathVQA panel SHA256:
  `b6b44fe1753799741674d7d08e0c2be5b9f8280a3bc249913d4ef52ad718e34f`
- Frozen MMMU panel SHA256:
  `367853d748f878f5edfb1a0259cefb73a0a9cf1fe9a24b3351c0c06695ee88f7`

The directory name contains the historical string `2seed`, but only seed 42 was run. Seed 43 was
not launched after the user selected a single-seed design.
