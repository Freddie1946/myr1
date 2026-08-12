# Formal A-SFT3000 projector mechanism isolation and test results

Date: 2026-08-11

## Question and frozen comparison

This run isolates whether the PathVQA decision-policy drift observed after the selected
L-SFT3000 run is caused specifically by freezing the multimodal projector, or by narrow-domain
PathMMU SFT itself. It is a mechanism control and does not reopen the already completed
architecture selection.

The A arm differs from L in exactly one trainability decision:

| Component | L | A |
|---|---|---|
| Vision encoder | frozen | frozen |
| Multimodal projector | frozen | fully trainable |
| Language model | LoRA r16 | LoRA r16 |

Both arms use the same 3,000 records, order, seed 42, three epochs, LR `5e-5`, BF16/SDPA,
eight GPUs, per-device batch 12 and global batch 96. A has 84,944,640 trainable parameters.
Its two-step eight-GPU smoke passed before formal launch. Formal A training completed all 96
steps in 213.13 seconds (42.23 samples/s). The trainability verifier found 196 language LoRA
modules, changes in all five projector tensors, no trainable vision block and finite nonzero
LoRA-B weights.

## Validation curves

| Step | Epoch | A PathMMU val | A PathVQA val yes/no | PathVQA Yes rate |
|---:|---:|---:|---:|---:|
| 16 | 0.5 | 209/385 = 54.29% | 330/512 = 64.45% | 43.36% |
| 32 | 1.0 | 205/385 = 53.25% | 329/512 = 64.26% | 37.70% |
| 48 | 1.5 | 208/385 = 54.03% | 325/512 = 63.48% | 35.74% |
| 64 | 2.0 | 209/385 = 54.29% | 324/512 = 63.28% | 34.38% |
| 80 | 2.5 | 210/385 = 54.55% | 324/512 = 63.28% | 34.77% |
| 96 | 3.0 | 213/385 = 55.32% | 321/512 = 62.70% | 34.57% |

The test-report checkpoint was frozen as step 96 because it had the highest PathMMU validation
accuracy. Test results were not used to select it.

At every predeclared same-step comparison, A's PathVQA point estimate was below L's:

| Step | L PathVQA val | A PathVQA val | A - L | McNemar p |
|---:|---:|---:|---:|---:|
| 16 | 65.23% | 64.45% | -0.78 pp | 0.665 |
| 64 | 64.26% | 63.28% | -0.98 pp | 0.383 |
| 80 | 64.45% | 63.28% | -1.17 pp | 0.263 |
| 96 | 63.67% | 62.70% | -0.98 pp | 0.383 |

The individual differences are not statistically significant, but their direction is consistent
and does not support projector adaptation as a retention fix.

## Retention and visual-dependence diagnostics at A step 96

| Metric | A step 96 | L step 96 | Interpretation |
|---|---:|---:|---|
| Train probe (500) | 55.80% | 56.00% | effectively equal |
| PathVQA normal | 62.70% | 63.67% | A lower |
| PathVQA shuffled image | 50.00% | 50.59% | chance-like control |
| PathVQA blank image | 49.41% | 49.41% | chance-like control |
| Normal - shuffle | 12.70 pp | 13.09 pp | A does not improve visual dependence |
| Normal - blank | 13.28 pp | 14.26 pp | A does not improve visual dependence |
| MMMU non-medical | 60/116 = 51.72% | 63/116 = 54.31% | A lower; one 4096-token cap |

## Complete requested tests

| Model | PathMMU test999 | PathVQA test yes/no3362 |
|---|---:|---:|
| Base | 486/999 = 48.65% | 2232/3362 = 66.39% |
| L step 80 (selected RL parent) | 566/999 = 56.66% | 2102/3362 = 62.52% |
| L step 96 | 566/999 = 56.66% | 2100/3362 = 62.46% |
| A step 96 | 571/999 = 57.16% | 2090/3362 = 62.17% |

A step 96 versus Base:

- PathMMU: +8.51 pp, image-cluster bootstrap 95% CI [+5.32, +11.69], McNemar
  `p=2.99e-7`.
- PathVQA: -4.22 pp, image-cluster bootstrap 95% CI [-5.32, -3.12], McNemar
  `p=2.90e-13`.

A step 96 versus selected L step 80:

- PathMMU: +0.50 pp, 95% CI [-1.52, +2.50], McNemar `p=0.704`.
- PathVQA: -0.36 pp, 95% CI [-1.17, +0.45], McNemar `p=0.441`.

All PathMMU generations were parsed (999/999), with no empty outputs and no 1024-token cap.
PathVQA used full-coverage forced-binary Yes/No logits, so its decline is not a parser or output
truncation artifact.

## Conclusion

The evidence supports the narrow-domain-specialization hypothesis rather than the L-only
hypothesis. Training the projector does not recover PathVQA retention, visual dependence or MMMU
retention. It yields only a nonsignificant +0.50 pp PathMMU test difference while losing a
nonsignificant 0.36 pp PathVQA relative to the selected L checkpoint. The main architecture
therefore remains L, and formal rule-RL starts from L step 80 with both Vision and Projector
frozen and language LoRA as the only trainable policy parameters.

## Artifacts

- A training: `/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/formal_a_sft3000_mechanism_20260811/formal`
- Evaluation: `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/formal_a_sft3000_mechanism_20260811`
- Complete machine-readable summary: `complete_summary.json` under the evaluation directory
- Paired statistics: `paired_stats/` under the evaluation directory
