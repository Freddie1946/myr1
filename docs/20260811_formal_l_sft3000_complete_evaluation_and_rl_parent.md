# Formal L-SFT3000: complete evaluation and rule-RL parent selection

Timestamp: 2026-08-11 Asia/Shanghai

## Outcome

The formal architecture is **L**: frozen vision encoder, frozen multimodal projector, and rank-16
LoRA on the language model. The 3,000-sample, seed-42 SFT completed all three epochs. The frozen
validation/OOD/readiness funnel selects **checkpoint step 80 (about epoch 2.5)** as the parent for
the 1,000-prompt rule-based RL run.

The selection did not use test results. PathMMU test999 and PathVQA test Yes/No are included below
as post-hoc reporting-only diagnostics and must not be described as untouched final tests.

## Training configuration and throughput

| Item | Value |
|---|---:|
| SFT pool | 3,000 PathMMU training records |
| Epochs / optimizer steps | 3 / 96 |
| GPUs | 8 x A100 80 GB |
| Per-device / global batch | 12 / 96 |
| Precision / attention | BF16 / SDPA |
| Vision / projector | frozen / frozen |
| Language model | LoRA r16 |
| Runtime | 204.36 s |
| Throughput | 44.04 samples/s; 0.470 steps/s |
| Final train loss | 1.090153 |

All eight GPUs reached approximately 100% SM during steady-state training, with no OOM or
non-finite loss/gradient. Note that step 16 is only about epoch 0.5 and therefore had not yet seen
every one of the 3,000 records; this was not used as an exclusion rule.

## Complete PathMMU validation curve

Greedy decoding used a 1,024-token safety limit. Every completion terminated below the cap and all
385 responses at every checkpoint had the required output format. Choice extraction was 384/385
at each checkpoint.

| Step | Approx. epoch | Correct | Accuracy | Distance from best | Pass `best - 1 pp` funnel |
|---:|---:|---:|---:|---:|---|
| 16 | 0.5 | 212/385 | 55.06% | 0.26 pp | Yes |
| 32 | 1.0 | 207/385 | 53.77% | 1.56 pp | No |
| 48 | 1.5 | 208/385 | 54.03% | 1.30 pp | No |
| 64 | 2.0 | 213/385 | **55.32%** | 0.00 pp | Yes |
| 80 | 2.5 | 210/385 | 54.55% | 0.78 pp | Yes |
| 96 | 3.0 | 212/385 | 55.06% | 0.26 pp | Yes; later fails PathVQA threshold |

The validation curve is a plateau with checkpoint churn, not a monotonic improvement curve. The
difference between steps 16 and 64 is one question, so validation accuracy alone is insufficient
for the RL-parent decision.

## Retention and visual-dependence diagnostics

`Train500` is a fixed training-pool probe, not the training loss. PathVQA is a balanced,
image-unique validation panel scored by forced `Yes`/`No` next-token comparison, which gives 100%
coverage and removes free-generation parsing as a confound. MMMU uses the frozen 116-case
non-medical dev panel and a 4,096-token generation ceiling. All MMMU prediction files were also
rescored with the same current parser; this produced zero parser changes.

| Checkpoint | Train500 | PathMMU val385 | Train-Val gap | PathVQA normal | Shuffle | Blank | Normal-Shuffle | Normal-Blank | MMMU dev116 | MMMU parsed | Cap hits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 258/500 (51.60%) | 196/385 (50.91%) | +0.69 pp | 338/512 (66.02%) | 259/512 (50.59%) | 261/512 (50.98%) | 15.43 pp | 15.04 pp | 66/116 (56.90%) | 116/116 | 0 |
| step16 | 258/500 (51.60%) | 212/385 (55.06%) | -3.46 pp | 334/512 (65.23%) | 260/512 (50.78%) | 258/512 (50.39%) | 14.45 pp | 14.84 pp | 65/116 (56.03%) | 114/116 | 1 |
| step64 | 275/500 (55.00%) | 213/385 (55.32%) | -0.32 pp | 329/512 (64.26%) | 260/512 (50.78%) | 254/512 (49.61%) | 13.48 pp | 14.65 pp | 58/116 (50.00%) | 111/116 | 3 |
| **step80** | 277/500 (55.40%) | 210/385 (54.55%) | +0.85 pp | 330/512 (64.45%) | 258/512 (50.39%) | 251/512 (49.02%) | 14.06 pp | 15.43 pp | 62/116 (53.45%) | 111/116 | 4 |
| step96 | 280/500 (56.00%) | 212/385 (55.06%) | +0.94 pp | 326/512 (63.67%) | 259/512 (50.59%) | 253/512 (49.41%) | 13.09 pp | 14.26 pp | 63/116 (54.31%) | 112/116 | 3 |

Step 96 is below the frozen 64.02% PathVQA-normal floor and is therefore excluded before
RL-readiness. Step 64 passes but has the weakest MMMU point estimate. Step 80 retains a healthier
PathVQA visual-dependence delta and MMMU point estimate than step 64 while remaining within one
percentage point of the PathMMU validation best.

## RL-readiness: 300 prompts x 8 rollouts

Each passing checkpoint used the same fixed 300-prompt panel, eight rollouts per prompt,
temperature 1.0, top-p 1.0, maximum 192 tokens and deterministic per-prompt seeds. `Mixed` means a
group contained at least one correct and one incorrect rollout, which provides nonzero within-group
GRPO signal.

| Checkpoint | Mean accuracy reward | Mean group reward std | Mixed groups | All-zero | All-one | Mean format reward | Cap hits / 2,400 |
|---|---:|---:|---:|---:|---:|---:|---:|
| step16 | 49.42% | 0.2904 | 69.33% | 16.00% | 14.67% | 96.38% | 13 |
| step64 | 47.75% | 0.3288 | 77.33% | 11.67% | 11.00% | 98.08% | 10 |
| **step80** | 48.75% | **0.3435** | **80.33%** | **10.33%** | **9.33%** | **98.83%** | **8** |

Step 80 has the densest and most balanced RL signal: the highest mixed-group fraction and reward
standard deviation, the lowest all-zero/all-one fractions, the best format rate and the fewest cap
hits. This is why it is selected over the one-question PathMMU-validation leader step 64.

## PathMMU test999 (reporting only)

| Model | Correct | Accuracy | Difference vs Base | Image-cluster bootstrap 95% CI | McNemar p | Base-only / model-only flips | Format | Choice extraction | Cap hits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 486/999 | 48.65% | -- | -- | -- | -- | 973/999 | 989/999 | 0 |
| step16 | 522/999 | 52.25% | +3.60 pp | [+0.81, +6.32] pp | 0.0131 | 82 / 118 | 999/999 | 998/999 | 0 |
| step64 | 575/999 | **57.56%** | +8.91 pp | [+5.73, +12.12] pp | 6.18e-8 | 90 / 179 | 999/999 | 998/999 | 0 |
| **step80** | 566/999 | 56.66% | +8.01 pp | [+4.81, +11.21] pp | 1.29e-6 | 95 / 175 | 999/999 | 998/999 | 0 |
| step96 | 566/999 | 56.66% | +8.01 pp | [+4.78, +11.25] pp | 1.85e-6 | 99 / 179 | 999/999 | 999/999 | 0 |

The selected step 80 gains 175 questions that Base missed and loses 95 questions that Base got
right, for a real net gain of 80. This is incompatible with the explanation that the apparent
improvement is only Base guessing incorrectly or formatting differently. Step 64 has the highest
test point estimate, but test is not permitted to overrule the frozen validation/OOD/readiness
selection.

## PathVQA test Yes/No subset, 3,362 questions (reporting only)

This is the complete official test Yes/No subset, not a sample. Forced-binary scoring gives 100%
coverage, so the decline below cannot be caused by output truncation or free-text parsing. The
target set is 54.02% `Yes`.

| Model | Correct | Accuracy | Difference vs Base | Image-cluster bootstrap 95% CI | McNemar p | Base-only / model-only flips | Predicted Yes | Mean target margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 2,232/3,362 | **66.39%** | -- | -- | -- | -- | 41.64% | 0.9469 |
| step16 | 2,200/3,362 | 65.44% | -0.95 pp | [-1.56, -0.35] pp | 0.00381 | 74 / 42 | 41.94% | 0.8134 |
| step64 | 2,108/3,362 | 62.70% | -3.69 pp | [-4.61, -2.73] pp | 1.79e-13 | 206 / 82 | 36.64% | 0.6620 |
| **step80** | 2,102/3,362 | 62.52% | -3.87 pp | [-4.83, -2.89] pp | 1.03e-13 | 220 / 90 | 35.93% | 0.6530 |
| step96 | 2,100/3,362 | 62.46% | -3.93 pp | [-4.89, -2.94] pp | 2.24e-14 | 218 / 86 | 36.11% | 0.6494 |

This is a genuine OOD decision-distribution shift, not a parser artifact. Later SFT checkpoints
become increasingly biased toward `No`, even though the test has a `Yes` majority, and their mean
target margin steadily falls. Step 16 largely preserves Base's answer distribution but gives much
less mature in-domain performance and weaker RL-readiness. The selected step 80 therefore accepts
a documented PathVQA-retention cost in exchange for stronger PathMMU and rule-RL readiness; the
subsequent RL evaluation must check whether this cost recovers or worsens.

## Short bad-case interpretation

- In-domain gains are substantive: Base-to-step80 has 175 improvements versus 95 regressions on
  PathMMU test999. Previously inspected gains include glomerular-loop collapse, cartilage cellular
  distribution, immunohistochemical staining and nuclear morphology.
- In-domain regressions also exist and are not dominated by formatting: representative failures
  include lymphocyte-arrow identification, ovarian atresia, muscle-atrophy arrows and fatty
  infiltration. The validation plateau reflects turnover between such cases.
- PathVQA degradation is broader and directional: step80 produces 220 Base-correct/model-wrong
  flips but only 90 reverse flips, alongside a 5.71-pp drop in predicted-`Yes` rate. This supports
  task/output-distribution specialization rather than lost parsing coverage.
- MMMU is only a 116-case sanity panel. Step80 is four questions below the uniformly rescored Base,
  and four generations hit the 4,096-token cap; this suggests possible general-capability drift but
  is too small to support a precise magnitude claim.

## Reproducibility and scientific boundaries

- Formal SFT root:
  `/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/formal_selected_sft3000_20260811`
- Evaluation root:
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/formal_selected_sft3000_20260811`
- Machine-readable parent selection:
  `protocol/formal_selected_sft3000_rl_parent_step080_20260811.json`
- Selected merged RL parent:
  `/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/formal_selected_rule_rl1000_20260811/parents/sft_step080_merged`
- PathVQA test was evaluated at `batch_size=64` after batch 128 and 96 correctly failed controlled
  OOM smokes. Batch 64 peaked near 71.4 GB with about 91% SM on a single GPU; four checkpoints were
  each split into two exact, disjoint shards to occupy all eight GPUs. Merge validation proves exact
  coverage of source indices 0--3,361 with no duplicate or missing record.
- PathMMU test999 and PathVQA test are post-hoc reporting-only diagnostics and were not inputs to
  step80 selection. MMMU uses its public dev panel because its official test labels are not locally
  available.
