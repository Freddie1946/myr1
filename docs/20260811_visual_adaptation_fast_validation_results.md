# Visual-adaptation SFT rapid validation (2026-08-11)

## Scope

This is a rapid architecture-screening experiment, not a formal paper result. It used 500 fixed
SFT records for 10 epochs, seed 42, global batch 96 on eight A100 GPUs. No test split was accessed.

The compared arms were:

- Base: unchanged Qwen2.5-VL-7B-Instruct.
- C0: full-parameter LLM training; vision tower and multimodal projector frozen.
- L: LLM LoRA; vision tower and multimodal projector frozen.
- A: LLM LoRA; multimodal projector fully trained; vision tower frozen.
- B2: A plus LoRA on vision blocks 30--31.
- B4: A plus LoRA on vision blocks 28--31.

## Training throughput

| Arm | Samples/s | Runtime (s) | Train loss |
|---|---:|---:|---:|
| C0 | 33.955 | 147.25 | 0.2899 |
| L | 41.401 | 120.77 | 1.1275 |
| A | 41.325 | 120.99 | 1.1045 |
| B2 | 40.030 | 124.91 | 1.1041 |
| B4 | 39.558 | 126.40 | 1.1035 |

The selected per-device batch was 12 (global batch 96). Training-stage utilization reached roughly
80--100% SM, with the worst observed memory use about 74 GB on an 80 GB card. Lower utilization
during model loading, tokenization, and checkpoint saving was expected and was not treated as a
training-throughput defect.

## Frozen PathVQA validation panel

The panel contains 512 image-unique Yes/No cases, balanced 256/256, frozen before these model
outputs and disjoint from the earlier 128-case calibration panel. The metric compares the next-token
logits of the single-token verbalizers `Yes` and `No`; therefore coverage is exactly 100% and the
result is not confounded by free-generation parsing or truncation.

| Arm | Correct | Accuracy | Difference vs Base | Paired bootstrap 95% CI | McNemar p |
|---|---:|---:|---:|---:|---:|
| Base | 338/512 | 66.02% | -- | -- | -- |
| C0 | 287/512 | 56.05% | -9.96 pp | [-14.26, -5.66] pp | 1.35e-5 |
| L | 329/512 | 64.26% | -1.76 pp | [-4.10, +0.59] pp | 0.188 |
| A | 325/512 | 63.48% | -2.54 pp | [-5.47, +0.39] pp | 0.105 |
| B2 | 326/512 | 63.67% | -2.34 pp | [-5.27, +0.39] pp | 0.134 |
| B4 | 327/512 | 63.87% | -2.15 pp | [-5.08, +0.78] pp | 0.169 |

C0 also shifted the predicted-Yes rate to 25.98%, compared with the balanced target rate of 50%
and the Base rate of 39.45%. L/A/B2/B4 recovered it to 37.70%, 36.13%, 37.11%, and 36.91%,
respectively. L exceeded C0 by 8.20 pp (paired bootstrap 95% CI [+4.10, +12.30] pp,
McNemar p=0.000180). L exceeded A by only 0.78 pp (95% CI [-1.37, +2.93] pp,
McNemar p=0.597).

## PathMMU validation385

This was greedy free generation with a 1,024-token cap. Every model produced a non-empty,
EOS-terminated response for all 385 records; no response hit the generation cap.

| Arm | Correct | Accuracy | Difference vs Base | Image-cluster bootstrap 95% CI | McNemar p |
|---|---:|---:|---:|---:|---:|
| Base | 196/385 | 50.91% | -- | -- | -- |
| C0 | 209/385 | 54.29% | +3.38 pp | [-2.03, +8.82] pp | 0.255 |
| L | 214/385 | 55.58% | +4.68 pp | [+0.26, +9.28] pp | 0.0505 |
| A | 219/385 | 56.88% | +5.97 pp | [+1.81, +10.23] pp | 0.00954 |
| B2 | 219/385 | 56.88% | +5.97 pp | [+1.60, +10.39] pp | 0.0128 |
| B4 | 220/385 | 57.14% | +6.23 pp | [+2.01, +10.63] pp | 0.00968 |

L, A, B2, and B4 are statistically indistinguishable from one another on this panel. A exceeded L
by 1.30 pp, with a cluster-bootstrap 95% CI of [-2.60, +5.05] pp and McNemar p=0.583. B4 minus A
was only +0.26 pp, with a cluster-bootstrap 95% CI of [-1.81, +2.35] pp and McNemar p=1.0. The gains
of A/B2/B4 over C0 were about 2.6--2.9 pp, but their intervals also crossed zero.

## Interpretation

Under this rapid-training regime, C0 provides strong evidence of an in-domain/OOD trade-off rather
than an output-parser artifact: PathMMU improves modestly while format-neutral PathVQA drops by
about ten points. L recovers 8.20 of those 9.96 PathVQA points without updating either the projector
or vision tower. This identifies constraining the LLM update to LoRA, rather than projector training,
as the main supported mechanism for OOD preservation in this screen.

Projector training shows only a small, statistically unresolved point-estimate trade-off: compared
with L, A is +1.30 pp on PathMMU and -0.78 pp on PathVQA. Consequently, this experiment does not
support a claim that projector training is necessary, nor does it prove that the projector is
harmful. L is the simplest OOD-preserving candidate; A remains a reasonable candidate if the small
PathMMU advantage repeats at larger scale or across seeds. The present data also do not justify
adapting the last 2--4 vision blocks: B2/B4 add complexity without a measurable advantage over A.

These results establish a plausible failure mechanism for the fast 500-sample setup; they do not by
themselves prove that the historical 3,000-sample SFT checkpoint failed for exactly the same reason.
