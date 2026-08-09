# GPT-4o Stage3 versus Stage2: paired PathMMU bad-case audit

This audit addresses an apparent conflict between the remembered Stage3 improvement and the
current frozen PathMMU `test999` result. On `test999`, Stage2 scores 609/999 (60.961%) and the
selected GPT-4o Stage3 checkpoint scores 604/999 (60.460%): the point estimate is five questions
lower, not higher.

## Full paired accounting

- Both correct: 549
- Both wrong: 335
- Stage2 correct, Stage3 wrong: 60
- Stage2 wrong, Stage3 correct: 55
- Predictions changed: 160/999
- Exact paired McNemar two-sided p-value: 0.7093
- Image-cluster bootstrap Stage3 minus Stage2: -0.50 percentage points, 95% CI
  [-2.64, +1.65]

The loss is spread across target letters A/B/C/D (-1/-2/-1/-1 correct questions), so it is not a
single answer-letter parsing artifact.

## Independent visual audit

Claude Sonnet 4.6 independently inspected the image, question, reference answer, and the two
anonymous model responses for four deterministically selected regressions and four improvements.
The audit cost $0.052983 under a $0.15 hard cap; all eight requests succeeded on their first attempt.

| Direction | Cases | Independent result |
|---|---:|---|
| Stage2 correct -> Stage3 wrong | 4 | All four confirmed as regressions: 2 visual-feature misreads, 1 pathology-knowledge error, 1 option-mapping error |
| Stage2 wrong -> Stage3 correct | 4 | All four confirmed as improvements: Stage3 repaired 2 visual-feature misreads, 1 pathology-knowledge error, 1 option-mapping error |

One case in each direction had mild reference/image ambiguity, but the judge still found the before
versus after ordering clear. This small visual audit is diagnostic rather than a population estimate.

## Interpretation

Stage3 changes individual decisions in both directions. It repairs genuine visual, medical, and
answer-mapping errors, but introduces similar categories of error elsewhere. The near-zero aggregate
change and non-significant paired tests therefore agree with the observed cases; the finding is not
caused by the statistical method hiding an accuracy increase.

The remembered improvement most likely refers to checkpoint-1000 on `validation_0385` (64.156%) or
the manuscript's older 500-sample result. Neither is interchangeable with the current image-disjoint
`test999`.

The same-split Stage2 validation run is now complete:

- Stage2: 236/385 (61.299%)
- selected Stage3 checkpoint-1000: 247/385 (64.156%)
- paired flips: 33 improvements and 22 regressions, net +11
- exact paired McNemar p = 0.1770
- image-cluster bootstrap difference: +2.86 points, 95% CI [-0.78, +6.67], p = 0.1402

Thus the validation point estimate really did improve, but its uncertainty includes zero. Moreover,
checkpoint-1000 was selected on this same validation split, so the validation gain is selection-biased
and should not be presented as independent confirmatory evidence. The independent `test999` result did
not reproduce it. Stage3 checkpoints 500 and 1500 are now being evaluated on `test999` to determine
whether that conclusion is specific to the selected checkpoint or consistent across epochs.
