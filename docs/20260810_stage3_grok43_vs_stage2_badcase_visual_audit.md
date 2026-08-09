# Grok 4.3 Stage3 versus Stage2: paired PathMMU bad-case audit

On the frozen PathMMU `test999`, Stage2 scores 609/999 (60.96%) and the selected Grok 4.3
Stage3 checkpoint scores 633/999 (63.36%). The point estimate therefore improves by 24 questions,
or 2.40 percentage points.

## Why the gain is not conventionally significant

- Both correct: 544
- Both wrong: 301
- Stage2 correct, Stage3 wrong: 65
- Stage2 wrong, Stage3 correct: 89
- Predictions changed: 200/999
- Exact paired McNemar two-sided p-value: 0.0635
- Image-cluster bootstrap Stage3 minus Stage2: +2.40 points, 95% CI [0.00, +4.82],
  two-sided p = 0.0540

The aggregate gain is real as a point estimate, but the paired evidence contains 65 regressions as
well as 89 improvements. Their difference (24) is not large enough, under either the exact paired
test or the image-cluster bootstrap, to cross the pre-specified conventional 0.05 threshold. This is
not a contradiction and is not caused by treating paired questions as independent. The correct
wording is **positive but borderline/inconclusive on PathMMU**, not “no improvement.”

The improvement is also not an answer-letter parsing artifact. Accuracy changes by target answer
are A +3.59 points, B -0.40, C +3.98, and D +2.42; predicted-letter counts remain close to balanced.

## Independent visual audit

Claude Sonnet 4.6 independently inspected four deterministically selected regressions and four
improvements, with model identities hidden as BEFORE and AFTER. All eight calls succeeded on the
first attempt and cost $0.054522.

| Direction | Cases | Independent finding |
|---|---:|---|
| Stage2 correct -> Grok Stage3 wrong | 4 | All four confirmed regressions caused by visual-feature misreads |
| Stage2 wrong -> Grok Stage3 correct | 4 | All four confirmed improvements; repaired 2 pathology-knowledge/reasoning errors and 2 visual-feature misreads |

Three of eight cases had some reference/image ambiguity, but the judge still found the paired
ordering clear. The audit therefore supports the mechanical paired score: Stage3 repairs genuine
errors and introduces genuine new visual errors. It does not indicate a parser, response-format, or
statistics bug. This eight-case audit is diagnostic rather than a population estimate.

## Interpretation

Grok 4.3 Stage3 has the strongest PathMMU point estimate among the trained checkpoints evaluated in
this run, and it is significantly better than the selected GPT-4o Stage3 checkpoint (633/999 versus
604/999; paired McNemar p = 0.0149; image-cluster bootstrap 95% CI [+0.72, +5.11]). Against Stage2,
however, the more defensible conclusion is a promising +2.40-point gain whose uncertainty still
touches zero. It should be reported with the effect size and confidence interval rather than promoted
to a definitive significant improvement.
