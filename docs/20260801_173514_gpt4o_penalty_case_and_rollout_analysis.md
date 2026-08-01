# GPT-4o penalty case and training-rollout analysis

Timestamp: 2026-08-01 17:35:14 +0800

## Scope and reproducibility

This is a strictly offline development analysis of the matched penalty-0.3, 0.4 and recovered-0.5
100-step pilots. It made zero network calls, zero paid calls and did not access PathMMU test999.
`scripts/analyze_stage3_penalty_cases.py` fails closed unless all three training audits align by
rank, call index, item index, record, question, reference answer and image hash, and unless all
validation sources align by index, image, question and reference answer.

The generated full case package is outside Git at:

```text
/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/stage3_process_grpo/
  gpt4o_penalty_case_analysis_20260801/case_analysis.json
```

Its SHA-256 is
`b329b645f6c7c38dcee02b308da2b999e3e5c77becf08ee1e08f200b33e021f1`. It contains all 34
validation disagreements, all three completions for each case, training-event summaries and exact
source-artifact hashes. Raw generated outputs remain outside Git as required by repository policy.

## Training rollout retention

The training rollouts were preserved locally. Each successful arm contains 800 unique candidates
from 200 training records with four generations per record. Accuracy, format and process audit
records each contain the full candidate completion, producing 2,400 serialized audit rows per arm.
Each row also retains the question, reference answer, image path and SHA-256, rank, call index,
record index and reward. The Judge cache retains the exact served model, six event booleans,
evidence spans, request, response, score components and cost record. The stopped original 0.5 arm
separately retains its 792 completed candidates and is not merged with fresh01.

These files are local run artifacts, not model-weight tensors. They were not added to Git or to a
checkpoint upload by this analysis. The reward-audit payload is only about 5.2 MB per completed
arm, so a future sanitized remote backup is practical, but raw Judge caches should not be uploaded
blindly because they include request/response metadata.

## Training rollout comparison

All 800 training slots align across arms, but sampled candidates diverge quickly:

| Comparison | Exact completion | Same final choice |
| --- | ---: | ---: |
| 0.3 vs 0.4 | 13/800 | 564/800 |
| 0.4 vs 0.5 | 34/800 | 582/800 |
| 0.3 vs 0.5 | 23/800 | 588/800 |

Aggregate candidate behavior was:

| Measure | 0.3 | 0.4 | 0.5 |
| --- | ---: | ---: | ---: |
| Answer accuracy | 68.625% | 69.000% | 68.375% |
| Format accuracy | 100.000% | 100.000% | 99.875% |
| Image-feature event | 94.375% | 94.994% | 93.875% |
| Option-elimination event | 78.500% | 78.223% | 77.125% |
| Medical-knowledge event | 94.125% | 93.617% | 94.250% |
| Histological-definition error | 35.125% | 34.293% | 37.250% |
| Logical contradiction | 5.500% | 4.631% | 3.750% |
| Incorrect/outdated criterion | 5.250% | 6.383% | 5.625% |

From 0.4 to 0.5, 29 aligned slots lost a contradiction flag while 22 gained one, a net reduction
of seven. In contrast, 130 slots gained a histological-definition-error flag while 106 lost one, a
net increase of 24. Image-feature and option-elimination events decreased by nine and eight net
slots respectively; medical-knowledge support increased by five. Thus, the larger penalty shows a
small reduction in explicit contradiction but no general improvement in medical or image-grounded
process quality.

The raw process-reward means (0.8817, 0.8446 and 0.7984) are not quality-comparable because changing
the coefficient mechanically changes the scoring scale.

## Validation disagreement structure

The models agree in correctness on 351/385 cases (91.17%); only 34 cases differ.

| Pattern in arm order 0.3/0.4/0.5 | Count | Indices |
| --- | ---: | --- |
| 001 | 7 | 104, 170, 216, 223, 286, 335, 347 |
| 011 | 8 | 25, 106, 129, 142, 253, 309, 336, 362 |
| 010 | 3 | 9, 217, 258 |
| 110 | 8 | 92, 133, 204, 279, 289, 313, 369, 372 |
| 101 | 7 | 148, 187, 267, 291, 298, 307, 367 |
| 100 | 1 | 34 |

For five of the seven 0.5-only wins, 0.3 and 0.4 produced exactly the same completion. For six of
the eight 0.5-only losses, 0.3 and 0.4 were also exactly identical. Conversely, 0.3 and 0.5 were
identical in eight of the ten cases where 0.4 alone changed correctness. This clustered behavior is
more consistent with each short run entering a few different local answer modes than with a smooth,
monotonic coefficient effect.

## Manual case audit

This is a model-output audit with targeted image inspection, not a pathologist reannotation. The
following labels are provisional and cases with contextual ambiguity should be reviewed by a domain
expert before entering a paper.

### Plausible 0.5-only improvements

- `104`: 0.5 recognizes small, uniform spindle-cell nuclei without prominent nucleoli; lower arms
  call them hyperchromatic and irregular.
- `216`: 0.5 correctly distinguishes lymphocytes from basophils using nuclear-to-cytoplasmic ratio
  and granules.
- `223`: 0.5 correctly links pink/lavender cytoplasm to eosin uptake rather than inclusions.
- `335`: 0.5 identifies segmented nuclei as neutrophils.
- `347`: 0.5 identifies a closely packed granulosa-cell layer around the oocyte.

The other two 0.5-only answer gains have rationale concerns. At `170`, 0.5 selects the visibly dense
staining at `vg` but calls it an ovarian granulosa cell, which is not the reference context. At
`286`, it gets the paler keratinocyte option but adds an unsupported koilocyte/HPV interpretation.

### Plausible 0.5 regressions relative to 0.4

- `92`: the 0.5 reasoning itself says “mesothelial windows” but ends on cytoplasmic inclusion
  bodies, a direct reasoning-to-answer mapping failure.
- `133`: 0.5 treats the eosinophilic content in region C as collagen rather than keratin debris.
- `279`, `289`, `313`, `369`, `372`: 0.5 changes the relevant diagnosis, cellularity, tissue-change
  or organization feature away from the reference-supported option.
- `9`, `217`, `258`: 0.4 alone gives the reference-supported morphology, neutrophil identity or
  uniform-depth organization; both 0.3 and 0.5 take the competing option.

Case `204` should not be counted as a clean 0.5 process regression. Both lower arms explicitly say
viral inclusion bodies are present and then select “no visible viral inclusion bodies,” receiving
answer credit through a reasoning-answer contradiction. The 0.5 completion is internally aligned
but disagrees with the reference answer. This is an answer-only scoring artifact and a candidate for
expert review.

### Higher-coefficient shared gains and isolated 0.4 behavior

The 0.4 and 0.5 arms both plausibly improve over 0.3 at `25`, `106`, `129`, `309` and `362`.
However, shared correct answers at `142`, `253` and `336` retain rationale concerns: diagnostic
context is thin at `142`; the `253` rationale describes collagen-rich/fibrous tissue while selecting
myxoid stroma; and `336` treats extravasated red cells as sufficient evidence of angiogenesis.

The seven 0.4-only losses are `148`, `187`, `267`, `291`, `298`, `307` and `367`. Four look like
ordinary local answer regressions. Three require caution: all arms use problematic language about
nuclear loss in non-keratinized epithelium at `267`; `298` is a vague physiological interpretation;
and “chicken-wire” versus “delicately branching” vessels at `307` has overlapping semantics.

The single 0.3-only win at `34` is not a clean reasoning win: its rationale says the finding is not
consistent with thickening/reduplication and then selects thickening/reduplication. It is another
answer-only correctness artifact.

## Conclusion for coefficient selection

The case audit does not show that 0.5 produces uniformly better reasoning. Its highest validation
point estimate combines genuine-looking morphology/knowledge corrections, several clear
regressions, and answer-correct/rationale-wrong cases. The training rollout audit likewise shows a
tradeoff: fewer explicit contradictions but more histological-definition errors.

If one coefficient must be chosen before further evidence, 0.4 is the more conservative primary
default: it has the best sampled training accuracy, the lowest observed histological-definition
error rate, near-best validation accuracy, and avoids selecting the largest validation point
estimate after looking at the same split. Penalties 0.3 and 0.5 should remain disclosed sensitivity
arms. This is a recommendation, not an authorization or frozen formal-run decision.
