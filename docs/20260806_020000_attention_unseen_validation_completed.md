# Layerwise attention unseen validation completed

Recorded at: 2026-08-06 02:00 CST

## Frozen validation flow

The calibration cases 03, 06, 15, and 21 were excluded before any new external
annotation or attention generation. The remaining 20 cases from the already
frozen 24-case visual-fidelity panel formed the unseen validation set.

Claude Sonnet 4.6 annotated all 20 original images through AIGCBest while blind
to every heatmap and computed metric. All 20 responses validated on the first
attempt. Total reported API usage was 27,583 prompt tokens and 1,990 completion
tokens.

Annotation applicability was:

| Evidence type | Cases | Spatial overlap applicable |
|---|---:|---|
| focal | 2 | yes |
| multifocal | 3 | yes |
| diffuse | 12 | no |
| not localizable | 3 | no |

This is an important correction to the four-case calibration set, where all
cases happened to be spatially boxable. The formal localization denominator
must therefore disclose an applicability flow and must not assign artificial
boxes to diffuse or non-localizable questions.

## Held-out spatial results

The five spatially evaluable cases were panel indices 01, 02, 09, 10, and 20.
The same GPT-4o Stage3 checkpoint, query definitions, top-10%-area metric, and
predefined early/middle/late bands were used without selecting a best layer.

Question/options-only attention results:

| Band | Top-area IoU | Pointing hit rate | ROI enrichment | Cases with enrichment > 1 |
|---|---:|---:|---:|---:|
| early 0-7 | 0.032 +/- 0.029 | 0.075 +/- 0.168 | 0.821 +/- 0.189 | 1/5 |
| middle 8-17 | 0.117 +/- 0.055 | 0.360 +/- 0.152 | 1.429 +/- 0.649 | 3/5 |
| late 18-27 | 0.111 +/- 0.055 | 0.460 +/- 0.261 | 1.346 +/- 0.605 | 4/5 |

The predefined middle/late trend therefore replicated directionally on unseen
spatial cases. It remains a small external-model-pseudo-label validation, not an
expert-ground-truth localization result.

## Confidence sensitivity

Restricting the pseudo-reference annotations to confidence at least 0.70 leaves
cases 01, 09, and 20:

| Band | Top-area IoU | Pointing hit rate | ROI enrichment |
|---|---:|---:|---:|
| early 0-7 | 0.032 | 0.125 | 0.763 |
| middle 8-17 | 0.087 | 0.367 | 1.076 |
| late 18-27 | 0.083 | 0.433 | 1.053 |

The direction persists but the effect is weaker. The larger all-five-case
enrichment is partly driven by case 02, whose annotation confidence is 0.60 and
whose middle/late attention alignment is unusually strong.

## Per-case visual audit

- Case 01: weak and inconsistent alignment with the three bilayered epithelial
  regions; many layers emphasize image boundaries or unrelated stroma.
- Case 02: clear middle/late concentration on central eosinophilic fibrinoid
  material; this is the strongest held-out spatial example.
- Case 09: middle/late layers partly align with the lower smooth-muscle bundles,
  but later layers also shift toward gland lumina and boundaries.
- Case 10: attention often remains near the central glomerulus instead of the
  peripheral proteinaceous casts. The pseudo annotation itself has low
  confidence (0.45), and this case should not be presented as a successful map.
- Case 20: middle/late layers show modest multifocal concentration over dense
  lymphocyte aggregates, mixed with edge and unrelated-tissue hotspots.

Only cases 01 and 10 were answered correctly by the evaluated model; cases 02,
09, and 20 were answered incorrectly. Raw question-conditioned attention does
not rank regions using answer logits, so overlap with a reference ROI is not
evidence that the region caused or supported the final answer. This distinction
must remain explicit in the reviewer response.

## Artifacts

- External annotations SHA-256:
  `a62d73d551b6b7a6482cc4cbae357a86643c3c0786d69debb602fd135e3891ea`
- Held-out attention JSON SHA-256:
  `2c4797c7d1eca578fbb39e707b5640791c14fa5d76de81e4e8c335fc780a246c`
- Held-out layer metrics SHA-256:
  `74d2520bc07023f6424fec916eb70ffa5db95974ab3247790a3c8e4e3f1c4570`
- Held-out metric curves SHA-256:
  `86aa7828b5294cb5e482300240e4f382b878c904bb880bf128d950fd844a403b`

Raw paths are recorded in the companion protocol manifest. The layer-analysis
tree is 264 MiB; the source attention tree is 98 MiB.

## Next causal-fidelity step

Do not choose a single layer from these results. A defensible next pilot is to
average the predeclared middle and late bands (layers 8-27), rank visual regions
with that answer-independent attention, and compare attention-guided deletion
and retention against area-matched random controls. The primary endpoint should
be generated answer change/accuracy, with A-D next-token probability retained
only as a secondary diagnostic. Mean-fill and blur perturbations should both be
reported so a result is not attributable to one masking artifact.
