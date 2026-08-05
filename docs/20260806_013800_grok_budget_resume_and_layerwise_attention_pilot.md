# Grok budget resume and layerwise attention pilot

Timestamp: 2026-08-06 01:38 CST

## Grok Stage3 status

- Run: `grok43_full3epoch_seed42_20260805_attempt02`
- Judge route: AIGCBest `grok-4.3`
- The local protective budget ceiling was raised from USD 50 to USD 100 without
  changing the frozen physical-request ceiling of 12,360.
- Spend and completed-request history were preserved during the amendment.
- Segment 01 restored the complete 102 GiB DeepSpeed checkpoint at step 200,
  including all eight optimizer, model-state, and RNG-state shards.
- New judge calls and optimizer steps 201 through 217 were observed after the
  restore. At the last audit, committed spend was USD 9.81205875 and 2,097
  unique requests were complete. This is a live progress observation, not a
  completion claim.

## Attention question definition

The four calibration cases remain panel indices 03, 06, 15, and 21. No primary
layer is selected from these cases.

Three query definitions are compared layer by layer:

1. `last_query`: the final prompt query position, averaged across heads.
2. `blog_text_query`: every query position after the image, including formatting
   and generation-prompt tokens, averaged across heads and queries.
3. `question_options_query`: only the token positions belonging to the question
   and answer options. The fixed output-format suffix, assistant prefix, special
   tokens, and whitespace-only tokens are excluded before averaging.

For case 03, the new selector retained 43 tokens and decoded exactly to the
question plus options. This boundary was checked with the real checkpoint
tokenizer. It did not include the fixed `Return only one option letter...`
instruction.

All maps are raw text-to-visual attention summaries. They are not causal
explanations and do not use A/B/C/D logits to rank regions. Each layer map is
normalized over visual tokens. The overlap audit selects the top 10% of visual
area and compares it with Claude Sonnet 4.6 pseudo-reference boxes created
without seeing the heatmaps. These boxes are not pathology-expert ground truth.

## Predefined layer-band results

The descriptive bands were fixed as early 0-7, middle 8-17, and late 18-27.
Values are four-case means; the `+/-` value is the sample standard deviation
across the four case-level band means.

| Query definition | Band | Top-area IoU | Pointing hit rate | ROI enrichment | Cases with enrichment > 1 |
|---|---|---:|---:|---:|---:|
| last query | early 0-7 | 0.039 +/- 0.045 | 0.219 +/- 0.438 | 0.714 +/- 0.478 | 1/4 |
| last query | middle 8-17 | 0.057 +/- 0.043 | 0.100 +/- 0.200 | 0.967 +/- 0.555 | 2/4 |
| last query | late 18-27 | 0.096 +/- 0.040 | 0.350 +/- 0.252 | 1.429 +/- 0.170 | 4/4 |
| all post-image text | early 0-7 | 0.047 +/- 0.049 | 0.250 +/- 0.500 | 0.887 +/- 0.264 | 1/4 |
| all post-image text | middle 8-17 | 0.120 +/- 0.085 | 0.350 +/- 0.208 | 1.404 +/- 0.350 | 4/4 |
| all post-image text | late 18-27 | 0.091 +/- 0.038 | 0.450 +/- 0.387 | 1.381 +/- 0.294 | 4/4 |
| question/options only | early 0-7 | 0.053 +/- 0.049 | 0.188 +/- 0.375 | 0.905 +/- 0.243 | 1/4 |
| question/options only | middle 8-17 | 0.126 +/- 0.096 | 0.375 +/- 0.206 | 1.420 +/- 0.378 | 4/4 |
| question/options only | late 18-27 | 0.102 +/- 0.047 | 0.525 +/- 0.386 | 1.392 +/- 0.340 | 3/4 |

The question/options-only result supports a limited descriptive statement:
task-relevant spatial concentration appears more often in middle and late
layers than in early layers. The large case-to-case variance does not support a
universal best layer.

## Visual audit

- Case 03 remains diffuse. Layers 12-15 and 20-23 overlap more of the widespread
  membrane-stained tissue, but boundary hotspots remain.
- Case 06 develops a clear focus on the red-cell-containing vessel in middle to
  late layers, strongest around layers 19-22. It covers only part of the relevant
  vascular evidence.
- Case 15 shifts strongly toward the two large vessels around layers 14-22, but
  some layers also emphasize blank tissue or slide-boundary artifacts.
- Case 21 is different: early and middle layers more often align with the
  superficial dermal band, while several late layers drift toward the lower
  image edge.

These observations agree with the numerical heterogeneity and are the reason
not to choose layer 14, 21, or any other single layer post hoc.

## Artifacts and integrity

- Metrics and full provenance:
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/attention_layer_analysis_question_options_20260806/layer_metrics.json`
  (`ab406234c489a8750dd695ad1a9e2769162b51efd13684afa275150f8e1c2cd3`)
- Metric curves:
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/attention_layer_analysis_question_options_20260806/layer_metric_curves.png`
  (`9b283d04faf5b855005e284818828b38f822a951330460cc2c4c8a6492d2951c`)
- Case 03 attention JSON:
  `06736b02b300c7384dfb003496bdf337aa5b3f0f35fbab26eae678be745f79da`
- Cases 06/15/21 attention JSON:
  `47499955ea605626871052b3358a450923f80516d25d8e32e843e8cfae8c84ec`

The layer-analysis directory is 194 MiB and contains shared-scale and
per-layer-scale mosaics for all three query definitions and all four cases.

## Next experiment boundary

The four cases are now explicitly a calibration pilot. Any method or layer-band
choice informed by them must be evaluated on a separate, previously unseen
case set. Before a formal reviewer-facing localization claim, the pseudo boxes
must also be independently checked by a pathology expert when one becomes
available.
