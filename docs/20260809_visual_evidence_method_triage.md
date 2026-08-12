# Visual evidence: current proof chain and method triage

## Target claim

The target is not that Stage3 is better than pretraining. The target is the
narrower, falsifiable statement that the model's answer uses image-specific,
question-relevant evidence. Three claims must remain separate:

1. **Image dependence:** changing the image changes the answer score.
2. **Image-specific decision evidence:** the correctly paired image supports the
   ground-truth option more than matched wrong-case images, and the induced
   internal state causally recovers that option.
3. **Spatial localization:** a reported heatmap identifies the tissue region
   that supports the option.

Evidence for (1) does not automatically prove (2), and neither proves (3).

## Current evidence matrix

| Experiment | Cases / models | Status | What it supports |
|---|---|---|---|
| Appearance-matched paired-image counterfactual | 96 / 6 | Complete; positive for 5/6 models | Image-specific decision evidence |
| Exact-pixel patch shuffle, 4x4/8x8/16x16 | 96 / GPT-4o and Grok Stage3 | Complete; modest 4x4 effect | Some dependence on coarse spatial organization |
| Geometry-matched residual activation patching | 96 / GPT-4o and Grok Stage3 | Complete; layers 20/24 positive | Image-induced information causally enters the answer state |
| Equal-norm permuted activation directions | 96 / GPT-4o and Grok Stage3 | Complete; correct direction wins at layers 20/24 | Recovery is direction-specific, not generic vector injection |
| Raw attention and option-conditioned gradient x attention | 96 / 6, eight layers | Complete; strict confirmation negative | These heatmaps are not faithful enough for a localization claim |
| Option-conditioned black-box RISE | 96 / GPT-4o and Grok Stage3 | Complete; strongly positive in deletion and retention | Localized, option-specific spatial decision evidence |
| External boxes and pathologist review | Frozen subset / final models | Pending | Spatial face validity and later expert validation |

## Method selection after the attention result

### 1. Black-box, option-conditioned RISE — highest immediate priority

The same random soft masks score all four A/B/C/D margins. The ground-truth
option map is generated with blurred replacement but validated with hard
mean-fill deletion and retention, against low-saliency and five area/shape-
matched shifted controls. This avoids selecting a layer or differentiating
through the model. A positive result requires both deletion and retention; an
effect that merely suppresses wrong options does not count as support.

This follows the perturbation logic of [RISE](https://cse.iitkgp.ac.in/~adas/papers/BMVC_2018_RISE.pdf)
while adding option selectivity, cross-perturbation validation, and matched
controls for the present multiple-choice setting.

The 96-case confirmation is now complete. GPT-4o high-region advantages are
`+0.2407` for deletion and `+0.2097` for retention; Grok advantages are
`+0.2205` and `+0.2241`. All four confidence intervals exclude zero and all
four sign-flip tests remain significant after Holm correction over the two
models and two directions (`p=0.00004`). The high region also outperforms the
low-saliency region in all four comparisons. This clears the predeclared RISE
spatial-decision gate but does not replace independent lesion annotations.

### 2. Component-level causal tracing — next non-spatial confirmation

At layers 20 and 24, separately restore the clean-image contribution of the
self-attention output, MLP output, and residual state in a geometry-matched
wrong-image run. Use the same ground-truth option margin, opposite direction,
and equal-norm permuted directions. This can identify whether cross-modal
evidence is aggregated by attention, transformed by the MLP, or only visible in
the combined residual stream.

This is motivated by [Fine-grained Cross-modal Causal Tracing](https://arxiv.org/abs/2511.05923),
which evaluates visual/text token categories and MHSA, FFN, and hidden-state
components across layers. We retain appearance-matched real images instead of
Gaussian corruption to reduce out-of-distribution confounding.

This experiment strengthens the *pathway* claim, not the spatial localization
claim.

### 3. Head-level sensitivity — conditional priority

[CausalLens](https://openaccess.thecvf.com/content/CVPR2026/html/Ji_CausalLens_Sensitivity-Guided_Multi-Head_Causal_Intervention_for_Hallucination_Mitigation_in_Large_CVPR_2026_paper.html)
suggests measuring causal sensitivity at individual attention heads. A limited
head ablation/patching study at the already identified layers 20 and 24 is
useful only if component-level tracing attributes a substantial effect to MHSA.
Otherwise a full head sweep would add cost without addressing the failed
spatial maps.

### 4. Region boxes and saliency mass — requires independent annotations

For a model-blind subset, an external vision model can first return image-space
boxes or polygons; a pathologist can later accept, edit, or reject them. Report
continuous saliency mass inside the verified region, point-hit, soft-IoU, and
area-normalized enrichment rather than only thresholded grid IoU. The box must
not be generated from the tested heatmap.

This is aligned with the region-supervised evaluation in
[Saliency-R1](https://openaccess.thecvf.com/content/CVPR2026/papers/Gong_Saliency-R1_Enforcing_Interpretable_and_Faithful_Vision-language_Reasoning_via_Saliency-map_Alignment_CVPR_2026_paper.pdf).
External-model boxes are provisional until expert review and must be described
as such.

### 5. Geometric and semantic invariance — supportive behavioral evidence

Apply label-preserving flips/rotations only where pathology semantics permit,
and compare them with evidence-removing or mismatched-image transformations.
The answer margin should be stable under valid geometry changes but fall when
task evidence is removed. This is inspired by the geometric and semantic
invariance distinction in [VISE](https://arxiv.org/abs/2606.27373).

Because orientation, arrows, scale bars, and multi-panel layouts can be
semantically meaningful, transformations must be case-screened or restricted;
unconditional augmentation would create invalid counterfactuals.

## Reporting rule

- RISE passes the predeclared bidirectional test. If it later agrees with
  independent verified regions, report spatial evidence as quantitative but
  still limited by annotation quality.
- If RISE fails, retain the paired-image and activation-patching results as
  evidence that the model uses image-specific information, and explicitly
  withdraw the claim that its attention heatmap localizes the reason.
- Do not select a favorable layer, head, option, or case after seeing results
  without labeling it exploratory and confirming it on a new frozen panel.
