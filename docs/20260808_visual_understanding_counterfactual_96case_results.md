# 96-case paired-image visual-understanding confirmation

## Question and claim boundary

This experiment asks whether a model's correct-option score benefits from the
image paired with the current PathMMU question, rather than merely from seeing
some pathology-looking image. It is a behavioral test of image-specific,
task-relevant visual dependence. It does **not** by itself prove human-like
pathology understanding, expert-grade localization, or that an attention map is
a faithful explanation.

## Frozen design

- Panel: 96 PathMMU validation cases, balanced at 24 cases per answer position,
  frozen without using any model outputs.
- Panel SHA-256:
  `c1222d9ff2d3a1b038a744b2f1c1addd18e777de3fb4f31c09107acef7e52931`.
- Counterfactual manifest SHA-256:
  `69bbe60cc057266448cb71bcc0bd6ff52281b9bd298995a7d0cb314be4e7a4d6`.
- Every model received the same question and the same six image conditions.
- Scoring used the A/B/C/D next-token distribution with batch size 1. Each
  original was scored twice. The maximum repeatability error was exactly zero
  for all six models.
- `nearest_same_target` replaces the image with the most globally similar image
  among cases whose correct answer is in the same A/B/C/D position. This is the
  stricter mismatch control because it preserves answer-position distribution.
- `nearest_different_target` uses the nearest globally similar image with a
  different correct-answer position.
- Matching was model-blind and based on low-resolution RGB mean, standard
  deviation, histograms, and aspect ratio. Thus these are appearance-matched
  same-domain negatives, not pathology-subtype-matched negatives.
- Patch shuffle preserves the exact pixels while permuting an 8x8 grid of
  spatial tiles. Strong blur and mean-color images are supportive but more
  out-of-distribution controls.

The continuous endpoint is the change in ground-truth option margin:

`margin(correct) = logit(correct) - logsumexp(logits(other three))`

Positive `Delta margin` means the original paired image supported the correct
option more than the counterfactual image did. `Delta acc` is paired original
accuracy minus counterfactual accuracy, in percentage points.

## Results

| Model | Original acc | Same-position mismatch Delta margin [95% bootstrap CI] / Delta acc | Different-position mismatch Delta margin [95% bootstrap CI] / Delta acc | 8x8 shuffle Delta margin / Delta acc | Strong blur Delta margin / Delta acc |
|---|---:|---:|---:|---:|---:|
| Base Qwen2.5-VL-7B | 56.2% | +0.501 [+0.097, +0.935] / +6.2 pp | +0.987 [+0.574, +1.395] / +12.5 pp | +0.154 / +0.0 pp | +1.357 / +25.0 pp |
| SFT-3000 | 55.2% | +0.370 [+0.165, +0.582] / +9.4 pp | +0.467 [+0.273, +0.668] / +7.3 pp | +0.033 / +0.0 pp | +0.419 / +8.3 pp |
| Stage2-RL | 57.3% | +0.381 [+0.171, +0.600] / +7.3 pp | +0.470 [+0.272, +0.679] / +9.4 pp | +0.023 / +3.1 pp | +0.402 / +9.4 pp |
| SFT-4000 control | 42.7% | +0.116 [-0.035, +0.270] / +2.1 pp | +0.181 [+0.026, +0.338] / +3.1 pp | -0.074 / -1.0 pp | -0.025 / +2.1 pp |
| GPT-4o Stage3, step 1000 | 57.3% | +0.451 [+0.215, +0.710] / +10.4 pp | +0.557 [+0.327, +0.807] / +11.5 pp | +0.050 / +3.1 pp | +0.500 / +10.4 pp |
| Grok-4.3 Stage3, step 1500 | 62.5% | +0.440 [+0.204, +0.684] / +9.4 pp | +0.570 [+0.343, +0.812] / +8.3 pp | +0.079 / +5.2 pp | +0.497 / +9.4 pp |

For the same-position mismatch, the mean correct-option margin confidence
interval excludes zero for five of six models. SFT-4000 is the exception. The
two completed Stage3 arms both lose about 9--10 accuracy points and about
0.44--0.45 correct-option margin when the image is replaced by an
appearance-matched wrong-case image.

The exact paired-accuracy tests are less powerful because accuracy discards
the magnitude of score changes. After a conservative two-comparison Holm
adjustment over the two mismatch conditions, GPT-4o remains below 0.05 for
both mismatch comparisons. Grok's adjusted accuracy p-values are about 0.07;
its continuous margin intervals nevertheless exclude zero for both mismatch
conditions. These two statistical views should be reported together rather
than selecting whichever is more favorable.

The effect is not confined to one answer position. For GPT-4o, the
same-position mismatch mean margin gains for A/B/C/D are respectively
`+0.462/+0.649/+0.295/+0.398`; for Grok they are
`+0.503/+0.621/+0.238/+0.396`. All eight strata therefore have the expected
direction. The effect also remains positive when restricted to cases the model
originally answered incorrectly: `+0.197` for GPT-4o and `+0.230` for Grok.
This reduces, but does not eliminate, the possibility that the aggregate is
only a consequence of conditioning on originally correct cases.

## Interpretation

The strongest defensible conclusion is:

> Across a frozen 96-case panel, the final GPT-4o- and Grok-supervised models
> assigned higher ground-truth answer margins and achieved higher paired
> accuracy with the correct case image than with appearance-matched images
> from other cases. This supports use of image-specific, task-relevant visual
> information in the decision.

The result is not Stage3-specific: the base, SFT-3000, and Stage2 models also
show visual dependence. Therefore this experiment supports the claim that the
model family uses the image; it should not be used alone to claim that Stage3
created visual understanding. Stage3-specific changes require a paired
between-checkpoint analysis and are secondary to the present objective.

The weak SFT-4000 result is scientifically useful. It shows that the test is
not guaranteed to produce a positive effect for every checkpoint and suggests
that this control checkpoint relies less reliably on the paired visual input.

The 8x8 tile shuffle has little effect. A likely explanation is that it
preserves local texture and many cell-level morphological cues while destroying
only coarse global arrangement. It should not be presented as evidence that
spatial organization is irrelevant. A registered granularity sweep is needed
before making a structural claim.

### Exact-pixel patch-shuffle granularity follow-up

The registered follow-up permuted 4x4, 8x8, and 16x16 spatial tiles while
preserving the exact RGB-pixel multiset. On average, the transformations moved
pixels to locations with different RGB values at rates of 87.0%, 95.2%, and
98.0%, respectively.

| Model | 4x4 Delta margin [95% CI] / Delta acc | 8x8 Delta margin [95% CI] / Delta acc | 16x16 Delta margin [95% CI] / Delta acc |
|---|---:|---:|---:|
| GPT-4o Stage3 | +0.165 [+0.028, +0.322] / +6.25 pp | +0.113 [-0.028, +0.264] / +3.12 pp | +0.085 [-0.092, +0.272] / +2.08 pp |
| Grok Stage3 | +0.183 [+0.044, +0.341] / +6.25 pp | +0.105 [-0.046, +0.268] / +5.21 pp | +0.153 [-0.039, +0.369] / +4.17 pp |

Only the 4x4 condition has a bootstrap interval excluding zero in both models.
A deterministic one-sided sign-flip robustness check, Holm-adjusted across the
three granularities within each model, gives `p=0.0403` for GPT-4o and
`p=0.0207` for Grok. Because the effect is not monotonic with finer shuffling
and the positive-case rate is only about 51--56%, this is modest evidence for
some coarse tissue-organization dependence, not proof of comprehensive spatial
understanding. The appearance-matched image replacement remains the stronger
behavioral result.

### Geometry-matched layerwise causal tracing

To remove the remaining image-size and visual-token-count confound, each
same-answer-position mismatch image was center-cropped and resized to the exact
geometry of the correct image. The processor then produced identical sequence
lengths and visual grids for each pair. At each selected decoder layer, the
final prompt-token residual state in the mismatch run was replaced by the state
from the correct-image run. An equal-magnitude extrapolation in the opposite
image-induced direction was used as a directional control.

The geometry-matched mismatch still reduced GPT-4o accuracy from 57.3% to
49.0% (`-8.33 pp`) and Grok accuracy from 62.5% to 56.2% (`-6.25 pp`). The
ground-truth margin gap was `+0.350 [0.117, 0.602]` for GPT-4o and
`+0.322 [0.098, 0.554]` for Grok. This rules out native image size or a changed
number of visual tokens as the explanation for the paired-image result.

| Layer | GPT-4o correct-state recovery [95% CI] / aggregate fraction | GPT-4o correct-minus-opposite [95% CI] | Grok correct-state recovery [95% CI] / aggregate fraction | Grok correct-minus-opposite [95% CI] |
|---:|---:|---:|---:|---:|
| 0 | +0.004 [-0.012, +0.019] / 1.1% | -0.005 [-0.022, +0.011] | -0.016 [-0.034, +0.001] / -5.1% | -0.015 [-0.034, +0.004] |
| 4 | +0.006 [-0.011, +0.025] / 1.8% | +0.012 [-0.008, +0.031] | +0.006 [-0.014, +0.024] / 1.8% | +0.011 [-0.011, +0.033] |
| 8 | -0.015 [-0.034, +0.005] / -4.2% | -0.017 [-0.042, +0.007] | -0.004 [-0.024, +0.016] / -1.2% | -0.000 [-0.028, +0.027] |
| 12 | +0.011 [-0.026, +0.045] / 3.1% | +0.006 [-0.054, +0.064] | -0.002 [-0.039, +0.033] / -0.6% | +0.007 [-0.054, +0.068] |
| 16 | -0.008 [-0.046, +0.028] / -2.4% | -0.022 [-0.085, +0.040] | -0.002 [-0.040, +0.034] / -0.7% | +0.008 [-0.055, +0.072] |
| 20 | +0.218 [+0.065, +0.391] / 62.4% | +0.418 [+0.144, +0.711] | +0.203 [+0.054, +0.368] / 63.1% | +0.413 [+0.144, +0.701] |
| 24 | +0.316 [+0.131, +0.527] / 90.4% | +0.617 [+0.277, +0.990] | +0.288 [+0.111, +0.481] / 89.4% | +0.590 [+0.253, +0.950] |
| 27 | +0.350 [+0.117, +0.602] / 100% | +0.800 [+0.359, +1.273] | +0.322 [+0.098, +0.554] / 100% | +0.754 [+0.330, +1.205] |

Replacing the final-layer state recovers the original A/B/C/D probabilities
with maximum absolute error exactly `0.0`, an implementation correctness gate.
The informative result is not this guaranteed endpoint but the synchronized
transition at layer 20 in both independently trained Stage3 arms: earlier-layer
patches have intervals spanning zero, whereas layers 20 and 24 recover a large
and selectively correct-option-directed portion of the image-induced margin.

Because the question text, answer options, token count, positions, and geometry
are fixed within each pair, the patched difference is induced by image content.
This provides causal evidence that image-specific information is carried into
the decision state. It traces *when* visual evidence affects the answer but does
not identify *where* the visual evidence is located.

#### Equal-norm activation-direction control

The layer-20 and layer-24 result was then tested against four deterministic
coordinate permutations of the correct-minus-mismatch activation delta. Each
control has exactly the same norm and is injected at exactly the same layer and
token position as the correct image-induced direction. This asks whether the
recovery is specific to the learned image-induced direction rather than a
generic consequence of injecting an equally large residual vector.

| Model | Layer | Correct-direction recovery [95% CI] | Mean permuted-direction effect | Correct minus permuted [95% CI] | Correct beats mean permutation |
|---|---:|---:|---:|---:|---:|
| GPT-4o Stage3 | 20 | +0.218 [+0.065, +0.391] | +0.014 | +0.204 [+0.051, +0.374] | 63.5% |
| GPT-4o Stage3 | 24 | +0.316 [+0.131, +0.527] | -0.001 | +0.318 [+0.126, +0.528] | 66.7% |
| Grok Stage3 | 20 | +0.203 [+0.054, +0.368] | +0.010 | +0.194 [+0.044, +0.355] | 58.3% |
| Grok Stage3 | 24 | +0.288 [+0.111, +0.481] | +0.008 | +0.280 [+0.099, +0.471] | 60.4% |

The correct-minus-permuted intervals exclude zero at both layers in both
models, while the mean same-norm permuted effects are near zero. Together with
the opposite-direction control, this supports a direction-specific causal
trace of image information into the answer state. It remains a non-spatial
result and must not be presented as lesion localization.

A reproducibility audit on frozen case 0 identified the original runtime as
Torch `2.11.0+cu130` with Transformers `5.12.1` and eager attention. Re-running
the original, mismatch, correct-direction, and opposite-direction paths under
that stack reproduced all four A/B/C/D probability vectors and the activation
delta norm with maximum absolute difference exactly `0.0`. Transformers 4.49
produces small numerical changes, so activation-patching confirmations are
pinned to the reproduced 5.12.1 stack rather than mixing runtime versions.

### Six-model, eight-layer option-conditioned spatial-support audit

The frozen 96 cases were also evaluated at layers 0, 4, 8, 12, 16, 20, 24,
and 27 for all six checkpoints. For every ground-truth option, the top 25% of
its positive gradient-times-attention map was compared with low-relevance and
area/shape-matched circular-shift controls. A candidate map was required to
beat random controls in both directions: deleting it should selectively reduce
the ground-truth option margin, and retaining it should selectively preserve
that margin. One-sided sign-flip p-values were Holm-adjusted across the eight
layers within each model.

No model-layer pair passed this bidirectional criterion. The strongest isolated
deletion result was SFT-4000 layer 12 (`+0.1087`, 95% CI
`[+0.0392,+0.1821]`, Holm `p=0.013`), but its retention advantage was only
`+0.0320 [-0.0526,+0.1129]` with Holm `p=1`. Conversely, Grok layer 8 had a
retention advantage of `+0.1830 [+0.0437,+0.3465]`, but its deletion advantage
was `-0.0057 [-0.1360,+0.1160]`. The per-case minimum of the deletion and
retention advantages had a negative mean and a confidence interval below zero
at every evaluated layer in every checkpoint. Raw attention likewise failed
to show consistent bidirectional selectivity.

Some shallow or final-layer case counts are below 96 because the positive
ground-truth-option attribution is exactly empty, so no candidate support mask
exists; these cases were not silently imputed with a favorable mask. All model
runs themselves contain 96 unique cases, and every repeatability gate has
maximum absolute A/B/C/D probability error `0.0`.

This is an informative negative result: the current raw-attention and
gradient-times-attention heatmaps are not faithful enough to establish *where*
the model found the answer, even though the independent paired-image and
activation-patching experiments show that image content affects the decision.
The predeclared full table is in
`docs/20260809_option_conditioned_spatial_support_96case_results.md`. A
model-agnostic, option-conditioned RISE perturbation analysis is therefore used
as the next spatial-evidence attempt rather than selecting a visually appealing
attention layer.

### Option-conditioned black-box RISE confirmation

The follow-up used 256 antithetic random soft masks per case to estimate a
separate spatial map for every A/B/C/D option margin. Map generation used a
blurred-image background, whereas causal validation used hard mean-color
deletion and retention. Five circular shifts of each selected mask preserved
its exact area and shape and served as random-location controls. The primary
target was always the ground-truth option margin; all four option scores were
recorded from the same masked forwards.

| Model | Validation | RISE-high vs matched random [95% CI] | Holm p | RISE-high vs RISE-low [95% CI] | Cases favoring RISE-high |
|---|---|---:|---:|---:|---:|
| GPT-4o Stage3 | deletion | +0.2407 [+0.1940, +0.2903] | 0.00004 | +0.4593 [+0.3772, +0.5465] | 93.8% |
| GPT-4o Stage3 | retention | +0.2097 [+0.1642, +0.2601] | 0.00004 | +0.4445 [+0.3676, +0.5245] | 90.6% |
| Grok Stage3 | deletion | +0.2205 [+0.1690, +0.2802] | 0.00004 | +0.4489 [+0.3675, +0.5372] | 91.7% |
| Grok Stage3 | retention | +0.2241 [+0.1754, +0.2784] | 0.00004 | +0.4189 [+0.3371, +0.5053] | 88.5% |

The p-values are one-sided sign-flip tests with Holm correction across the two
models and two validation directions. Both directions are positive within the
same case for 87.5% of GPT-4o cases and 82.3% of Grok cases. The mean per-case
minimum of the two directional advantages is `+0.1305 [0.0972,0.1647]` for
GPT-4o and `+0.1291 [0.0924,0.1679]` for Grok. Baseline, masked-input sentinel,
and intervention sentinel repeatability errors are all exactly `0.0`.

This is substantially stronger spatial-decision evidence than the attention
maps: a region found from independent randomized perturbations selectively
supports the correct option under both removal and retention, generalizes to a
different validation perturbation, beats same-shape random locations and
low-saliency regions, and replicates across two independently supervised Stage3
models. It supports the statement that the model's answer depends on localized
image content. It still does **not** prove that the region is the pathologist's
diagnostic lesion; agreement with independently generated and later
expert-verified regions is a separate pending claim.

The complete confirmatory table is in
`docs/20260809_option_conditioned_rise_96case_results.md`. These runs used the
frozen panel hash above and the GRPO runtime (Python 3.10, Torch 2.6.0+cu124,
Transformers 4.49.0, eager attention). They were not mixed with the 5.12.1
activation-patching runtime; each experiment is internally deterministic and
version-locked.

### Fixed-case visual audit of the depth profiles

Cases 0--11 were rendered before inspecting their intervention outcomes; they
were not selected for attractive maps. Each figure shows the original image,
raw final-query attention at eight layers, and the ground-truth-option positive
gradient x attention map at the same layers. Cyan boundaries are the exact
top-25% cells used by the causal intervention.

Direct visual inspection finds that raw attention has substantial border and
corner bias across nearly every case. Option conditioning changes the selected
regions and sometimes moves emphasis onto plausible tissue, arrows, or an
inset panel, but it also frequently leaves fragmented masks, blank-space
hotspots, or unrelated tissue. No single layer is uniformly convincing across
the 12 fixed cases. GPT-4o and Grok maps are often similar, which is expected
from their shared backbone and relatively small weight changes, but this is not
evidence that the maps are correct.

Therefore the rendered heatmaps cannot serve as the primary proof that the
model understood the image. Their role is diagnostic and illustrative; the
paired-image behavior and geometry-matched activation intervention are the
stronger evidence. The completed 96-case deletion/retention statistics confirm
that no inspected attention layer provides robust bidirectional spatial
support. Black-box RISE has now supplied positive spatial-decision evidence.
Independent region annotations and external/pathologist review remain necessary
before claiming clinically correct lesion localization.

## Audit

- Six metric files contain 96 unique records each; their recorded JSONL hashes
  match the files on disk.
- All runs use the identical frozen panel and counterfactual manifest hashes.
- No case was paired with itself.
- All same-position pairs preserve the target letter; all different-position
  pairs change it.
- All original-repeat checks have maximum absolute A/B/C/D probability
  difference `0.0`.

## Remaining evidence chain

1. The predeclared layer profile (layers 0, 4, 8, 12, 16, 20, 24, 27) has been
   completed on the same 96 cases and six checkpoints. Raw attention and
   option-conditioned positive gradient x attention both fail the strict
   spatial-support confirmation criterion.
2. The required bidirectional deletion/retention analysis has been completed;
   no model-layer pair passes both directions. A black-box,
   option-conditioned RISE confirmation on the same frozen panel has been
   completed and is strongly positive in both Stage3 models.
3. Repeat the causal intervention on a frozen subset with a blurred fill or
   content replacement to ensure the result is not specific to mean-color
   occlusion artifacts.
4. Add an attribution model-parameter randomization sanity check. A visually
   plausible map that is insensitive to model parameters is not a valid model
   explanation.
5. Add external closed-model boxes and later pathologist verification for a
   small, model-blind subset. Localization evidence and decision-dependence
   evidence should be reported as complementary, not interchangeable.
6. Sweep patch-shuffle granularity (for example 4x4, 8x8, and 16x16 cells) on
   a frozen subset to distinguish reliance on local morphology from reliance
   on global tissue arrangement. This has now been completed for the two final
   Stage3 arms; interpretation remains secondary because the effect is weak and
   non-monotonic.

The geometry-matched residual activation-patching and equal-norm direction
controls have also been completed for both final Stage3 arms. The attention-
based spatial intervention is complete and negative; black-box RISE is complete
and positive. Both are deliberately kept separate from the non-spatial causal
trace, and the negative attention result is retained rather than hidden.

The methodology follows the general warning from Adebayo et al., *Sanity
Checks for Saliency Maps* (NeurIPS 2018), that visual plausibility is not enough,
and the black-box perturbation logic of RISE. Saliency-metric conclusions will
also be kept subordinate to direct behavioral controls because saliency metric
rankings themselves can be unstable.

The next layerwise causal-tracing experiment is additionally motivated by
[Pixels Versus Priors (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1262/),
which uses visual counterfacts and layerwise activation interventions to study
when visual evidence overrides language priors. The decision to combine
gradients with multiple layers rather than select one attractive raw-attention
map is consistent with the faithfulness findings in
[On the Faithfulness of Vision Transformer Explanations (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_On_the_Faithfulness_of_Vision_Transformer_Explanations_CVPR_2024_paper.html).

## Artifacts

- Frozen panel: `protocol/visual_understanding_confirmation_panel_v2_96case_20260808.json`
- Frozen pairings: `protocol/visual_understanding_counterfactual_pairs_v1_96case_20260808.json`
- Runner: `scripts/run_visual_understanding_counterfactual.py`
- Patch-shuffle runner: `scripts/run_patch_shuffle_granularity.py`
- Layerwise causal-tracing runner: `scripts/run_layerwise_visual_activation_patching.py`
- Strict multi-layer spatial-support analysis:
  `scripts/analyze_option_conditioned_confirmation.py`
- Option-conditioned RISE runner:
  `scripts/run_option_conditioned_rise_confirmation.py`
- Strict RISE confirmation analysis:
  `scripts/analyze_option_conditioned_rise_confirmation.py`
- Per-model metrics:
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/visual_understanding_counterfactual_96case_*_20260808/metrics.json`
