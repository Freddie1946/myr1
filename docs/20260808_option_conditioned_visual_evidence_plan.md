# Option-Conditioned Visual Evidence Confirmation

## Question

Does a region selected from a model's visual signal support a particular answer
option, or is the model merely using an unrelated region to eliminate other
options?

## Frozen data and models

The 24-case exploratory panel and the earlier 48-case implementation panel are
excluded. The confirmation panel was frozen before any outputs from the six
models below and contains 96 fresh PathMMU validation cases, balanced at 24
each for A/B/C/D. Its manifest is
`protocol/visual_understanding_confirmation_panel_v2_96case_20260808.json`
with SHA-256
`c1222d9ff2d3a1b038a744b2f1c1addd18e777de3fb4f31c09107acef7e52931`.

The comparison includes base Qwen2.5-VL, SFT-3000, Stage2-RL, the 4000-SFT
control, GPT-4o Stage3 at step 1000, and Grok Stage3 at step 1500. Every model
uses the same cases, layer indices, region fraction, and scoring contract.
Kimi is optional because it is a separate API-training arm, not a prerequisite
for the visual-understanding claim.

## Attribution and intervention

For each model, case, option A/B/C/D, and transformer layer, the script computes
the positive and negative parts of `attention x gradient` for the option margin

`m_j = logit(j) - logsumexp(logits(other options))`.

Only the positive map is treated as candidate support.  Its top 25% visual
cells are mapped to the original image coordinates.  The same-area low-positive
map and two circularly shifted masks are controls.  Each mask is evaluated by
deletion and retention using mean-color fill, and every intervention records all
four option probabilities and margins.

All causal scores use `batch_size=1`.  The smoke gate showed that BF16
multimodal option probabilities can change with batch position even when the
image and mask are identical.  Each original image is therefore scored twice
through the same no-attention inference path used for interventions, and the
maximum repeatability difference is recorded.  Probabilities from the
attention-producing forward pass are diagnostic only and are not the causal
baseline.

The top region of the unconditioned final-query attention map is also tested
with exactly the same mask area.  This is the direct comparator for the concern
that raw attention may be useful for rejecting distractor options without
supporting any particular answer.

The key support statistic is the candidate region's drop in its own option
margin after deletion, minus the mean drop for the other three options.  A
candidate is considered evidence of option support only when this selectivity
exceeds the area-matched controls and has the same direction under retention.
If all options fall together, or only incorrect options are affected, the
result is reported as visual dependence or option exclusion, not support.

## Layers and reporting

All attention layers are extracted.  The default confirmation run evaluates
layers 0, 4, 8, 12, 16, 20, 24, and 27; a full layer sweep can be requested with
`--layers all`.  This avoids selecting a single favorable layer while retaining
the ability to inspect the complete depth profile.  Aggregate reports include
per-layer and per-option means, control advantages, and the rate at which the
candidate beats its random controls.

The result is still a model-internal causal diagnostic, not expert ground
truth. It will be interpreted together with three independent checks:

1. the completed 96-case paired-image counterfactual, which tests whether the
   correct case image improves the ground-truth option over appearance-matched
   wrong-case images;
2. exact-pixel patch shuffling, which tests spatial organization while keeping
   RGB content fixed; and
3. geometry-matched layerwise residual activation patching, which causally
   traces when the correct-image state starts to recover the decision relative
   to the wrong-image state.

Rendered maps will still receive blinded external-model review and later
pathologist verification before any clinical localization claim. Parameter
randomization sanity checks are required before a heatmap is described as a
faithful model explanation.
