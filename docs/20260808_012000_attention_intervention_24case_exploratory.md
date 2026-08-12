# Twenty-four-case attention-guided intervention experiment

Recorded at: 2026-08-08 CST

## Scope and frozen inputs

This exploratory experiment tests whether raw question/options-to-vision
attention identifies image regions that affect the GPT-4o Stage3 model's own
decision. It uses all 24 cases in the pre-existing PathMMU validation385 visual
fidelity panel. The panel was selected before model visualization outputs,
contains six cases for each target choice, and was not selected by model
performance.

Evaluated checkpoint:

`gpt4o_full3epoch_seed42_20260801/epoch_model_snapshots/checkpoint-1000`

The attention ranking does not use answer logits, target labels, perturbation
outcomes, or external ROI annotations. It averages raw attention from the
question and answer-option query tokens to all visual tokens. Five fixed layer
strategies were compared: layer 14, layer 16, layers 14--17, middle layers
8--17, and late layers 18--27. Reversing the layers-14--17 order provides a
low-attention control.

## Intervention design

For each frozen attention ranking, image regions were progressively deleted or
retained at fractions 0%, 10%, 25%, 50%, 75%, and 100%. Both global mean-fill
and Gaussian-blur replacements were used. Each attention-guided curve was
compared within case with five deterministic, area-matched random orders.

The primary continuous endpoint is the evaluated model's probability for its
own original decision, not the ground-truth option probability. This avoids
treating an incorrect baseline answer as though attention must support the
correct label. Answer change is a discrete companion endpoint; accuracy under
intervention is secondary. Baseline accuracy on this small panel was 11/24
(45.8%).

For deletion, fidelity advantage is random-order AUC minus attention-order AUC;
for retention, it is attention-order AUC minus random-order AUC. Positive
values favor the attention ranking in both cases.

## Exploratory results

Across both perturbations and both intervention modes, paired case-level
bootstrap results were:

| Attention strategy | Mean probability-AUC advantage | Paired bootstrap 95% CI | Cases with positive mean advantage |
|---|---:|---:|---:|
| layer 14 | +0.0056 | [-0.0035, +0.0162] | 15/24 |
| layer 16 | +0.0006 | [-0.0070, +0.0082] | 11/24 |
| layers 14--17 | +0.0051 | [-0.0051, +0.0165] | 12/24 |
| middle 8--17 | +0.0079 | [+0.0006, +0.0161] | 17/24 |
| late 18--27 | +0.0070 | [+0.0006, +0.0143] | 14/24 |
| low-attention control | -0.0061 | [-0.0185, +0.0051] | 9/24 |

At the 25% intervention point, middle layers 8--17 had a mean original-decision
probability advantage of +0.0195 versus random (95% paired bootstrap interval
[+0.0067, +0.0329]). Its combined deletion-change/retention-preservation
advantage was +0.0521 in absolute answer-behavior rate (95% interval [+0.0146,
+0.1000]). Among the 11 initially correct cases, the corresponding 25% mean
accuracy advantage was approximately +0.055.

The middle-layer result was strongest for mean-fill retention (+0.0159 AUC;
95% interval [+0.0073, +0.0253]; positive in 17/24 cases). The blur-only
conditions were directionally weaker and their individual intervals crossed
zero. Therefore the experiment does not establish perturbation-invariant strong
causal fidelity.

## Interpretation

The data favor the predefined middle layer band over a single post-hoc selected
layer. High-attention regions have a small, directionally consistent advantage
over area-matched random regions, while the low-attention control trends in the
opposite direction. This is compatible with weak decision-relevant visual
information in raw middle-layer attention.

The effect is too small and heterogeneous to claim that raw attention faithfully
explains the model decision in general. It should be reported as an exploratory
fidelity signal and combined with spatial overlap, random controls, multiple
perturbations, and later expert review. No multiple-comparison-adjusted formal
hypothesis test was performed, and these 24 cases were previously used for
visualization calibration; a new held-out cohort is required for confirmatory
reporting.

## Artifacts

- Merged metrics:
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/attention_intervention_24case_20260808/merged/metrics.json`
- Per-case curves and decisions:
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/attention_intervention_24case_20260808/merged/case_results.jsonl`
- Frozen attention maps:
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/attention_intervention_24case_20260808/merged/attention_maps.jsonl`
- Implementation:
  `scripts/run_attention_intervention_experiment.py`
- Shard merger:
  `scripts/merge_attention_intervention_shards.py`
- Unit tests:
  `scripts/test_run_attention_intervention_experiment.py`
