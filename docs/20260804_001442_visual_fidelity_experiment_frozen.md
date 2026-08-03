# Visual fidelity experiment frozen

Recorded at 2026-08-04 00:14:42 CST, before any model was run on the visualization panel.

## Scope and claim

Reviewer feedback correctly notes that one attention heatmap cannot establish localization or
explainability. The revision will therefore avoid treating attention as a causal explanation. The
new experiment measures post-hoc *fidelity under image perturbation* and will be reported with that
narrow wording.

The frozen panel contains 24 distinct PathMMU validation images, with six target answers for each
of A, B, C and D. Cases were selected without model outputs by ranking canonical record hashes with
a fixed salt while enforcing target-choice quotas and unique resolved image paths. The panel
manifest SHA-256 is `0c7e09a2e870abdb07b63df9cca1a410c73c9922e996bede9771777b636280e8`.
Primary manuscript examples are also fixed before inference as panel indices
`0, 3, 6, 9, 12, 15, 18, 21`; all 24 cases remain in the quantitative result.

## Frozen comparison arms

1. SFT3000 epoch 3 step 1125.
2. SFT4000 control trained on the same additional 1000 records.
3. Stage2 outcome-GRPO epoch 2 step 1000.
4. GPT-4o Stage3 selected on validation385 at epoch 2 step 1000.
5. Kimi 2.6 Stage3 selected by the same validation385 rule after training completes.

No arm is selected by visualization performance.

## Perturbation contract

- Split: frozen PathMMU validation385 only; test999 is not used to choose or visualize cases.
- Grid: 6 by 6 non-overlapping patches covering the complete original image.
- Occlusion: replace a patch with that image's RGB channel mean.
- Importance: drop in log normalized target-option probability after single-patch occlusion.
- Option probability: softmax over next-token logits for the four single-token candidates A-D under
  an answer-only diagnostic prompt. It is explicitly not a calibrated full-vocabulary probability.
- Curves: deletion and insertion at fractions 0, 0.10, 0.25, 0.50, 0.75 and 1.0.
- Control: five deterministic random patch permutations per case.
- Primary quantitative outputs: deletion AUC advantage over random, insertion AUC advantage over
  random, 25% comprehensiveness, answer accuracy on the fixed panel, and per-case raw values.
- Heatmaps: all cases receive a numeric colorbar. Every case within a model uses one shared symmetric
  color scale, and the stored metrics disclose its limits.

The experiment does not claim lesion overlap because no expert region annotations are currently
available. It instead provides a quantitative perturbation-fidelity result, multiple preselected
cases, and an explicit limitation. Expert localization agreement/IoU remains future work rather
than a fabricated proxy.

Implementation: `scripts/prepare_visual_fidelity_panel.py` and
`scripts/run_visual_fidelity_experiment.py`. Seven CPU-only preparation/perturbation tests passed at
the freeze boundary.

