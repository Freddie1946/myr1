# External evidence-region annotation — 2026-08-12

The frozen 80-case panel was annotated by `gpt-4o-2024-08-06` through Aigcbest
before any target-checkpoint output was inspected. The complete JSONL audit is
kept outside the code repository and its hash is recorded in the selected-panel
manifest.

## Screening result

- Candidate cases: **80** (20 each for A/B/C/D; test data were not used).
- Focal: **24**; multifocal: **6**; diffuse: **47**; not localizable: **3**.
- Cases with nonempty boxes: **30**.
- Frozen selected panel: **27** cases, using only the predeclared rule
  `focal or multifocal` + nonempty boxes + confidence ≥ 0.90 + redundancy ≤ 0.50.
- Selection did not use target-model predictions, attention, RISE, or deletion outcomes.

## Important limitation

The external model predicted `little_change_expected` for all 80 cases when a
small reference region is deleted. This is not evidence that the regions are
irrelevant; it indicates that the annotator does not assert causal necessity,
likely because many images contain redundant or diffuse pathology evidence. The
27 selected cases should therefore be described as **diagnostically relevant
reference regions**, not ground-truth causal masks. The causal claim will come
only from the target-model intervention comparison:

`reference deletion` vs `area-matched neighboring deletion` vs `area-matched random deletion`.

The primary explainability panel will further retain only clean-correct target
predictions after the panel is frozen, and will report the full 80-case funnel.
If too few clean-correct cases remain, the 27-case panel will be expanded using
the same rule to the remaining annotated candidates; no post-hoc attention-based
case selection is allowed.

## Files

- Candidate panel: `protocol/predefined_evidence_candidate_panel_v1_80case_20260812.json`
- Frozen selected panel: `protocol/predefined_evidence_selected_panel_v1_20260812.json`
- Raw annotation audit: `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/predefined_evidence_annotation_aigcbest_20260812/case_annotations.jsonl`

