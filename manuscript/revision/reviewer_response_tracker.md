# PathVLM-R1 reviewer-response and evidence tracker

Last updated: `2026-07-20T02:51:12+08:00`

This is the living point-by-point revision record for JBHI manuscript `JBHI-06328-2025`. Update it
after every material experiment, correction, analysis, figure/table change, or rebuttal decision.
The original decision PDF remains authoritative; comments below are concise paraphrases.

## Status vocabulary

- `completed-formal`: complete evidence eligible for the revised paper, subject to final audit.
- `completed-engineering`: useful implementation evidence, not a paper result.
- `partial`: some required evidence exists, but the reviewer request is not fully answered.
- `pending-experiment`: requires additional data collection, training, inference, or human scoring.
- `pending-writing`: evidence exists or no experiment is needed, but manuscript/rebuttal edits remain.
- `blocked-decision`: cannot proceed until the scientific definition or human protocol is approved.
- `invalid`: an attempt is preserved for audit but must not support a scientific claim.

## Associate Editor

### AE-1 — Address all open questions and highlight every revision

- Comment: The work is potentially interesting, but substantial weaknesses must be resolved; all
  revisions must be highlighted and explained point by point.
- Response strategy: maintain this tracker, immutable timestamped execution records, exact run/data/
  code manifests, and a final rebuttal cross-reference to revised page/line numbers.
- Evidence/actions: repository handoff, mandatory reading order, timestamped `docs/` history, formal
  run manifests, and this tracker are in place.
- Status: `partial`.
- Remaining: finish the experiments below, revise the manuscript, then add exact page/line references.

## Reviewer 1

### R1-1 — Validate GPT-4o reward scores against pathology experts

- Comment: repeated GPT-4o consistency does not establish clinical correctness; compare its scores
  with human pathology experts.
- Planned response: blinded stratified sample of reasoning traces, at least two pathology experts,
  fixed integrity/knowledge rubric, expert-expert and GPT-expert agreement (Spearman, weighted kappa
  or ICC with confidence intervals).
- Evidence/actions: historical GPT scoring scripts and the manuscript rubric have been audited, but
  no valid expert-score dataset has been collected.
- Status: `pending-experiment`.
- Remaining: approve sampling/rubric, collect expert scores, run agreement statistics, report limits.

### R1-2 — Add representative failure-case analysis

- Comment: analyze typical failures, including rare or difficult diseases, and derive concrete limits.
- Planned response: select cases from frozen validation during development and final test only after
  protocol lock; categorize visual-feature, concept-confusion, reasoning/answer mismatch, formatting,
  and low-frequency errors. Preserve raw generations.
- Evidence/actions: raw-generation persistence is mandatory; the invalid GRPO Attempt03 already
  exposed an option-enumeration parser failure, and validation-curve infrastructure saves every raw
  prediction. These are engineering lessons, not yet the final model failure analysis.
- Status: `partial`.
- Remaining: complete formal model(s), predeclare case-selection rules, analyze validation without
  tuning on test, and obtain expert comments for selected clinical errors.

### R1-3 — Explain degradation on Chest CT and retinal modalities

- Comment: the broad transfer claim conflicts with Table IV, where several modalities regress.
- Planned response: remove “remarkable/broad cross-modal transferability”; report modality-wise
  effects, confidence intervals, domain-distance/task differences, and explicit negative transfer.
- Evidence/actions: the contradiction and exact Table IV deltas have been audited and recorded.
- Status: `pending-writing` (additional inference/statistics may still be added).
- Remaining: revise Abstract/Introduction/Discussion/Conclusion and decide whether to rerun OOD sets.

### R1-4 — Justify the 0.4 process-reward penalty

- Comment: explain the empirical or theoretical basis for deducting 0.4 per missing/error step.
- Planned response: treat 0.4 as a heuristic unless historical evidence proves otherwise; perform a
  frozen sensitivity analysis (for example 0.3/0.4/0.5) and report ranking/agreement robustness.
- Evidence/actions: Stage 3 implementation/provenance audit found no complete historical online
  process-reward training implementation. No basis has been invented.
- Status: `blocked-decision`.
- Remaining: freeze Stage 3 process unit, rubric, aggregation, model/version, and sensitivity protocol.

### R1-5 — Break the GPT-4o training/evaluation circularity

- Comment: using GPT-4o for both optimization and final reasoning evaluation is not independent.
- Planned response: primary outcome accuracy remains rule-based; reasoning quality must be validated
  by pathologists and/or an independent evaluator, with GPT-4o results clearly secondary.
- Evidence/actions: online outcome rewards and offline parser consistency are now separated and
  audited; no valid independent clinical reasoning evaluation has yet been completed.
- Status: `pending-experiment`.
- Remaining: complete R1-1/R3-7 and predeclare the independent final evaluation protocol.

### R1-6 — Enlarge and improve Figure 4

- Response: redraw at publication resolution with readable labels, panel annotations, sample count,
  and correlation definitions.
- Status: `pending-writing`.

### R1-7 — Disclose baseline configurations and prompts in Table II

- Response: add model/revision, prompt template, image preprocessing, decoding, sample count, parser,
  and raw-output provenance for every baseline; avoid comparing unmatched settings without caveats.
- Evidence/actions: baseline repository/environment audit exists, but a final comparable-baseline
  manifest and complete raw outputs are not yet frozen.
- Status: `partial`.

### R1-8 — Add a color scale to Figure 7

- Response: add color bar and normalization details; stop treating one attention map as proof of
  faithfulness and pair it with quantitative/multi-case evidence under R3-2.
- Status: `pending-writing` plus `pending-experiment` for quantitative grounding.

### R1-9 — Standardize terminology

- Response: use `PathVLM-α`, `PathVLM-β`, and `process reward` consistently; reserve “loss” for an
  optimization objective; fix spelling/grammar globally.
- Evidence/actions: inconsistent terms and locations have been catalogued.
- Status: `pending-writing`.

## Reviewer 2

### R2-1 — Small data regime does not support broad foundational-model claims

- Comment: 3000 SFT and 1000 RL samples are small for a 7B VLM with many optimization steps.
- Response strategy: narrow the claim to PathMMU pathology multiple-choice VQA; report SFT/RL data
  scaling and training-duration curves; do not claim a broadly generalizable clinical foundation model.
- Evidence/actions: nested SFT 500/1000/2000/3000 and RL 250/500/1000 sets are frozen. Formal seed-42
  n=3000 and n=2000 SFT runs completed with valid updates. Earlier n=500/n=1000 final models exist,
  but used a different ZeRO-3 offload backend and lack early-epoch snapshots. Their final-only
  validation scores are 49.87% and 51.43%, and current parser re-scoring has zero mismatches. Formal
  validation selected n=2000 epoch 5 at 57.92%
  and n=3000 epoch 3 at 60.52%; epoch 10 fell to 51.95% and 55.58%, respectively. All 8,085 raw
  predictions are preserved. No formal RL scale curve exists.
- Status: `partial`.
- Frozen decision: do not rerun n=500/n=1000. Report the fixed epoch-10 endpoints for all sizes
  (49.87/51.43/51.95/55.58%) and keep the n=2000/n=3000 early-stopping curves separate. Explicitly
  disclose the ZeRO-3 versus ZeRO-2 backend limitation. See
  `docs/20260720_001137_no_n0500_n1000_rerun_decision.md`.
- Remaining: run RL scale after Outcome-GRPO gates and rewrite/narrow claims.

### R2-2 — Specify image/case-level splitting and prevent leakage

- Comment: PathMMU can contain multiple QAs per image; disclose the split unit and image/case/patient
  deduplication across SFT, RL, validation, and test.
- Response strategy: group all QAs for one image together; verify basename and exact file-content
  SHA-256 isolation; state that current records contain no case/patient identifier, so patient-level
  isolation cannot be claimed; consider perceptual-near-duplicate auditing as an additional limit.
- Evidence/actions:
  - Historical Q&A-level split was rejected after 437 shared SFT/test basenames were found.
  - v1 grouped by basename and passed six basename-overlap gates.
  - A stronger audit found one exact JPEG duplicated under different basenames across RL/test.
  - Before any formal RL/test, v2 removed the single opaque test QA; SFT/RL/validation and every
    nested subset remain byte-identical to v1.
  - v2 contains 3000/1000/385/999 QA, 2121/708/272/707 basenames, 3808 referenced basenames and
    3807 exact contents. All six basename and exact-content overlaps are zero; all 3808 physical
    file hashes pass. The removed question/answer and any model performance were not used.
- Status: `completed-formal` for basename and exact-byte image isolation; `partial` for patient-level
  and perceptual-near-duplicate claims.
- Evidence path: `docs/20260719_174130_pathmmu_exact_content_v2_and_reviewer_tracker.md` and
  `data/pathmmu_image_disjoint_v2/`. The formal v2 adapter is installed and the strengthened machine
  preflight passes every gate.
- Remaining: disclose unavailable patient IDs in the manuscript/rebuttal and decide whether to add
  a perceptual-near-duplicate audit.

### R2-3 — Add pathology-specialized and strong non-RL baselines

- Comment: current baselines do not establish superiority over pathology-specialized alternatives.
- Planned response: position UNI, CONCH, PathChat and PLIP by task/capability; run compatible
  pathology/medical VQA baselines where weights and licenses allow; retain same-base SFT-only and
  continued-SFT controls.
- Evidence/actions: CONCH, UNI, PLIP, and LLaVA-Med repositories/requirements were audited; their
  task mismatches are documented. No new formal baseline result is complete.
- Status: `partial`.

### R2-4 — Fully specify and validate GPT-4o reward execution

- Comment: disclose online/offline behavior, exact prompt/model/decoding/parser/fallbacks/calls/cost;
  address answer-conditioned rationalization and obtain human validation.
- Evidence/actions: historical code audit found offline GPT-4o scorers but no complete Stage 3 online
  training implementation. Outcome-only GRPO now has shared online/offline parser v2, raw reward
  JSONL, variance/gradient/tensor-delta gates. The exact seed-42 n=3000 epoch-3 one-step formal-
  hardware gate passed with 16/16 online/offline-consistent reward events, reward std 0.25, gradient
  norm 2.93, a changed language tensor and exactly unchanged visual tensor. This is engineering-only
  and does not prove Stage 3. It also exposed an all-zero format-reward branch caused by a JSON-format
  prompt conflicting with the strict think/answer reward contract. The user approved removing only
  the contradictory JSON sentence. The prompt-v2 re-gate then passed: 8/8 format rewards positive,
  16/16 online/offline reward events consistent, reward std 0.2887, gradient norm 3.3764, language
  tensor changed and visual tensor exactly unchanged. Formal long Outcome GRPO has not run.
- Frozen next action: a 50-step save/resume/full-validation engineering pilot precedes the priority
  n=1000, seed-42, three-epoch Outcome-GRPO arm. Every online completion is bound to its exact frozen
  record/image hash/question/prompt and is re-scored offline. This does not address GPT-4o or Stage 3.
- Status: `blocked-decision`.
- Remaining: define and implement reconstructed Stage 3 transparently, cache every judge event, then
  run expert validation. Never claim exact historical recovery without provenance.

### R2-5 — Broaden evaluation, use full held-out data, and report uncertainty

- Comment: add official/independent pathology evaluation, full held-out results, CIs and repeated runs.
- Response strategy: validation is development-only; corrected 999-QA test is opened only after
  prompt/checkpoint/parser lock; save all raw predictions; report Wilson/bootstrap CIs and paired
  tests. Add independent pathology benchmarks if feasible.
- Evidence/actions: deterministic 385-QA validation infrastructure and raw-output audits exist; final
  full-test and independent-benchmark results do not.
- Status: `pending-experiment`.

## Reviewer 3

### R3-1 — Cross-modal transfer claim is unsupported

- Response: same corrective action as R1-3; explicitly report negative transfer on three modalities
  instead of averaging it away.
- Status: `pending-writing`/`pending-experiment`.

### R3-2 — One attention heatmap is insufficient and may be unfaithful

- Planned response: use multiple success/failure cases and a quantitative faithfulness protocol such
  as expert ROI ratings, occlusion/deletion-insertion, or overlap metrics where annotations exist.
- Evidence/actions: no qualifying quantitative grounding experiment has been completed.
- Status: `pending-experiment`.

### R3-3 — Clarify novelty and ablate the process reward

- Planned response: novelty cannot be “SFT + GRPO” alone; restrict it to a validated pathology-aware
  process-reward design and staged evidence. Compare SFT-only, outcome-only RL, process-aware RL, and
  a simpler alternative reward under matched data/settings with statistical analysis.
- Evidence/actions: formal SFT infrastructure is valid; Outcome-GRPO parser/update plumbing was debugged;
  the corrected formal-hardware one-step Outcome-GRPO gate now passes with 8/8 strict format rewards.
  Stage 3 definition and formal matched ablation remain incomplete.
- Status: `blocked-decision` then `pending-experiment`.

### R3-4 — Expand pathology foundation-model/VLM related work

- Response: compare UNI, CONCH, PathChat, PLIP and related systems by task, supervision, scale,
  generation/reasoning capability, clinical scope and limitations; explain when direct VQA comparison
  is invalid.
- Evidence/actions: repository and environment audit completed; manuscript section not yet revised.
- Status: `pending-writing`.

### R3-5 — Run systematic SFT and RL data-scaling experiments

- Planned matrix: nested SFT 500/1000/2000/3000 and RL 250/500/1000, with a predeclared feasible
  combination design rather than favorable-point selection.
- Evidence/actions: all nested sets and hashes exist. Formal n=2000/n=3000 seed-42 SFT completed with
  ten epoch snapshots; n=500/n=1000 historical formal finals exist but are not retention-matched.
  Validation Attempt01 failed on a chat-template archival race; the explicit frozen-template fix and
  retry code passes tests. Attempt02 then completed all 21 v2 validation jobs with 8,085 raw
  predictions and parser consistency. It selected n=2000 epoch 5 (57.92%) and n=3000 epoch 3
  (60.52%), demonstrating a non-monotonic duration effect. The user declined matched n=500/n=1000
  reruns; the scale table is therefore frozen to common epoch-10 endpoints with the backend mismatch
  disclosed. The one-step Outcome-GRPO engineering gate passed, but all eight format rewards were
  zero because the JSON-format prompt conflicted with the strict think/answer reward. The approved
  prompt-v2 correction re-gate passed with 8/8 positive format rewards, positive total-reward
  variance and a nonzero update. No formal RL-scale result exists. See
  `docs/20260720_025112_outcome_grpo_prompt_v2_regate_completed.md`.
- Frozen priority under the user's revision-time constraint: run n=1000/seed42 first from the selected
  n=3000 epoch-3 SFT parent after a 50-step save/resume/full-validation pilot. RL 250/500 and other
  SFT/RL combinations are deferred, so the systematic RL scale request remains incomplete.
- Status: `partial`.

### R3-6 — Add confidence intervals, significance tests, and multiple seeds

- Planned response: seeds 42/43/44 for main formal lineages without favorable-seed selection; raw
  paired predictions; mean/std across training seeds; Wilson/bootstrap CIs and McNemar/paired tests.
- Evidence/actions: seed-42 n=2000/n=3000 SFT is complete. Seeds 43/44, formal RL seeds and final
  statistical comparisons are not complete.
- Status: `pending-experiment`.

### R3-7 — Avoid GPT-4o reward/evaluator circularity

- Response: same as R1-1/R1-5/R2-4; expert or independent evaluation is mandatory for reasoning claims.
- Status: `pending-experiment`.

### R3-8 — Correct typographical and formatting errors

- Response: fix `Doubao-1.5-vison`, grammar, labels, figure/table sizing, and global nomenclature.
- Status: `pending-writing`.

## Formal operation and evidence ledger

This section records major work completed so far and the reviewer request it supports. Detailed
commands, failures, paths and gates remain in timestamped `docs/` files and external run manifests.

1. Frozen an image-grouped PathMMU split and nested SFT/RL sizes; deprecated `picked.json`.
   Supports R2-2, R3-5. Later exact-content audit discovered and corrected the v1 blind spot in v2.
2. Pinned Qwen2.5-VL-7B revision, LLaMA-Factory commit/hash, SFT/GRPO environments, PathMMU archive,
   adapters and preflight. Supports reproducibility requested throughout R1-7/R2-4/R3-6.
3. Completed formal 7B optimizer/save/reload/resume smoke; full language parameters trainable,
   vision/projector frozen, language tensor changed and visual tensor remained identical.
   Engineering evidence only.
4. Diagnosed throughput and selected eight-GPU ZeRO-2, fused AdamW, gradient-checkpointed backend.
   Engineering evidence only; failed/mislabeled attempts remain invalid or non-formal.
5. Implemented two-tier checkpoint retention: per-epoch model snapshots plus recent resumable states.
   Enables duration selection and R3-5 without retaining every optimizer state.
6. Completed formal seed-42 n=3000 and n=2000 SFT runs with ten snapshots and all gates. Supports
   R2-1/R3-5/R3-6 after validation results are complete.
7. Implemented deterministic validation curves with raw 385-row generations and offline parser
   consistency. Attempt01 failed because an archival timing race left four snapshots without archived
   chat templates and is not reused. Attempt02 completed all 21 jobs and selected n=2000 epoch 5 and
   n=3000 epoch 3; all 8,085 raw predictions are retained.
8. Debugged Outcome GRPO. Attempt02 had zero reward variance/no update; Attempt03 updated language
   parameters but used a false-positive parser and is scientifically invalid. Parser v2 regression
   tests pass; a valid formal Outcome-GRPO gate/run has not occurred. Supports R2-4/R3-3 only as
   engineering preparation.
9. Preserved raw logs/manifests and recorded approved storage pruning without rewriting failures.
10. Audited the official image archive by exact bytes. Found one RL/test duplicate under different
    basenames; created v2 by blindly excluding the test QA before formal RL/test. Directly supports R2-2.
11. Completed the seed-42 n=3000 epoch-3 fail-closed eight-GPU, one-step Outcome-GRPO gate. All
    predefined gates passed: 16/16 raw reward events, online/offline parser consistency, positive
    reward variance, nonzero gradient, loadable save, language delta and visual equality. The format
    branch was all zero because the upstream JSON-format instruction conflicts with the strict
    think/answer reward, so long RL remains blocked pending an explicit correction and re-gate.
    Engineering evidence only; supports R2-4/R3-3/R3-5 as preparation.
12. Completed the user-approved prompt-contract-v2 re-gate. Removing only the contradictory JSON
    instruction changed format rewards from 0/8 to 8/8; every online/offline reward, variance,
    gradient, save/load, trainability and tensor-freeze gate passed. Engineering evidence only; formal
    long RL and the RL-scale curve remain pending a separately frozen protocol.
13. Froze the next priority pipeline: step-25 save/resume to step 50, complete 385-QA validation
    gate, then only RL n=1000/seed42 for three epochs from the original SFT parent. Supports
    R2-4/R3-3/R3-5/R3-6 as a predeclared plan; no long-run result is claimed yet.

## Mandatory update rule

For every material future change:

1. add a timestamped `docs/YYYYMMDD_HHMMSS_*.md` record and update `docs/LATEST.md`;
2. update the relevant run/data/code manifest without rewriting failed history;
3. update every affected reviewer ID in this file with status, evidence path and remaining work;
4. distinguish engineering smoke, invalid attempt, formal validation evidence and final test evidence;
5. add exact revised-manuscript page/line references only after the manuscript edit exists;
6. never use test outputs to choose prompts, checkpoints, seeds, parsers, rewards or thresholds.
