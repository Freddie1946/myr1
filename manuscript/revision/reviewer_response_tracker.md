# PathVLM-R1 reviewer-response and evidence tracker

Last updated: `2026-07-25T01:05:33+08:00`

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
- Protocol change: on 2026-07-28 the user explicitly authorized PathMMU v2 test999 baseline and
  selected Stage 2 diagnostic runs for bad-case analysis and Stage 3 strategy adjustment. Therefore
  test999 is no longer eligible to support an untouched final Stage 3 claim. PathVQA and
  OmniMedVQA remain reserved for independent post-Stage-3 confirmation. Evidence:
  `docs/20260728_215133_pathmmu_test999_reclassified_as_stage3_development_diagnostic.md`.
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
- 2026-08-01 update: the user approved an offline 0.2--0.6 diagnostic followed by matched GPT-4o
  0.3/0.4/0.5 training pilots selected on validation only.  The approved execution record is
  `protocol/stage3_future_training_plan_approved_20260801_010917.json`.  The offline diagnostic over
  2,784 non-smoke judgments and both bounded GPT-4o smokes are complete in
  `protocol/stage3_sensitivity_gpt4o_smoke_20260801_012103.json`; no paid pilot has started.
- 2026-08-01 launch update: the user approved the separate $18 gate for three matched 100-step
  GPT-4o pilots.  Exact attempt accounting, cost caps and no-automatic-restart behavior are frozen
  in `protocol/stage3_gpt4o_coefficient_pilots_launch_20260801_013252.json`.
- 2026-08-01 execution update: 0.3 and 0.4 completed; 0.5 stopped after step 99 when retry overhead
  exhausted the physical-attempt cap.  It has no checkpoint and was not restarted.  The exact
  failure and remaining aggregate budget are recorded in
  `protocol/gpt4o_pilot_progress_0p5_attempt_cap_stop_20260801_143832.json`.
- 2026-08-01 recovery update: a single fresh, no-cache 0.5 recovery was approved with a separately
  bounded 24-attempt retry allowance and $5.50 cap; the contract is
  `protocol/gpt4o_0p5_fresh_recovery_approved_20260801_144436.json`.
- Evidence/actions: Stage 3 implementation/provenance audit found no complete historical online
  process-reward training implementation. No basis has been invented.
- Status: `in-progress` (protocol approved; offline sensitivity and bounded smoke verification are
  authorized, while paid pilots remain behind a measured-cost approval gate).
- Remaining: run the matched 0.3/0.4/0.5 pilots, select on the frozen validation endpoint, and
  report the sensitivity without claiming a post hoc theory.

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
  manifest and complete raw outputs are not yet frozen. The formal-machine audit confirms only
  Qwen2.5-VL-7B is currently end-to-end ready; historical scripts/results do not supply reproducible
  environment, weight and raw-output provenance for the other Table II/IV baselines. The user has
  now made every one of the fourteen Table II baselines and the additional Table IV
  HuatuoGPT-Vision-7B row mandatory; none may be dropped for cost or setup convenience. The exact
  historical 500 IDs and several exact API-version identities remain unresolved, so contemporary
  reruns must not be mislabeled as exact historical recovery.
- Evidence path: `docs/20260725_010533_baseline_access_and_environment_status.md` and
  `docs/20260725_011528_all_manuscript_baselines_mandatory.md`. The fixed-revision metadata/storage
  audit is `docs/20260725_013154_manuscript_baseline_metadata_and_access_audit.md`: MedGemma access
  passes, Meta Llama Vision 11B/90B access returns 403, and bulk download is storage-blocked.
  The new A100 preparation now has all isolated baseline environments, the fixed formal preflight
  and external data audit passing. Fixed-revision weight staging is active; actual Meta Llama
  11B/90B weight requests still require repository approval despite readable metadata. See
  `docs/20260728_021824_a100_core_preflight_and_external_asset_audit_completed.md` and
  `docs/20260728_015359_llama_vision_weight_access_gate.md`.
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
  validation scores are 49.87% and 51.43%, and current parser re-scoring has zero mismatches. The
  completed 1024-token, 26-candidate validation audit selected n=2000 epoch 5 at 57.92% and n=3000
  epoch 3 at 60.52%; epoch 10 fell to 51.95% and 55.58%, respectively. Outcome-GRPO n=1000 reached
  61.56/62.60/61.56% over epochs 1/2/3 and selected epoch 2. All 10,010 raw predictions are
  preserved. RL n=250/n=500 and seeds 43/44 remain unrun.
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
  task mismatches are documented. A preparation-only protocol now authorizes isolated, pinned
  environments for PLIP/CONCH/UNI and a separately gated LLaVA-Med environment. The isolated
  environment now passes CPU imports and dependency checks. PLIP metadata is prepared; authenticated
  fixed-revision weight HEAD checks now pass for CONCH and UNI. Weights and formal adapters are not
  yet local. Only the manuscript Qwen2.5-VL-7B baseline is end-to-end ready on this machine; other
  manuscript baseline environments/weights/raw outputs are absent or unauditable. No new formal
  baseline result is complete. All fifteen manuscript baselines are now mandatory in addition to,
  rather than instead of, the compatible pathology-specialized reviewer-response group.
- Evidence path: `docs/20260725_010533_baseline_access_and_environment_status.md` and
  `docs/20260725_011528_all_manuscript_baselines_mandatory.md`, with exact local-model metadata and
  current access/storage gates in
  `docs/20260725_013154_manuscript_baseline_metadata_and_access_audit.md`. New-machine environments,
  fixed formal preflight, pathology weights, Qwen 3B and MedVLM-R1 size gates, and the remaining
  download/access gates are recorded in
  `docs/20260728_021824_a100_core_preflight_and_external_asset_audit_completed.md`.
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
  tensor changed and visual tensor exactly unchanged. The formal three-epoch Outcome-GRPO n=1000,
  seed-42 arm subsequently completed from the frozen n=3000 SFT epoch-3 parent.
- Frozen execution contract: a 50-step save/resume/full-validation engineering pilot preceded the
  priority n=1000, seed-42, three-epoch Outcome-GRPO arm. Every online completion was bound to its
  exact frozen record/image hash/question/prompt and re-scored offline. This does not address GPT-4o
  or Stage 3.
- Result: the pilot passed all gates. Step 25 saved and resumed to step 50; 800/800 reward events had
  exact online/offline and source provenance; full validation was 232/385 versus parent 233/385,
  with exact McNemar p=1.0, 100% format/extraction and zero empty outputs. The formal arm did not
  start because the post-pilot storage projection failed closed.
- Formal result: after audited pruning of only the consumed pilot resume checkpoint, the formal-only
  continuation completed from the original SFT n=3000 epoch-3 parent. Corrected 1024-token validation
  scored epochs 1/2/3 at 237/241/237 of 385 (61.56/62.60/61.56%), selecting epoch 2 under the frozen
  rule. All raw validation predictions and online reward evidence remain preserved. This concerns
  outcome-only rewards and does not validate GPT-4o or Process Reward.
- Status: `blocked-decision`.
- Remaining: define and implement reconstructed Stage 3 transparently, cache every judge event, then
  run expert validation. Never claim exact historical recovery without provenance.

### R2-5 — Broaden evaluation, use full held-out data, and report uncertainty

- Comment: add official/independent pathology evaluation, full held-out results, CIs and repeated runs.
- Response strategy: validation is development-only; corrected 999-QA test is opened only after
  prompt/checkpoint/parser lock; save all raw predictions; report Wilson/bootstrap CIs and paired
  tests. Add independent pathology benchmarks if feasible.
- Evidence/actions: deterministic 385-QA validation infrastructure and raw-output audits exist; final
  full-test and independent-benchmark results do not. Preparation is frozen for the four full
  requested OmniMedVQA sources, PathMMU additional-source metadata and a normalized copy of the
  author-provided PathVQA test, with mandatory revision/hash and overlap audits before evaluation.
  All four OmniMedVQA sources and the 6,719-QA PathVQA test are now present; exact-content audits
  pass against formal PathMMU v2, but no inference or statistical result exists.
- Evidence path: `docs/20260725_004340_external_evaluation_assets_and_baseline_environment_prepared.md`.
  The same fixed archives were independently staged and audited on the A100 machine; see
  `docs/20260728_021824_a100_core_preflight_and_external_asset_audit_completed.md`.
- Status: `pending-experiment`.

## Reviewer 3

### R3-1 — Cross-modal transfer claim is unsupported

- Response: same corrective action as R1-3; explicitly report negative transfer on three modalities
  instead of averaging it away.
- Evidence/actions: preparation is frozen for full Chest CT, ISIC2020, Retinal OCT-C8 and Diabetic
  Retinopathy source subsets. All four are downloaded, path-complete and exact-content-disjoint from
  formal PathMMU v2. No rerun or statistical result is complete.
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
- Preparation action: exact source revisions, dependency versions, access limits and task
  comparability for UNI, CONCH and PLIP are recorded. CONCH/UNI license access is resolved; weights,
  adapters and manuscript text remain pending.
- Status: `pending-writing`.

### R3-5 — Run systematic SFT and RL data-scaling experiments

- Planned matrix: nested SFT 500/1000/2000/3000 and RL 250/500/1000, with a predeclared feasible
  combination design rather than favorable-point selection.
- Evidence/actions: all nested sets and hashes exist. Formal n=2000/n=3000 seed-42 SFT completed with
  ten epoch snapshots; n=500/n=1000 historical formal finals exist but are not retention-matched.
  Validation Attempt01 failed on a chat-template archival race; the explicit frozen-template fix and
  retry code passes tests. The corrected 1024-token Attempt02 completed all 26 candidates with 10,010
  raw predictions and parser consistency. It selected n=2000 epoch 5 (57.92%), n=3000 epoch 3
  (60.52%), and Outcome-GRPO n=1000 epoch 2 (62.60%), demonstrating non-monotonic duration effects.
  Two isolated 1024-token loops were preserved and scored as model failures. The user declined
  matched n=500/n=1000 reruns; the scale table is therefore frozen to common epoch-10 endpoints with
  the backend mismatch
  disclosed. The one-step Outcome-GRPO engineering gate passed, but all eight format rewards were
  zero because the JSON-format prompt conflicted with the strict think/answer reward. The approved
  prompt-v2 correction re-gate passed with 8/8 positive format rewards, positive total-reward
  variance and a nonzero update. The priority formal RL n=1000/seed42 arm is now complete; n=250,
  n=500, other SFT/RL combinations and seeds 43/44 remain deferred. See
  `docs/20260722_014643_validation_1024_attempt02_completed.md`.
- Additional control evidence: the exact selected SFT3000 parent was continued for two ordinary
  SFT epochs on the same ordered RL1000 records used by Stage2. On the development-classified
  PathMMU test999 split, SFT3000/SFT4000/Stage2 scored 582/595/609. This exposure-matched control
  shows that ordinary continuation accounts for part, but not all, of the Stage2 point-estimate
  gain. It remains one seed and does not complete the requested scaling matrix. Evidence:
  `docs/20260729_223742_pathmmu_sft3000_sft4000_stage2_diagnostics_completed.md`.
- Frozen priority under the user's revision-time constraint was n=1000/seed42 first from the selected
  n=3000 epoch-3 SFT parent after a 50-step save/resume/full-validation pilot. That priority arm is
  complete; RL 250/500 and other SFT/RL combinations are deferred, so the systematic RL scale request
  remains incomplete.
- The 50-step prerequisite pilot passed; approved checkpoint pruning restored the frozen storage
  margin; and the formal n=1000 continuation completed. Validation selected epoch 2 at 62.60%, versus
  the SFT parent at 60.52%. This is one RL size and one seed, not a systematic RL scaling result.
- Status: `partial`.

### R3-6 — Add confidence intervals, significance tests, and multiple seeds

- Planned response: seeds 42/43/44 for main formal lineages without favorable-seed selection; raw
  paired predictions; mean/std across training seeds; Wilson/bootstrap CIs and McNemar/paired tests.
- Evidence/actions: seed-42 n=2000/n=3000 SFT and Outcome-GRPO n=1000 are complete with paired raw
  validation outputs. Seeds 43/44 and final confidence intervals/paired statistical comparisons are
  not complete. The pre-Stage3 PathMMU diagnostic now also preserves paired raw predictions for
  SFT3000, SFT4000 and Stage2. Exact two-sided McNemar p-values are 0.490646, 0.030482 and 0.451463
  respectively; none is below 0.05 after Holm correction across the three exploratory comparisons.
  This is diagnostic evidence only and does not replace the planned multi-seed final analysis.
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
6. Completed formal seed-42 n=3000 and n=2000 SFT runs with ten snapshots and all gates. The complete
   validation curves now support R2-1/R3-5/R3-6, subject to the stated one-seed limitation.
7. Implemented deterministic validation curves with raw 385-row generations and offline parser
   consistency. The earlier 21-job SFT curve selected n=2000 epoch 5 and n=3000 epoch 3. The final
   corrected 1024-token Attempt02 completed all 26 Base/SFT/Outcome-GRPO candidates and retained all
   10,010 predictions; isolated cap hits are preserved rather than hidden.
8. Debugged Outcome GRPO. Debug Attempt02 had zero reward variance/no update; debug Attempt03 updated
   language parameters but used a false-positive parser and is scientifically invalid. Parser v2
   regression tests pass; later formal gates and the seed-42 n=1000 run completed. The debug attempts
   remain invalid and are not parents of the formal 7B lineage.
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
    gradient, save/load, trainability and tensor-freeze gate passed. Engineering evidence only; it
    enabled, but is distinct from, the later formal long run.
13. Froze the priority pipeline: step-25 save/resume to step 50, complete 385-QA validation gate,
    then only RL n=1000/seed42 for three epochs from the original SFT parent. This was the predeclared
    plan that governed the later completed run.
14. Completed that pilot with all gates: genuine save/resume, 400 audited completions, full validation
    232/385 versus parent 233/385, exact McNemar p=1.0, language change and visual equality. At that
    point, formal launch stopped before creation on the 550-GiB-reserve projection.
15. After explicit approval, hashed all 44 files and pruned only the consumed 109.3-GB pilot resume
    checkpoint. The final model, raw reward events, logs, tensor audit, 385 predictions and manifests
    remained. This restored the formal storage margin before the later launch.
16. Froze the separately manifested formal-only continuation from SFT n=3000/seed42/epoch3 to
    Outcome-GRPO n=1000/seed42 for three epochs after explicit user authorization. The launcher
    rechecked immutable pilot/pruning evidence and every formal gate.
17. Completed the formal Outcome-GRPO continuation and the corrected 26-candidate, 1024-token
    validation audit. Epochs 1/2/3 scored 61.56/62.60/61.56%, selecting epoch 2; the SFT parent was
    60.52%. All 10,010 raw predictions passed offline parser audits and test remained sealed. Supports
    R2-1/R2-4/R3-5/R3-6, but does not substitute for RL-size scaling, multiple seeds, Stage 3, expert
    reasoning validation, or final test.
18. Froze a preparation-only external-evaluation protocol for full requested OmniMedVQA sources,
    additional PathMMU metadata, independent PathVQA test, and isolated PLIP/CONCH/UNI environments.
    It authorizes no inference, training, GPU use, test opening or cleanup. Supports preparation for
    R1-3/R2-3/R2-5/R3-1/R3-4; no scientific result is claimed.
19. Prepared and exact-content-audited the four requested OmniMedVQA sources plus the 6,719-QA
    PathVQA test, and built a CPU-import-valid PLIP/CONCH/UNI environment. No external inference was
    run; CONCH/UNI weights remain license-blocked. Supports R1-3/R2-3/R2-5/R3-1/R3-4 preparation.
20. Re-audited baseline access after user license approval. Fixed-revision CONCH/UNI weight HEAD
    checks now pass, while downloads/adapters remain pending. Confirmed that only Qwen2.5-VL-7B among
    the manuscript baselines is end-to-end ready on this machine; the rest require reproducible
    environment/weight/provider work. Supports R1-7/R2-3/R3-4 preparation.
21. Recorded the user decision that all fourteen Table II baselines plus the additional Table IV
    HuatuoGPT-Vision-7B baseline are mandatory reruns. PLIP/CONCH/UNI remain an additional comparison
    group. Historical API identities and the old 500-example IDs are not yet recoverable, so the
    exhaustive scope is frozen without making a false exact-reproduction claim. Supports R1-7 and
    R2-3 preparation.
22. Queried official fixed-revision metadata for the nine missing local manuscript rows. Selective
    recognized weights total 301.82 GiB, while unfiltered repositories total 487.18 GiB; bulk
    download is blocked against the formal storage reserve. MedGemma gated access passes, but both
    Meta Llama 3.2 Vision repositories return 403. No weight/GPU/test use occurred. Supports R1-7 and
    R2-3 preparation.
23. Split future execution across two formal machines without narrowing the scientific plan. The
    existing machine is restricted to a separately frozen 4000-SFT training control; the new
    eight-A100 machine owns Stage 3, all validation/test, all original/additional baselines, OOD and
    remaining analyses. Added a memoryless personal-Codex bootstrap, workspace migration paths,
    checkpoint-transfer gates and a complete inherited experiment order. No training/inference/test
    occurred. Supports preparation across R1-2/R1-3/R1-7/R2-1/R2-3/R2-4/R2-5/R3-1/R3-4/R3-5/R3-6.
24. Completed the new A100 formal preflight with all 22 gates true and independently staged/audited
    the fixed OmniMedVQA four-source and PathVQA assets with zero missing images and zero
    exact-content overlap against formal PathMMU. Built all isolated baseline environments; all
    non-Llama local weight queues now pass exact-size and index-reference gates, while Meta Llama
    11B/90B remain repository-approval gated. The user approved PathMMU contract items 1/2/5,
    deferred hosted APIs, and retained items 3/4 for discussion. No model inference, training, paid
    API call or test selection occurred. Supports
    R1-7/R2-3/R2-5/R3-1/R3-4 preparation.
25. Before any new-machine test inference, recorded the user decision to use PathMMU v2 test999
    scores and bad cases to adjust Stage 3. Reclassified test999 as a development diagnostic and
    reserved PathVQA/OmniMedVQA for independent post-Stage-3 confirmation. The exact selected Stage
    2 checkpoint remains transfer-blocked; no substitute is allowed. No inference or Stage 3
    training occurred in the decision step.
26. Completed the first four authorized PathMMU test999 development diagnostics: Lingshu-7B
    583/999, Qwen2.5-VL-7B 485/999, Qwen2.5-VL-3B 449/999 and MedVLM-R1 413/999. Preserved raw
    predictions and generated a joint bad-case package with 162 all-model-wrong and 232
    one-model-only-correct cases. MedGemma, InternVL and HuatuoGPT-Vision remained in progress at
    this ledger update. DeepSeek-VL2 passed single-GPU BF16 inference but failed the fixed
    extractable-answer response-contract smoke; it was not run on test999. LLaVA-Med also produced
    nonempty natural-language option text without an extractable letter and was stopped before
    test999 instead of being assigned a parser-induced zero. Evidence:
    `docs/20260728_222041_pathmmu_test999_first_four_baseline_results_and_badcases.md` and
    `protocol/pathmmu_deepseek_vl2_response_contract_gate_20260728_222041.json` plus
    `protocol/pathmmu_llava_med_response_contract_gate_20260728_222938.json`. These results are
    development diagnostics only and cannot support an untouched final Stage-3 claim. The selected
    Stage-2 checkpoint is still absent, and Stage-3 training has not started.
27. Completed every currently runnable local PathMMU development diagnostic with an approved
    scoring contract. Generative scores are Lingshu 583/999, InternVL 553/999, Huatuo 531/999,
    Qwen7 485/999, Qwen3 449/999, MedVLM-R1 413/999 and MedGemma 400/999. Approved image-text
    matching scores are CONCH 338/999 and PLIP 334/999. Rebuilt a nine-model joint package with 55
    all-wrong and 88 sole-model-correct cases, retaining raw completions or four-option similarity
    scores. UNI remains representation-only; DeepSeek-VL2/LLaVA-Med failed the validation
    extractable-letter gate; Meta Llama weights and hosted APIs remain externally gated. Evidence:
    `docs/20260728_225929_pathmmu_test999_available_baselines_completed.md` and
    `protocol/pathmmu_test999_available_baselines_completion_manifest_20260728_225929.json`.
    Stage-2 is still blocked on exact checkpoint transfer, and no Stage-3 or other A100 training
    was started.
28. Downloaded the exact fixed SFT3000 and selected Stage2 checkpoints, completed the
    exposure-matched two-epoch SFT4000 control with all trainability/freeze/save/reload gates, and
    evaluated the three checkpoints under one PathMMU test999 development protocol. Scores were
    SFT3000 582/999, SFT4000 595/999 and Stage2 609/999. The paired net gains were +13, +27 and +14
    for SFT4000-minus-SFT3000, Stage2-minus-SFT3000 and Stage2-minus-SFT4000. A twelve-model bad-case
    package now contains 23 all-model-wrong cases. Evidence:
    `docs/20260729_223742_pathmmu_sft3000_sft4000_stage2_diagnostics_completed.md` and
    `protocol/pathmmu_test999_sft3000_sft4000_stage2_completion_manifest_20260729_223742.json`.
    These remain pre-Stage3 diagnostics; Stage3 has not started.
29. Completed the matched 0.3/0.4/0.5 GPT-4o 100-step sensitivity pilots and the retained-rollout
    audit without using test999 for selection.  The paired differences are nonsignificant, so the
    result supports local coefficient robustness rather than an empirical claim that 0.4 is best.
    The user cancelled a 300-step extension; the proposed 0.4 setting is instead justified as a
    pre-existing historical/mechanistic setting whose sensitivity was checked, and remains
    formally unfrozen pending acceptance.  Hardened Stage3 recovery so only classified transient
    Judge outages may relaunch; interrupts, resource/numeric, budget, identity/schema,
    source/contract, total-fallback and unknown failures stop.  Evidence:
    `docs/20260801_181414_stage3_fault_tolerance_hardening_and_coefficient_interpretation.md` and
    `protocol/stage3_fault_tolerance_hardening_20260801_181414.json`.  Supports R2-1/R2-4/R3-5/R3-6
    engineering reproducibility; no full Stage3 efficacy or expert-validity claim is made.
30. Corrected the coefficient interpretation after noting the monotonic 100-step point estimates
    and limited power: the pilots do not prove insensitivity.  Froze 0.4 only as the pre-existing
    three-event scale setting, for which zero/one/two/three event counts map to
    1.0/0.6/0.2/0.0, and retained 0.3/0.5 as sensitivity bounds.  Hardened the GPT-4o/AIGCBest
    transport so ambiguous requests are not resent and contract mismatches stop.  Evidence:
    `docs/20260801_182543_stage3_coefficient_frozen_aigcbest_hardening_and_launch_boundary.md` and
    `protocol/stage3_coefficient_freeze_and_aigcbest_hardening_20260801_182543.json`.  No efficacy
    result or claim of coefficient optimality is added.
31. Prepared the complete GPT-4o seed-42 Stage3 launcher with immutable parent/data/smoke hashes,
    bounded physical requests, classified checkpoint recovery and 500/1000/1500 snapshots.  Froze
    the user-ordered execution sequence and the rule that every hosted/local baseline must pass
    fixed behavioral smoke cases before full evaluation.  Evidence:
    `docs/20260801_184826_gpt4o_formal_launcher_and_execution_order_frozen.md` and
    `protocol/gpt4o_formal_launcher_frozen_20260801_184826.json`.  This is preparation, not an
    efficacy or baseline result.
32. Ran the fixed hosted-baseline behavioral gate before any full hosted evaluation. Qwen-VL-Plus
    and Haiku 4.5 passed exact-identity four-case smokes; an explicitly requested contemporary
    Grok 4.3 also passed. The historical Grok-4-Fast and Llama 3.2 90B aliases redirected to other
    model families, Llama 3.2 11B had no endpoint, and the dated Doubao endpoint returned 404, so
    none may be silently reported under the historical labels. The gate also caught and corrected
    a target-blind OmniMedVQA parsing failure for an exact option on the first response line, while
    preserving the legacy full-response similarity metric. GPT-5 mini schema-valid PathVQA
    semantic-judge smokes passed for the three runnable identities. Evidence:
    `docs/20260801_191001_hosted_baseline_behavioral_smoke_results.md` and
    `protocol/hosted_baseline_behavioral_smoke_manifest_20260801_191001.json`. Supports R1-7 and
    R2-3; smoke accuracy is not a model-selection or efficacy result.
33. Audited Doubao replacements and local Llama feasibility. Fixed exact identities
    `doubao-seed-1-6-vision-250815` and `doubao-seed-2-0-mini-260428` both passed the four-case
    visual adapter/scoring smoke and GPT-5 mini PathVQA judge path. The 1.6 Vision dated model is
    the lineage-based continuity replacement and began PathMMU test999; 2.0 Mini remains an
    optional stronger contemporary baseline. Llama 3.2 Vision local inference is technically
    supported by the prepared Mllama environment and available 8xA100 resources, but both official
    HF repositories return rejected gated-access 403s and the local directories contain no
    weights. Evidence: `docs/20260801_194327_doubao_replacement_and_local_llama_feasibility.md` and
    `protocol/doubao_replacement_llama_local_feasibility_20260801_194327.json`. Supports R1-7 and
    R2-3; no mirror/substitute weight is accepted as an official Llama baseline.
34. Corrected the formal GPT-4o Stage3 and PathVQA semantic-Judge recovery behavior after a billed
    truncated Stage3 response stopped the arm at step 820 and strict PathVQA handling stopped two
    resumable Judge passes.  The later user authorization permits bounded, fully audited retries
    for ambiguous transport, truncation and response-schema failures while keeping identity,
    refusal and non-transient contract failures terminal.  Stage3 recovery is from the validated
    step-800 eight-rank checkpoint; the first-attempt 320-token contract, penalty 0.4, reward
    events and total/consecutive fallback bounds are unchanged.  Failed PathVQA predictions are
    explicitly skipped rather than fabricated.  Evidence:
    `docs/20260802_134947_stage3_and_pathvqa_bounded_retry_recovery.md` and
    `protocol/stage3_pathvqa_bounded_retry_recovery_20260802_134947.json`.  Supports R2-4 and R3-7
    reproducibility; it is an engineering recovery record, not an efficacy result.
35. Added a verified off-machine backup for all currently static A100 evaluation artifacts.  The
    private HF dataset snapshot at revision `8e4348a96eb7164de0b39eb5d5a7018203674fc8`
    contains 241 files (338,397,446 bytes) covering external-VQA, hosted-baseline smoke/full,
    PathMMU diagnostic and SFT4000/Stage2 results.  Active PathVQA semantic-Judge outputs are
    deliberately excluded until completion and will be added as an immutable increment.  Source,
    summaries and exact recovery metadata remain on GitHub; in-progress Stage3 optimizer
    checkpoints remain local to avoid active-run I/O contention.  Evidence:
    `docs/20260802_141632_evaluation_results_private_hf_backup.md` and
    `protocol/evaluation_results_private_hf_backup_20260802_141632.json`.  This is provenance and
    disaster-recovery support for R1-7/R2-3/R2-4/R3-7, not a new efficacy result.
36. Prepared a fail-closed post-training selection pipeline for each complete Stage3 Judge arm.
    The pipeline verifies every file in the step-500/1000/1500 model-only snapshots, runs the
    fixed 385-record validation split on all three, verifies prediction hashes and source/model
    identity, and applies only the frozen accuracy/format/earlier-epoch tie-break.  PathMMU
    test999 is excluded.  Seven unit tests and shell/Python static checks pass; no GPU inference or
    paid call occurred during preparation.  Evidence:
    `docs/20260802_142834_stage3_post_training_validation_pipeline_prepared.md` and
    `protocol/stage3_post_training_validation_pipeline_20260802_142834.json`.  Supports R2-4 and
    R3-7 reproducibility; the pipeline is prepared but no full-arm validation result is claimed.
37. Reclassified the two Llama 3.2 Vision manuscript baselines from absent weights to
    ModelScope-mirror snapshots that are structurally complete and exactly match the official
    metadata weight-byte totals.  Official HF access remains gated and mirror provenance will be
    disclosed; byte-equivalence is not claimed before full shard hashing.  Added an unrun BF16
    Mllama adapter for fixed PathMMU, PathVQA and OmniMedVQA contracts, including eight-GPU 90B
    placement without quantization.  Evidence:
    `docs/20260802_143422_modelscope_llama32_snapshots_and_local_adapter_prepared.md` and
    `protocol/modelscope_llama32_local_adapter_20260802_143422.json`.  Supports R1-7/R2-3; smoke
    and full evaluation remain pending after GPT-4o Stage3.
38. Prepared an isolated official-interface DeepSeek-VL2 adapter for PathVQA and OmniMedVQA while
    preserving its prior PathMMU response-contract failure.  The external adapter uses the shared
    frozen prompts, deterministic scorer, source hashes and append-only recovery, but has not
    loaded weights or accessed either test set.  Both task-specific smokes remain mandatory after
    Stage3 releases the GPUs.  Evidence:
    `docs/20260802_143841_deepseek_vl2_external_vqa_adapter_prepared.md` and
    `protocol/deepseek_vl2_external_vqa_adapter_20260802_143841.json`.  Supports R1-7/R2-3;
    no DeepSeek external score is claimed yet.
39. Hardened the future fresh Kimi 2.6 Stage3 arm to use separately audited physical attempts,
    bounded retry for ambiguous transport and malformed/truncated/schema-invalid responses, and
    the pre-existing total-24/consecutive-4 structural-fallback circuit breaker. Identity,
    refusal, accounting and non-transient contract failures remain fail-closed. The smoke plus
    12,000 logical training judgments share a 12,361-physical-attempt ledger. Evidence:
    `docs/20260802_145005_kimi_stage3_bounded_retry_hardening_prepared.md` and
    `protocol/stage3_kimi26_bounded_retry_hardening_20260802_145005.json`. Supports R2-4/R3-7
    recovery reproducibility; the arm is unlaunched and no Kimi efficacy claim is made.
40. Prepared a machine-side half-hour audit and private-HF rolling disaster-recovery backup for
    active GPT-4o Stage3 and PathVQA semantic-Judge evidence. The auditor records process,
    checkpoint, budget/fallback, JSON-integrity, GPU and disk state but cannot control training;
    the classified supervisor remains the only recovery owner. Evidence:
    `docs/20260802_145416_periodic_audit_and_private_hf_backup_prepared.md` and
    `protocol/periodic_audit_private_hf_backup_prepared_20260802_145416.json`. Supports R2-4/R3-7
    provenance and recovery; no efficacy claim is made.
41. Enabled the `batchcom` minute-0/minute-30 cron audit and verified its first private-HF rolling
    backup at revision `0f0fb0561194284e6f7b40a201b74881e1628d9b`. The remote live tree contains
    25 files and 50,919,746 bytes, including exact manifests but no weights or credentials. The
    existing immutable 241-file completed-results snapshot is retained separately; final active
    outputs will receive another immutable increment. Evidence:
    `docs/20260802_145732_periodic_audit_and_private_hf_backup_enabled.md` and
    `protocol/periodic_audit_private_hf_backup_enabled_20260802_145732.json`. Supports R2-4/R3-7
    off-machine reproducibility; no efficacy claim is made.
42. Corrected the first live-backup scope after remote inspection showed that GNU `find -maxdepth`
    had excluded nested Stage3 reward-audit and Judge-ledger files. The first HF live revision is
    retained as a partial snapshot; the immutable 241-file completed-results snapshot was
    unaffected. Separate explicit traversals now cover the omitted paths and passed a no-network
    dry run. Evidence: `docs/20260802_145957_periodic_backup_nested_path_filter_correction.md` and
    `protocol/periodic_backup_nested_path_filter_correction_20260802_145957.json`. Supports honest
    R2-4/R3-7 provenance; corrected remote verification remains pending.
43. Observed the first scheduled 15:00 cron invocation and recorded its transient HF
    authentication connection reset without advancing the remote or local completed-upload state.
    Added bounded three-attempt authentication and upload retries with 15/45-second delays;
    exhaustion remains visible and terminal per invocation. Evidence:
    `docs/20260802_150132_periodic_hf_transport_retry_correction.md` and
    `protocol/periodic_hf_transport_retry_correction_20260802_150132.json`. Supports R3-7
    operational reproducibility; corrected remote upload verification remains pending.
44. Completed and recursively verified the corrected private-HF live backup at revision
    `06a0de6b8a940fddeeba0ee011212a2cfc0e8f6e`: all 24 Stage3 rank reward-audit files, three Judge
    ledger/state files and active PathVQA evidence are present. The 52-file, 102,977,672-byte remote
    tree includes exact manifests and no weights or credentials. Evidence:
    `docs/20260802_150310_corrected_periodic_hf_backup_verified.md` and
    `protocol/corrected_periodic_hf_backup_verified_20260802_150310.json`. Supports R2-4/R3-7
    off-machine reproducibility; mutable live state will still be frozen after completion.
45. Replaced the invalid PathVQA semantic completion heuristic with a fail-closed per-case
    finalizer. It reconstructs content cache keys for all successful free-form predictions,
    permits one paid verdict to score duplicate-content cases individually, keeps failed
    generations wrong, excludes two validated historical Haiku verdicts attached to failed
    predictions, and combines free-form semantics with target-blind yes/no exact scoring. Evidence:
    `docs/20260802_151307_pathvqa_semantic_finalizer_and_completion_gate_prepared.md` and
    `protocol/pathvqa_semantic_finalizer_prepared_20260802_151307.json`. Supports R2-3/R3-7
    evaluation validity; active passes remain incomplete and no final score is claimed.
46. Froze the local GPU-baseline smoke criterion as aggregate adapter behavior rather than answer
    accuracy. The verifier checks immutable artifact identity plus nonempty, parseable and
    generation-cap rates; wrong answers remain untouched and cannot trigger per-case repair or
    re-Judging. Five unit tests and three existing real output schemas passed. Evidence:
    `docs/20260802_152807_aggregate_smoke_policy_frozen.md` and
    `protocol/aggregate_local_baseline_smoke_policy_20260802_152807.json`. Supports R1-7/R2-3
    evaluation reproducibility; this is an engineering gate and makes no efficacy claim.
47. Corrected the periodic private-HF transport after a reset followed by a hung authentication
    process showed that attempt-count bounds alone were insufficient. Future authentication and
    upload calls have explicit wall-clock limits while preserving three attempts, content-hash
    verification and fail-closed state advancement. Only the exact stuck HF child was terminated;
    training and evaluation were untouched. Evidence:
    `docs/20260802_153555_periodic_hf_wallclock_timeout_correction.md` and
    `protocol/periodic_hf_wallclock_timeout_correction_20260802_153555.json`. Supports R3-7
    disaster-recovery reproducibility; no efficacy result is affected.
48. Recorded that the 15:30 backup invocation failed after its script was edited while the old
    Bash process was between retries. It produced no remote commit and advanced no success state;
    the local audit and prior verified remote revision remain intact. The complete corrected
    script then passed a no-network dry run, and future edits are prohibited while its lock is
    held. Evidence: `docs/20260802_153819_periodic_backup_live_edit_attempt_failure.md` and
    `protocol/periodic_backup_live_edit_failure_20260802_153819.json`. Supports R3-7 audit
    honesty; training and evaluation results are unaffected.
49. Removed the managed half-hour cron entry after the user cancelled proactive timed monitoring.
    The final 16:00 audit completed locally and its three wall-clock-bounded HF authentication
    attempts failed closed without advancing remote state. Training supervisor and both semantic
    Judge workers remain active; future status checks are on demand. Evidence:
    `docs/20260802_160833_periodic_monitoring_disabled_by_user.md` and
    `protocol/periodic_monitoring_disabled_20260802_160833.json`. Supports R3-7 audit honesty;
    this scheduling change does not alter any efficacy result.
50. Finalized the full Qwen-VL-Plus and Claude Haiku 4.5 PathVQA evaluations with deterministic
    yes/no scoring and frozen GPT-5-mini free-form semantic scoring, completing both models'
    three-task PathMMU/PathVQA/OmniMedVQA evaluation sets. Failed predictions remain wrong and no
    case was manually corrected. The complete 14-file semantic package was downloaded again and
    hash-verified from private-HF revision `6dd3ef5a73d3b7603991adff7b2fd17a9883e390`.
    Evidence: `docs/20260802_214734_hosted_semantic_evaluation_completed_and_backed_up.md` and
    `protocol/hosted_semantic_evaluation_completion_20260802_214734.json`. Supports R1-7/R2-3;
    PathMMU test999 remains explicitly classified as development diagnostic evidence.
51. Completed GPT-4o Stage3, selected epoch 2 solely on the frozen PathMMU validation385 split,
    and independently verified its 18-file model-only snapshot at private-HF revision
    `3ade3cffd46b64abc864ed9f271b47632810ec9c`. Full ModelScope-mirror Llama shard hashes and fixed
    aggregate GPU smokes are now recorded. Eligible Llama 11B and DeepSeek-VL2 full tasks are
    running; Llama 11B PathVQA is withheld after its predeclared cap-hit gate failed, without
    post-hoc prompt/parser repair. Evidence:
    `docs/20260803_011601_gpt4o_stage3_selected_uploaded_and_local_gpu_baselines_started.md` and
    `protocol/gpt4o_stage3_selection_hf_and_local_baselines_20260803_011601.json`. Supports
    R1-7/R2-3/R3-7 reproducibility; no queued local-baseline or Kimi efficacy result is claimed.

## Mandatory update rule

For every material future change:

1. add a timestamped `docs/YYYYMMDD_HHMMSS_*.md` record and update `docs/LATEST.md`;
2. update the relevant run/data/code manifest without rewriting failed history;
3. update every affected reviewer ID in this file with status, evidence path and remaining work;
4. distinguish engineering smoke, invalid attempt, formal validation evidence and final test evidence;
5. add exact revised-manuscript page/line references only after the manuscript edit exists;
6. never use test outputs to choose prompts, checkpoints, seeds, parsers, rewards or thresholds.
