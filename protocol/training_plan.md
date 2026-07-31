# Training protocol outline

## Machine responsibility split (frozen 2026-07-27)

- Existing formal machine: only the next separately frozen 4000-SFT training control. It performs
  no validation, checkpoint selection, baseline/OOD evaluation, final test or Stage 3. It saves all
  predeclared candidates for hash-verified transfer.
- New eight-A100 machine at `/home/dataset-assist-0/czy/wjy`: every validation/test, Stage 3,
  baseline/OOD evaluation and all other retained remaining experiments. Machine location never
  cancels an experiment.
- The two machines write separate execution branches (`codex/sft4000` and
  `codex/a100-stage3-eval`) and preserve immutable timestamped history.
- Weights/checkpoints/raw outputs move outside Git with exact transfer manifests. Code, small
  manifests and documentation move through Git.

## Stage 1 SFT

- Framework: LLaMA-Factory.
- Base: pinned Qwen2.5-VL-7B-Instruct revision from `base_model_manifest.json`.
- Data: CoT SFT files from `pathmmu_image_disjoint_v2`. These SFT files are byte-identical to v1;
  v2 only removes one exact-content duplicate QA from test.
- Validation: frozen 385-QA validation split.
- Formal seeds: 42, 43, 44 for the full-data model.
- Scaling subsets: 500, 1000, 2000, 3000 QA; seed 42 for initial scaling runs.
- Frozen scale-reporting decision (2026-07-20): do not rerun n=500/n=1000. Compare all four sizes
  at the fixed epoch-10 endpoint. Report the n=2000/n=3000 epoch curves separately as a training-
  duration/over-SFT analysis. Disclose that n=500/n=1000 used ZeRO-3 optimizer offload while
  n=2000/n=3000 used ZeRO-2 fused AdamW; do not attribute every difference solely to data size.
- Seed-42 Stage 2 candidate parent: validation-selected n=3000 epoch 3 (step 1125), subject to an
  exact parent/hash/load gate before Outcome GRPO.
- Pending 4000-SFT control: do not launch until a new protocol chooses between (a) fresh pinned base
  trained on all 4,000 SFT records and (b) selected n=3000 parent continued on an additional 1,000
  SFT records. They answer different scientific questions. The existing machine trains and retains
  candidates; A100 performs all validation selection.

The old 7B full-fine-tuning configuration OOMed on 4x4090 at AdamW state allocation.
The revised formal protocol is no longer method-selective: it requires full-parameter
language-model fine-tuning with the vision tower and multimodal projector frozen.
Use ZeRO-3 and optimizer offload as required by the formal hardware. LoRA/QLoRA may
only be used in separately labelled engineering diagnostics and must not enter the
formal scaling curve or paper results.

Before any formal SFT run, complete the one-step save/reload/resume gate specified in
`AGENTS.md` and record trainable/frozen parameter counts and representative tensor deltas.

## Stage 2 Outcome GRPO

- Parent: validation-selected Stage 1 checkpoint from the same formal seed lineage.
- Rewards: accuracy plus format.
- Scaling subsets: 250, 500, 1000 QA.
- Frozen priority decision (2026-07-20): because revision time is limited, run only the seed-42,
  n=1000 arm first from the n=3000 epoch-3 SFT parent. Other RL sizes/combinations and seeds are
  deferred, not silently cancelled or inferred from this one result.
- Before that formal arm, run a 50-step engineering pilot on the same n=1000 adapter. Save a full
  checkpoint at step 25, terminate only after the save, start a fresh process, resume to step 50,
  and evaluate all 385 validation records. The pilot is always `formal_result: false`.
- The formal arm may start automatically only if the frozen pilot gates in
  `stage2_priority_n1000_seed42_manifest_20260720_033024.json` all pass. It restarts from the SFT
  parent, trains three explicit epochs (1,500 optimizer steps), retains rolling resume state plus
  model-only epoch snapshots, validates epochs 1--3, and selects by accuracy, then format, then the
  earliest epoch. Test remains sealed.

## Stage 3 Process GRPO

- Status: the future execution sequence, coefficient sensitivity, multi-Judge roles, expert sample,
  recovery behavior and evaluation separation were approved on 2026-08-01 in
  `stage3_future_training_plan_approved_20260801_010917.json`.  The authorized offline sensitivity
  and two GPT-4o smokes are complete in `stage3_sensitivity_gpt4o_smoke_20260801_012103.json`.
  The user then approved the separate $18 gate for the matched 0.3/0.4/0.5 100-step pilots in
  `stage3_gpt4o_coefficient_pilots_launch_20260801_013252.json`; a formal run remains unauthorized.
- Parent, process unit, supervision source, label space and aggregation are frozen by the approved
  contract; the launch implementation must conform to it and pass a timestamped preflight.
- If an external judge is selected, every request, response, parser result, fallback, model version, decoding parameter, and cost record must be cached and auditable.
- No historical-recovery claim is allowed unless provenance proves it.
- Execution location: only the new eight-A100 machine.
- Inherited direction: keep the original image-grounded two-part reasoning rubric, emit structured
  judge events, and compute numerical penalties/reward deterministically outside the judge.
- Required sensitivity: the zero-cost 0.2--0.6 cache analysis is complete.  The matched online
  0.3/0.4/0.5 pilots are authorized under strict per-arm attempt and cost caps; do not retrofit their
  meaning or select on test.

## Evaluation

- Development decisions use validation only.
- Test is the corrected 999-QA v2 split and is executed only after the protocol and checkpoints are
  frozen. It is never used for prompt, checkpoint, parser, seed, reward, or threshold selection.
- OOD evaluation is stored separately from in-domain test results.
- All validation and test inference now executes on the A100 machine, including selection of
  transferred 4000-SFT candidates.
- Required external/cross-modal reruns include Chest CT, ISIC2020, Retinal OCT-C8 and Diabetic
  Retinopathy; PathVQA and approved additional pathology data are reported separately.
- Every original manuscript baseline remains an obligation. PLIP/CONCH/UNI and other approved
  pathology comparisons are additional rather than replacements. Historical API retirement must be
  disclosed and must not be hidden by relabeling a successor.
