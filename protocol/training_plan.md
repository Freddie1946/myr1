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
- Execution update (2026-08-01): 0.3 and 0.4 completed 100 steps.  The 0.5 arm stopped after step
  99 because two prior retry attempts left insufficient room under the 800-physical-attempt ceiling
  for all eight final logical judgments.  It has no checkpoint and was not restarted.  See
  `gpt4o_pilot_progress_0p5_attempt_cap_stop_20260801_143832.json`.
- Recovery update (2026-08-01): the user approved conservative settlement of the six unresolved
  reservations and one fresh, no-cache 0.5 rerun with 800 logical judgments, at most 824 physical
  attempts and a $5.50 cap.  See `gpt4o_0p5_fresh_recovery_approved_20260801_144436.json`.
- Coefficient interpretation update (2026-08-01): all three 100-step pilots and matched
  validation runs completed, with no statistically significant paired difference.  The user
  cancelled the 300-step extension.  These pilots establish local sensitivity/robustness, not an
  empirically optimal coefficient.  The proposed use of 0.4 is as the prior historical and
  reward-geometry setting; it remains formally unfrozen pending acceptance of that justification.
- Coefficient freeze correction (2026-08-01): the 100-step point estimates are monotonic and the
  short single-seed experiment cannot prove insensitivity.  The user accepted the three-event
  scale rationale and froze 0.4 as an a-priori mechanistic/historical setting rather than an
  empirically optimal pilot choice.  The 300-step extension is cancelled.  See
  `stage3_coefficient_freeze_and_aigcbest_hardening_20260801_182543.json`.
- Recovery hardening update (2026-08-01): automatic relaunch is now whitelist-only.  Recognized
  transient Judge transport and the consecutive-outage circuit breaker may recover; user
  interrupt, OOM/disk/numeric, budget, identity/schema, source/contract, total-fallback and
  unknown failures stop.  Only a fresh remote Judge success resets the outage counter, and the
  automatic recovery cap cannot exceed three.  See
  `stage3_fault_tolerance_hardening_20260801_181414.json`.
- GPT-4o client hardening update (2026-08-01): AIGCBest retries only explicit transient HTTP
  statuses; ambiguous connection/body failures are not resent, billed identity/schema failures
  stop, and only a fresh remote success resets the outage counter.  The complete GPT-4o arm still
  requires a separate formal budget.
- Bounded-retry correction (2026-08-02): after the formal arm stopped at step 820 on a single
  `finish_reason=length` response, the user explicitly superseded the no-resend rule for transient
  failures.  Ambiguous transport and response JSON/schema/truncation failures now receive at most
  four physical attempts, all conservatively audited; response-validation retries use token caps
  512 then 768 after the unchanged 320-token first attempt.  Exhaustion enters the unchanged
  total-24/consecutive-4 structural fallback limiter.  Identity mismatch, refusal and proven
  non-transient contract failures remain terminal.  Recovery uses the independently validated
  eight-rank checkpoint at step 800.  See
  `stage3_pathvqa_bounded_retry_recovery_20260802_134947.json`.
- Formal execution update (2026-08-01): the user removed the USD stopping threshold and ordered
  GPT-4o Stage3 plus non-GPU hosted baselines first, GPU-local baselines second and Kimi Stage3
  last.  The physical request cap, bounded retries/fallback and full cost audit remain mandatory.
  The prepared GPT-4o launcher uses 12,000 logical and at most 12,360 physical attempts; its USD
  247.20 ledger value is technical capacity, not an expected spend.  Every baseline receives a
  fixed behavioral smoke before full evaluation.  See
  `gpt4o_formal_launcher_frozen_20260801_184826.json`.
- Post-training validation preparation (2026-08-02): a fail-closed snapshot verifier and formal
  three-epoch validation385 wrapper are prepared.  After training completion, model-only snapshots
  at 500/1000/1500 are verified file-by-file, evaluated separately, and selected by higher
  accuracy, then higher strict-format accuracy, then the earlier epoch.  Test999 remains excluded.
  See `stage3_post_training_validation_pipeline_20260802_142834.json`.
- Kimi bounded-retry preparation (2026-08-02): the fresh arm retries ambiguous transport and
  retryable response-validation failures at most three times after the first physical request,
  records each physical attempt separately, and only then enters the unchanged bounded structural
  fallback. Identity/refusal/accounting/non-transient failures remain terminal. The shared
  smoke-plus-training ledger ceiling is 12,361 physical attempts; the arm remains unlaunched and
  last in the approved execution order. See
  `stage3_kimi26_bounded_retry_hardening_20260802_145005.json`.

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
- Llama execution update (2026-08-02): ModelScope-sourced Llama 3.2 Vision 11B/90B directories are
  structurally complete and match the previously recorded official weight-byte totals.  The local
  Mllama adapter is prepared but unrun.  Official HF access remains gated, so mirror provenance
  must be disclosed.  Full shard hashes and fixed PathMMU/PathVQA/OmniMedVQA smokes are mandatory
  before the post-Stage3 full runs.  See `modelscope_llama32_local_adapter_20260802_143422.json`.
- DeepSeek-VL2 external-evaluation update (2026-08-02): its native PathVQA/OmniMedVQA adapter is
  prepared but unrun.  Each task requires a fixed smoke after GPT-4o Stage3 releases the GPUs.
  The prior PathMMU extractable-answer smoke failure remains terminal for that contract and is not
  converted into a post-test parser.  See `deepseek_vl2_external_vqa_adapter_20260802_143841.json`.
