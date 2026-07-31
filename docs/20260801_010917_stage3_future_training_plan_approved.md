# Stage3 future training plan approved

Recorded at: `2026-08-01T01:09:17+08:00`

Status: approved scientific execution plan.  The current authorization covers protocol freezing,
the zero-cost offline penalty analysis, and two GPT-4o feasibility/cost smokes.  It does not yet
authorize the three paid 100-step coefficient arms or any complete 1,500-step run.

The longer user-facing working draft is stored outside Git at:

`/home/dataset-assist-0/czy/wjy/STAGE3_FUTURE_TRAINING_PLAN_20260801.md`

SHA-256: `7b185558a768c8edaf2f307406c32b24905dea0513cbc2408c34ffbd73788061`

## Scientific objectives

The retained experiment must establish whether process-aware GRPO improves over SFT and
outcome-only GRPO, whether the historical penalty coefficient 0.4 is defensible, whether the
effect is robust to Judge identity, and whether an LLM Judge agrees with a manageable expert
sample.  It must also break the manuscript's GPT-4o training/evaluation circularity.

## Current fixed starting point

- SFT3000: 582/999 (58.26%) on the already-used PathMMU test999 diagnostic.
- SFT4000 continuation control: 595/999 (59.56%); checkpoint uploaded to private Hugging Face.
- selected Stage2 Outcome-GRPO: 609/999 (60.96%).
- Stage3 parent for seed 42:
  `/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/transferred_checkpoints/outcome_grpo_n1000_seed42_epoch02_step1000`.
- Kimi formal attempt 01 stopped at 274/1500 and attempt 02 was stopped at 11/1500.  Neither has a
  resumable formal checkpoint.  Both remain engineering/failure evidence and are not paper results.
- Eight A100 80GB GPUs are currently idle; approximately 5.3 TiB filesystem capacity remains.

## Retained model matrix

| Arm | Definition | Role | Required extent |
| --- | --- | --- | --- |
| A | SFT3000 | original SFT reference | complete |
| B | SFT4000 continuation | extra-supervision control | complete |
| C | Stage2 Outcome-GRPO | no-process-reward RL control | complete |
| D | Stage3 GPT-4o | historically aligned reconstruction | full seed42 required |
| E | Stage3 Kimi 2.6 | independent contemporary Judge robustness | full seed42 required |
| F | Stage3 third Judge | additional Judge sensitivity | fixed-pool plus short pilot first |
| G | target-blind structural process reward | simple-rule ablation | 100--300 steps first |

No started scientific arm may be hidden merely because another Judge produces a higher test score.

## Penalty sensitivity

The six Judge events remain fixed.  For coefficient `c`, numerical rewards remain deterministic:

```text
integrity = max(0, 1 - c * missing_integrity_count)
knowledge = max(0, 1 - c * present_knowledge_error_count)
process = mean(integrity, knowledge)
```

First perform a zero-cost offline grid over `0.2, 0.3, 0.4, 0.5, 0.6` using every valid existing
non-smoke Kimi event.  Report reward distribution, floor/ceiling rates, tie counts, rank
correlations and within-GRPO-group advantage changes.  This is a diagnostic over prior generated
trajectories and is not a causal training comparison.

Then run GPT-4o seed42 training pilots at coefficients `0.3, 0.4, 0.5`, initially 100 optimizer
steps each from the same Stage2 parent and identical data order.  Validate all three on validation385.
If the result is ambiguous, extend all three fairly to 300 steps; never extend only the temporary
leader.  Select with validation accuracy first, paired uncertainty and format second, independent
process assessment/training stability next, and retain 0.4 on a practical tie for historical
fidelity.  Test999 is not a coefficient-selection input.

## Fixed Judge comparison pool and expert validation

Freeze 100 validation questions and four Stage2 completions per question, yielding 400 hashed
trajectories.  GPT-4o 2024-08-06, Kimi 2.6, one independent third Judge and the structural rule
must score the identical pool.  Compare the six events, aggregate reward, pairwise agreement,
latency, failure, retry and cost.

The default expert packet contains 40 stratified validation trajectories scored independently by
two blinded experts.  The strata are apparently correct, incomplete reasoning, medical-knowledge
error, and difficult/rare/ambiguous (10 each).  If resources cannot support this, expert A scores
50 and expert B overlaps on 20; the reduced inter-rater evidence must be disclosed.  Report event
Kappa, aggregate ICC/Spearman and bootstrap confidence intervals.  This is a small-scale expert
validation, not a clinical-validity claim.

Judge selection is based on expert/validation agreement, never on which Judge gives PathVLM-R1
the highest test score.  GPT-4o remains the historical reconstruction even if a different Judge
becomes an expert-aligned revised arm.

## GPT-4o smoke and budget gate

Request the fixed `gpt-4o-2024-08-06` identifier through AIGCBest with temperature zero, strict
six-event JSON, fixed prompt/seed/output bound, and no automatic model fallback.  Preserve the
requested and returned strings, request ID, safe headers, usage, latency, retry and public pricing
snapshot.  A matching returned string is not independent upstream-provenance proof.

Run exactly two pre-training smokes under the current authorization:

1. one synthetic image contract smoke;
2. one non-test validation pathology-image smoke for representative token/cost measurement.

The migration estimate from 2,197 prior Kimi calls is 1,437.67 input and 226.18 output tokens per
call.  At the current AIGCBest GPT-4o ratio this projects USD 4.68 per 100-step/800-call arm,
USD 14.05 for three 100-step arms, and USD 70.27 for a 12,000-call full arm.  These estimates are
not authorizations.  New hard ceilings must be frozen after the two smokes.

Gateway technical availability and permission to reuse outputs for training are separate gates.

## Formal Stage3 contract

- parent: fixed Stage2 Outcome-GRPO epoch2 step1000;
- data: frozen image-disjoint RL1000;
- epochs/steps: 3/1500;
- training/data seed for first comparison: 42/42;
- global completion batch: 8;
- generations per prompt: 4;
- maximum training Judge requests: 12,000;
- learning rate: `1e-6`;
- beta: `0.04`;
- local completion maximum: 192 tokens;
- full language model trainable;
- vision tower and multimodal projector frozen.

Save full resumable state every 100 optimizer steps, retain the latest two, and retain model-only
epoch snapshots at 500, 1000 and 1500.  Validate every epoch snapshot and select by validation
accuracy, format, then earliest epoch.  A full checkpoint is approximately 102 GiB; two retained
resume checkpoints are safe under current capacity.

## Retry, fallback and recovery

- Retry explicit transient HTTP 408/429/500/502/503/529 with 15/45/90-second delays.
- Accept an incomplete HTTP body only if its partial bytes already form one complete valid JSON
  document.
- Do not resend an ambiguous connection/HTTP-200 truncation in the same process.
- After bounded request failure, permit only the target-blind structural fallback capped at process
  reward 0.5.
- Permit at most 24 fallback completions for the run and at most four consecutively.
- The fifth consecutive fallback stops the segment and enters checkpoint recovery.
- Use a new cache namespace for every recovered segment; keep the budget ledger cumulative.
- Resume only the newest structurally complete 100-step checkpoint.
- Permit at most three recoveries (four launches total).

Stop on model/provider identity mismatch, source/hash drift, budget exhaustion, fallback-limit
breach, missing valid checkpoint, recovery-limit breach, persistent zero reward variance, NaN/Inf
or OOM.

## Full Judge arms and later seeds

After coefficient selection, run complete seed42 arms for both fixed GPT-4o and Kimi 2.6 from the
same Stage2 parent and with the selected coefficient.  Report both.  A third Judge receives a full
arm only if the fixed-pool/expert/100-step evidence shows a scientifically material difference or
the reviewer response requires it.  The rule arm receives 100--300 steps first and may be extended
if compute permits.

Only after the Judge, coefficient, prompt and epoch rule are frozen, run seeds 43 and 44 for the
final primary pipeline.  Preferred evidence uses complete SFT3000 -> Stage2 -> Stage3 lineages for
each seed.  A time-limited design that changes only the Stage3 seed must be labelled Stage3
stochasticity rather than full-pipeline multi-seed evidence.  Ablations do not all require three
seeds.

## Evaluation and circularity control

- Development selection uses validation385 only.
- Continue to report the already-used test999, but do not claim it was untouched and do not use it
  for coefficient, Judge, checkpoint, seed, parser, prompt or threshold selection.
- PathMMU uses deterministic exact accuracy and paired statistics.
- PathVQA reports yes/no accuracy and, for free-form rows, strict exact match, macro token F1,
  BLEU and a supplemental semantic Judge score.
- OmniMedVQA reports both raw official QA scoring and target-blind final-answer-aligned scoring.
- Repeat the four manuscript OOD evaluations: Chest CT, ISIC2020, Retinal OCT-C8 and diabetic
  retinopathy.
- The final semantic/reasoning evaluator must be independent of the training Judge family where
  possible; deterministic task accuracy remains primary.

Report bootstrap confidence intervals, paired McNemar tests, per-seed results/mean/std, expert
agreement, and prespecified bad-case strata.  Do not select a favorable seed or Judge.

## Execution order and current boundary

1. Freeze this protocol and push it to GitHub.
2. Complete the reproducible offline coefficient analysis.
3. Build the fixed validation Judge pool.
4. Run the two approved GPT-4o smokes.
5. Report measured cost and request a hard ceiling for the three 100-step arms.
6. Run the coefficient pilots only after that ceiling is approved.
7. Select the coefficient on validation, then separately gate the full GPT-4o and Kimi arms.
8. Run conditional third-Judge/rule ablations, freeze the primary pipeline, add seeds 43/44, then
   perform uniform evaluation, expert validation, statistics and release.

No 100-step paid training, full Stage3 run or new test inference is authorized by this record.
