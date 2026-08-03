# Post-Kimi closed-question pipeline prepared and local baselines backed up

Recorded at: `2026-08-04 00:51:48 CST`

## Outcome

All currently useful non-GPU work was completed, then work stopped without polling or launching
GPU inference. Kimi Stage3 remained under its existing bounded supervisor. At the balance audit,
the run was at 178/1500, its ledger had 1,600 completed requests, committed spend was
`$2.69937744`, the local `$26.71782984` ceiling was not breached, and the account-level available
credit exceeded the remaining projected run cost. Account credit itself is not recorded in Git.

No cron job, foreground wait loop, tmux waiter, evaluation inference, validation inference or
visual-fidelity model run was started by this preparation.

## Current reporting correction

`scripts/run_stage3_selected_full_evaluations.sh` now follows the user's current reporting scope:

- run fixed 16-case adapter smokes and full evaluations for PathMMU, the 3,362 PathVQA yes/no
  questions and OmniMedVQA;
- evaluate the selected GPT-4o and Kimi Stage3 checkpoints under the same frozen contracts;
- explicitly select `answer_type=yes_no` before PathVQA inference, while retaining and checking the
  original 6,719-record data hash;
- exclude all 3,357 PathVQA free-form questions, semantic Judge work and combined scoring;
- retain PathVQA free-form as an explicitly postponed independent phase.

This changes only the unrun post-training evaluation script. It does not alter Kimi training,
rewards, checkpoints, retry/fallback behavior, data, Judge identity or budget ledger.

## Post-Kimi execution entry point

`scripts/run_post_kimi_closed_evaluations_and_visuals.sh` is a single-shot, fail-closed entry point.
It does not wait for Kimi. When invoked after training, it requires the final supervisor event to
be `training_completed`, validates step-500/1000/1500 snapshot state and the budget ledger, then:

1. runs frozen validation385 on all three Kimi epoch snapshots and selects by the existing rule;
2. runs six Stage3 evaluations: GPT-4o/Kimi by PathMMU/PathVQA yes/no/OmniMedVQA;
3. runs the five frozen visual-fidelity arms on five GPUs;
4. verifies every 24-case output and all 25 figures per arm;
5. creates a five-arm JSON/CSV/Markdown/PNG comparison package;
6. writes `completed.json` only after every gate succeeds.

It stops immediately if Kimi is incomplete, a selected checkpoint is invalid, a GPU is not idle,
an evaluation count/hash/source gate fails, or a visualization output is partial.

## Visualization verification

Two reusable tools were added:

- `scripts/verify_visual_fidelity_run.py` verifies model/panel identity, 24 ordered unique cases,
  36 patch scores per case, result hashes and the exact 25-figure inventory.
- `scripts/summarize_visual_fidelity_runs.py` requires exactly SFT3000, SFT4000, Stage2,
  GPT-4o Stage3 and Kimi Stage3, then creates the comparison package while retaining the narrow
  perturbation-fidelity claim. It does not label the outputs as attention, causal localization or
  lesion IoU.

The shared Qwen external-VQA runner and integrity gates now carry an explicit
`pathvqa_answer_scope=yes_no_only` field. This prevents a yes/no-only run from being mistaken for a
complete mixed PathVQA run and proves that free-form records were excluded before batching.

Shell parsing and Python compilation passed. Twenty-four CPU tests covering PathVQA scope,
evaluation integrity, panel preparation,
perturbations, verification and five-arm aggregation passed. No real model was loaded.

## Immutable local-baseline backup

The completed Aug 3 Llama 3.2 Vision and DeepSeek-VL2 run tree was uploaded to the private dataset
`Freddie1946/PathVLM-R1-Revision-Evaluation-Results` under:

`snapshots/20260804_local_gpu_baselines/`

The fixed HF revision is `e71610e68ddc7e2df59e081247f075aec9755f93`. The complete remote
prefix was downloaded again and compared recursively with the local source:

- files: 79;
- bytes: 58,862,418;
- aggregate SHA-256 over sorted relative-path/file-hash rows:
  `55aaf72168d979b37519238314b5ca87dd04ef87d7560c20626fef0d78c08085`;
- recursive local/remote difference: none.

The snapshot contains raw predictions, logs, metrics, smoke gates, full-integrity verifications and
sequence events. It contains no model weights or credentials.

## Remaining GPU-dependent work

- Let the existing Kimi supervisor finish; no Codex monitoring is active.
- After completion, invoke the single-shot post-Kimi entry point once.
- Verify and upload the selected Kimi model snapshot.
- Back up final Stage3 evaluations and visualization outputs as a new immutable HF increment.
- Produce final statistics, bad-case analysis and manuscript-ready figures only from verified
  completed artifacts. PathVQA free-form remains postponed.
