# PathMMU test999 reclassified as a Stage 3 development diagnostic

Created: 2026-07-28 21:51:33 +08:00

This record captures an explicit change in scientific use before any PathMMU test999 inference was
started on the A100 machine.

## User decision

The user authorized running the corrected PathMMU v2 test999 on currently runnable baselines and
the validation-selected Stage 2 Outcome-GRPO checkpoint, then using the scores and bad cases to
adjust the Stage 3 training strategy.

This supersedes the earlier intended use of test999 as a final model-selection-blind test. It does
not erase or rewrite the older frozen protocol; it records why that protocol no longer describes
the new diagnostic use.

## Consequence

Once any test999 score or bad case is inspected for Stage 3 design, test999 is a development
diagnostic set. It may be reported as:

- pre-Stage-3 diagnostic performance;
- a development-set comparison of baselines, SFT and Outcome-GRPO;
- evidence used to motivate Stage 3 strategy.

It must not subsequently be described as:

- an untouched final test of the Stage 3 model;
- unbiased post-training confirmation;
- a set that played no role in prompt, reward, strategy or model design.

Any post-Stage-3 result on these same 999 records must be clearly labelled development-guided.

## Replacement final-confirmation policy

PathVQA and the four OmniMedVQA sources remain result-sealed on this machine and are reserved for
independent post-Stage-3 confirmation. No OOD inference or score inspection is authorized by this
record. If those sets are later used during Stage 3 development, another untouched final
confirmation set must be frozen first.

## Execution gates

- The approved common PathMMU prompt, BF16/no-quantization/greedy decoding, 1024-token cap and
  parser remain unchanged.
- Raw generations, source IDs, target choices, parsed choices, failures and hashes must be kept.
- A baseline enters test999 only after its validation-only adapter smoke succeeds.
- Results are diagnostic, not final Stage 3 claims.
- The exact Stage 2 checkpoint is the selected Outcome-GRPO n1000 seed-42 epoch-2/step-1000
  snapshot. It is currently absent from the A100 machine and may not be replaced by another epoch,
  SFT checkpoint or Hugging Face model.
- Stage 3 training remains unstarted by this decision.
