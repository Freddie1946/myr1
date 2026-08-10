# External-VQA output-failure correction and OmniMedVQA rerun launch

## Outcome

The OOD degradation is a mixture of evaluator/output-contract failure and real model error.  A
target-blind PathVQA yes/no parser correction recovers some answers, but all trained checkpoints
remain well below the untouched Qwen2.5-VL-7B base.  OmniMedVQA's previous 64-token results are too
heavily truncated to support the same conclusion: a 192-token strict-final-answer corrective rerun
has passed both adapter smokes and is now running for the selected GPT-4o and Grok Stage3 models.

This work was triggered after inspection of test outputs.  It is therefore a post-hoc evaluation
correction/sensitivity analysis, not a preregistered primary test result.  The old outputs and old
scores remain immutable and the corrected values must not be used to choose prompts, parsers,
checkpoints or training settings.

## PathVQA yes/no correction

The old extractor returned the entire content of an answer tag.  Consequently,
`<answer>B) No</answer>` was compared literally with `no` and scored wrong.  V3 accepts only an
optional leading A--D label followed immediately by an unambiguous leading `Yes` or `No` inside the
final answer tag.  It never searches the reasoning body for either target word.

| Model | Legacy aligned | V3 aligned | V3 parseable | Correct answers recovered |
|---|---:|---:|---:|---:|
| Base Qwen2.5-VL-7B | 66.42% | 66.42% | 99.55% | 0 |
| SFT3000 | 51.40% | 52.65% | 90.54% | 42 |
| SFT4000 control | 30.34% | 36.67% | 61.78% | 213 |
| Stage2 Outcome-RL | 49.64% | 52.77% | 89.80% | 105 |
| Stage3 GPT-4o checkpoint-1000 | 52.26% | 53.30% | 91.43% | 35 |
| Stage3 Grok checkpoint-1500 | 52.29% | 53.45% | 91.23% | 39 |

No legacy-correct item regressed under V3.  The first rescoring implementation is preserved as
invalid because early prediction rows did not contain the old aligned-score field and the script
mistakenly interpreted absence as false.  Attempt02 reproduces the old parser directly and is the
only valid V3 audit output.

The scientific implication is not that formatting explains the OOD result.  It explains about
1--3 points for most trained models and 6.3 points for the unusually format-damaged SFT4000
control, while the remaining gap to the base model is still 13--30 points.  Existing bad-case
counts also show many parseable but clinically wrong answers.  The most plausible current
explanation is language-side output-schema overfitting/catastrophic forgetting from homogeneous
PathMMU multiple-choice training, compounded by frozen vision/projector modules; Stage3 improves
the in-domain objective but does not restore this lost OOD behavior.

Valid rescoring index:

`/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/reports/pathvqa_yesno_parser_v3_20260810_attempt02/index_v3.json`

SHA-256: `1cc380c7ca3577907ed3d35d2fd19b25c6f08e387368adc27ce4c647caf66730`

## OmniMedVQA corrective contract

The prior selected Stage3 runs reached the 64-token cap on about 81% of GPT-4o outputs and 78% of
Grok outputs.  Many completions discussed an option but ended before an explicit final answer; the
old SequenceMatcher fallback could then assign an arbitrary candidate from incomplete reasoning.

The fixed V3 contract keeps the prompt and deterministic decoding unchanged, raises only
`max_new_tokens` from 64 to 192, and makes the corrective primary endpoint strict final-option
accuracy.  The strict parser accepts only an explicit leading A--D label or a first non-empty final
answer line exactly matching a candidate after conservative normalization.  No fuzzy fallback is
used.  Responses without such an answer, including any 192-token truncation, count as unresolved
and wrong.  Legacy SequenceMatcher metrics remain secondary sensitivity outputs.

Both predeclared 16-case engineering smokes passed without using accuracy as a gate:

| Model | Strict answer coverage | Cap-hit rate | Mean tokens | Max tokens |
|---|---:|---:|---:|---:|
| Stage3 GPT-4o checkpoint-1000 | 100% | 0% | 84.56 | 122 |
| Stage3 Grok checkpoint-1500 | 100% | 0% | 85.56 | 105 |

For transparency only, both smokes happened to score 11/16.  That value did not influence the
parser, threshold or full-run decision.

## Full rerun status and safety

The detached sequence started at `2026-08-10T08:53:11Z` on GPU 7 with batch size 1.  It runs GPT-4o
checkpoint-1000 first and Grok checkpoint-1500 second.  At record time the GPT arm had written 16
ordered predictions and both the evaluator and the data-ratio training workers were live.

With the evaluator loaded, GPU 7 used about 57.9 GiB-equivalent MiB accounting and retained about
23.2 GiB free.  No OOM or training error occurred.  The concurrent 4000-rule-RL arm continued; the
extra evaluator caused only a temporary throughput reduction.  Based on the measured smokes and
initial full rows, the sequential rerun is expected to require about 23--25 hours.  A separate
completion manifest is required before any full corrected OmniMedVQA score is reported.

Run root:

`/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/omnimedvqa_corrective_v3_192_20260810`

## Evidence

- Frozen correction contract:
  `protocol/external_vqa_output_contract_correction_v3_20260810.json`
- Launch/smoke manifest:
  `protocol/external_vqa_output_contract_correction_v3_launch_20260810.json`
- Parser and scorer: `scripts/external_vqa_contract.py`
- Rescoring audit: `scripts/rescore_pathvqa_yesno_v3.py`
- Corrective runner: `scripts/run_external_vqa_qwen.py`
- Safe sequence: `scripts/run_omnimedvqa_corrective_v3_sequence.sh`
- Smoke verifier: `scripts/verify_omnimed_corrective_v3_smoke.py`
- Tests: 24 targeted CPU tests passed.
