# GPT-4.1-mini calibration and AIGCBest budget estimate

Recorded at: 2026-07-31 23:38:57 CST

## Meaning of the Stage3 rule reward ceiling

Stage3 sums three equally weighted component rewards:

`total_reward = accuracy_reward + format_reward + process_reward`

Accuracy and format are each binary in `[0, 1]`; the normal LLM process reward is in `[0, 1]`.
The normal maximum total is therefore 3.0.  The local outage rule replaces only the process
component and caps that replacement at 0.5.  It does not cap total reward at 0.5: a correct,
well-formatted completion using the structural fallback can receive at most 2.5.  An incorrect but
well-formatted completion with maximum fallback structure can receive 1.5.

The cap reflects missing evidence: the local rule can detect reasoning structure but cannot verify
image-grounded medical correctness.  Its separate total/consecutive usage caps remain 24 and 4.

## GPT-4.1-mini calibration

A balanced 30-case, human-checkable prelabel set was selected from actual Stage2 PathVQA
free-form completions: 15 expected correct and 15 expected incorrect.  It is an engineering
calibration set, not a pathologist-annotated final gold set.

The gateway expanded `gpt-4.1-mini` to the fixed served identifier
`gpt-4.1-mini-2025-04-14`.  One initial request returned under that alias and was rejected by the
old exact-route check before it could be written; it may have been billed.  The explicit alias
mapping was then frozen and the 30-case run completed.

Results:

- accuracy: 28/30 = 93.33%
- correct-class precision: 100%
- correct-class recall: 86.67%
- confusion: TP 13, TN 15, FP 0, FN 2
- mean latency: 3.18 seconds
- mean usage: 320.43 input and 27 output tokens

Both errors were conservative false negatives.  Index 519 contains `Capillary Lumen` but was
called an omission; index 4772 explicitly says `bone tissue` but was rejected because the model
treated the additional `cancerous cells` phrase as a materially false claim without image
evidence.  This suggests the mini judge is suitable as a cost-effective primary screen after a
prompt clarification, but it still needs an independent held-out expert or strong-model audit.
The observed 30-case accuracy must not be reported as final benchmark accuracy.

At the public default-group rates (approximately USD 0.40/M input and USD 1.60/M output), observed
usage estimates to USD 0.0001714 per free-form judgment:

- one model's 3,357 PathVQA free-form answers: USD 0.58
- all ten currently available PathVQA prediction sets: USD 5.75
- the current ten plus six planned hosted baselines: USD 9.20

Recommended hard caps with contingency are USD 7 for the current ten, or USD 11 after the six
hosted baselines are added.  OmniMedVQA and PathVQA yes/no do not call the LLM judge.

## Hosted vision-baseline estimate

The following provisional estimate applies each model to all 6,719 PathVQA and 8,518 OmniMedVQA
rows.  It uses the observed Qwen smoke request (177 input/5 output tokens) for OmniMedVQA and a
150-input/10-output assumption for PathVQA.  Image tokenization varies by provider, so these are
planning numbers, not bills.

| Model | Nominal USD | 15% contingency USD | Suggested hard cap USD |
| --- | ---: | ---: | ---: |
| qwen-vl-plus | 2.23 | 2.57 | 3 |
| doubao-1.5-vision-pro-250328 | 8.53 | 9.81 | 10 |
| grok-4-fast-non-reasoning | 0.56 | 0.64 | 1 |
| claude-haiku-4-5-20251001 | 3.06 | 3.52 | 4 |
| llama-3.2-11b-vision-instruct | 5.25 | 6.04 | 7 |
| llama-3.2-90b-vision-instruct | 16.08 | 18.49 | 19 |
| **Total** | **35.72** | **41.08** | **44** |

An overall authorization of USD 45 would cover these six planning caps, but execution should be
sequential with the per-model limits above.  After the first complete Qwen run, its measured token
distribution should update the remaining forecast.  No full hosted baseline was started here.

Artifacts:

- calibration labels: `protocol/pathvqa_gpt41mini_manual_calibration_cases_20260731.json`
  (SHA-256 `0880928edb99a2d987dad9ee31f0e50de7a2315eaa8ac3fed7035be3d89b8535`)
- GPT output:
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/aigcbest_pathvqa_judge_smoke_20260731/gpt-4.1-mini_manual30.jsonl`
  (SHA-256 `7a237935540bc17d239bc28777cf06b0199b02af0a278544ee9348b3ec10ff1f`)
