# GPT-5 mini PathVQA judge comparison

Recorded at: 2026-08-01 00:02:16 CST

## Contract

GPT-5 mini was evaluated on the same 30-case balanced, human-checkable PathVQA prelabel set and
the same semantic-judge prompt previously used for GPT-4.1 mini.  This is an engineering
calibration, not pathologist-annotated final gold.

The request used the AIGCBest `gpt-5-mini` alias with `reasoning_effort=minimal`, structured JSON
output, and a 256-token completion ceiling.  The gateway reported the fixed snapshot string
`gpt-5-mini-2025-08-07`.  OpenAI documents that snapshot, Chat Completions support, structured
outputs, and USD 0.25/M input plus USD 2.00/M output pricing on the GPT-5 mini model page.  A
matching response string is not independent proof of the gateway's upstream provenance, so this
run must be described as gateway-served rather than as a verified direct OpenAI execution.

One response exceeded the prompt's requested 120-character reason length and was rejected before
being written.  The local parser was relaxed to accept a bounded 240-character reason without
changing the prompt, and the 30-case set was then completed.  The rejected request may have been
billed; 31 paid calls are conservatively counted for the test.

## Results

| Metric | GPT-4.1 mini | GPT-5 mini |
| --- | ---: | ---: |
| Accuracy | 28/30 (93.33%) | 22/30 (73.33%) |
| TP / TN / FP / FN | 13 / 15 / 0 / 2 | 7 / 15 / 0 / 8 |
| Mean input tokens | 320.43 | 319.43 |
| Mean output tokens | 27.00 | 41.13 |
| Mean reasoning tokens | 0 | 0 |
| Mean latency | 3.18 s | 11.58 s |
| Estimated USD/request | 0.0001714 | 0.0001621 |

GPT-5 mini retained zero false positives but added six false negatives.  The additional failures
were all candidates that included the reference's core answer (`heart`, `foot`, `hand`, `lung`,
or `kidney`) plus extra description.  Under the current prompt, GPT-5 mini interpreted absent or
unverified extra detail as disqualifying more aggressively than GPT-4.1 mini.

The observed GPT-5 mini output was 52% longer, so its realized cost advantage was only about 5.4%
despite cheaper input tokens.  It was 3.64 times slower.  Estimated full costs at this observed
usage are USD 0.54 for one model's 3,357 free-form answers, USD 5.44 for the current ten models,
or USD 8.71 for sixteen models.

## Decision

Do not replace GPT-4.1 mini with GPT-5 mini under the current judge prompt.  The comparison does
not establish that GPT-5 mini is generally weaker; it establishes that this model/prompt pair is
over-conservative for the intended semantic-equivalence policy.  A revised prompt could explicitly
forbid rejecting extra detail merely because it is absent from the short reference, but that
prompt must be tuned on this calibration set and then evaluated on a separate held-out set before
any full paid run.

No full PathVQA judge run was started.

Artifact:

`/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/aigcbest_pathvqa_judge_smoke_20260731/gpt-5-mini_manual30.jsonl`

SHA-256: `25086cdb277fafee551f3e345d63fa89614047d22d68baca3a5fa958d1ba4482`
