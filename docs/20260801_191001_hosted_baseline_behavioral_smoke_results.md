# Hosted baseline behavioral smoke results

Recorded at: `2026-08-01T19:10:01+08:00`

No full hosted baseline evaluation was started before this gate. The fixed smoke uses the same four cases for every model: PathMMU validation index 0, PathVQA free-form index 1, PathVQA yes/no index 3, and OmniMedVQA index 0. Accuracy was not used to choose a model or prompt.

## Exact-identity results

| Requested model | Result | Observed identity/issue |
|---|---|---|
| `qwen-vl-plus` | passed 4/4 | served exactly `qwen-vl-plus` |
| `claude-haiku-4-5-20251001` | passed 4/4 | served exactly the requested ID |
| `doubao-1.5-vision-pro-250328` | failed at first case | terminal 404: endpoint absent or inaccessible |
| `grok-4-fast-non-reasoning` | failed at first case | redirected to `grok-4.3` |
| `llama-3.2-11b-vision-instruct` | failed at first case | no upstream endpoint; outer HTTP 500, inner code 404 |
| `llama-3.2-90b-vision-instruct` | failed at first case | redirected to Llama 4 Maverick FP8 |

The catalog alias `Doubao-1.5-vision-pro-32k` was also checked and served `doubao-1-5-vision-pro-32k-250115`; it therefore remains a candidate explicit alias mapping rather than an exact-identity pass. The independently requested contemporary `grok-4.3` passed all four cases with exact identity. It must be labelled as a contemporary replacement/additional baseline and not as a reproduced Grok-4-Fast result.

## Scoring-path checks

- PathVQA free-form generation was converted into a semantic-judge input. GPT-5 mini served the pinned `gpt-5-mini-2025-08-07` snapshot and returned schema-valid judgments for Qwen-VL-Plus, Haiku 4.5 and Grok 4.3.
- PathVQA yes/no used deterministic extraction and exact scoring without an LLM judge.
- OmniMedVQA used both the repository-style full-response similarity and the contract-aligned answer score.
- The Grok 4.3 OmniMedVQA response began with the exact option text `CT` and then hit its generation cap while explaining. The legacy full-response similarity was distracted by a later option mention and selected A. A target-blind symmetric correction now checks whether the first nonempty line exactly equals one unique candidate option before falling back to similarity. On the preserved raw response, the aligned score selects D while the legacy score remains preserved as A.
- PathMMU answer parsing and format scoring both executed. A model that gives a parseable choice without the required `<think>` wrapper receives the parsed choice and a format score of zero; the prompt is not adapted based on smoke accuracy.

The smoke runner preserves raw provider JSON, requested/served model IDs, response IDs, finish reasons, usage, explicit transient retries, terminal failures, prompts, images and all local scoring outputs. Ambiguous transport failures are never resent automatically.

## Stage 3 concurrent status

Formal GPT-4o Stage 3 started from the frozen Stage2 checkpoint under repository commit `4b7b1f0`. At the smoke audit checkpoint it had reached approximately step 18, completed 152 judge requests, committed about USD 0.7623, had no unresolved reservations and no recovery event. This is progress evidence, not an efficacy result.

Evidence root:

`/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/hosted_baseline_behavioral_smoke_20260801`

Full evaluation is allowed only for exact identities that passed, or for a separately and explicitly labelled contemporary/alias mapping. Failed historical identities cannot be silently replaced.
