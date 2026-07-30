# OpenRouter hosted-baseline and Stage3-judge audit

Timestamp: `2026-07-29T23:15:50+08:00`

## Scope and execution boundary

This is a read-only, no-credential audit of OpenRouter's public model and endpoint catalogs. No
paid request, model inference, Stage3 training, test inference, prompt smoke, credential lookup or
account mutation was performed. OpenRouter availability is time-varying; the live endpoint results
below are a timestamped planning observation rather than a permanent availability guarantee.

Authoritative live catalog interfaces queried:

- `https://openrouter.ai/api/v1/models`
- `https://openrouter.ai/api/v1/models/:author/:slug/endpoints`
- `https://openrouter.ai/api/v1/endpoints/zdr`

An OpenRouter model page can remain visible after serving endpoints disappear. For execution
readiness, this audit therefore requires a non-empty live `endpoints` array rather than relying on a
visible historical model page.

## Manuscript hosted and inaccessible-weight baselines

| Manuscript row | OpenRouter slug inspected | Live endpoints | Conclusion |
|---|---|---:|---|
| Doubao-1.5-vision | no matching current catalog slug; `bytedance/doubao-1.5-vision-pro` endpoint lookup returned 404 | 0 | Not recoverable through OpenRouter |
| Qwen-VL-Plus | `qwen/qwen-vl-plus` | 0 | Historical page exists, but no runnable endpoint |
| Claude-3.5-Haiku | `anthropic/claude-3.5-haiku` | 0 | Historical page exists, but no runnable endpoint |
| Grok-4-Fast | `x-ai/grok-4-fast` | 0 | Historical page exists, but no runnable endpoint |
| LLaMA3.2-90B Vision | `meta-llama/llama-3.2-90b-vision-instruct` | 0 | OpenRouter does not remove the exact-weight access block |
| Claude-4.5 | `anthropic/claude-sonnet-4.5` | non-empty | Runnable only if the ambiguous manuscript row is confirmed to mean Sonnet 4.5 |
| LLaMA3.2-11B Vision | `meta-llama/llama-3.2-11b-vision-instruct` | 0 | OpenRouter does not remove the exact-weight access block |

The runnable Claude mapping resolves to provider model
`anthropic/claude-4.5-sonnet-20250929`, accepts image input, and supports structured outputs and
temperature. Its OpenRouter list price at audit time is USD 3/M input tokens and USD 15/M output
tokens. The manuscript label must be corrected to `Claude Sonnet 4.5` if this identity is approved;
it must not remain the ambiguous `Claude-4.5`.

OpenRouter currently offers possible *contemporary comparison* rows, but they are not historical
reproductions:

- `x-ai/grok-4.20` or `x-ai/grok-4.5` instead of Grok-4-Fast;
- `meta-llama/llama-4-maverick` instead of Llama 3.2 Vision;
- `qwen/qwen3-vl-235b-a22b-instruct` instead of Qwen-VL-Plus;
- ByteDance Seed models instead of Doubao 1.5 Vision.

Any such run must be added as a separately labelled 2026 contemporary comparison. It cannot fill,
rename or silently replace the mandatory historical row. `openrouter/auto` is forbidden for a
scientific baseline because it does not bind a stable model identity.

## Stage3 judge suitability

### Recommended primary, historically aligned judge

`openai/gpt-4o-2024-08-06` is currently live through both OpenAI and Azure endpoints and is the
best match to the manuscript's stated GPT-4o process reward. It supports:

- image, text and file input;
- strict structured outputs / JSON schema;
- `temperature`, `seed`, maximum-token controls and stop sequences;
- a dated model identity rather than a moving `gpt-4o` alias.

Audit-time list pricing is USD 2.50/M input tokens and USD 10/M output tokens. For formal use, bind
both the model and provider, set `allow_fallbacks=false` and `require_parameters=true`, and cache
the complete request, response, provider, usage, request identifier, timing, retry and parser
record. Do not use an automatic model fallback. If zero data retention is required, the live ZDR
catalog includes the Azure endpoint for this exact model; pin that endpoint and preserve the
provider distinction.

The 2024-08-06 snapshot is preferred over the moving `openai/gpt-4o` alias and over silently
switching the process-reward method to a newer model. The exact historical manuscript GPT-4o
snapshot remains unproven, so the revision must describe this as a frozen reconstruction unless
new provenance is found.

### Independent evaluation and circularity control

`anthropic/claude-sonnet-4.5` is a suitable independent *secondary* evaluator because it accepts
images and strict structured outputs. `google/gemini-2.5-pro` is another viable independent
secondary judge. Neither substitutes for the requested pathologist comparison, and neither should
replace GPT-4o in the primary Stage3 reward without redefining the method and ablations.

Suggested separation:

1. GPT-4o 2024-08-06: primary Stage3 process-reward reconstruction.
2. Claude Sonnet 4.5 or Gemini 2.5 Pro: independent secondary re-score on a frozen subset.
3. Pathologist ratings: required clinical-validity comparison on a manageable frozen subset.
4. Final answer accuracy: rule-based, never judge-model based.

Cheap models such as GPT-4o-mini may be used only for synthetic schema/plumbing tests. They must
not tune the scientific rubric or stand in for the formal judge.

## Scale and preliminary cost envelope

The inherited Stage2-shaped Stage3 run has 1,000 RL records, four generations, three epochs and
24,000 online reward events. If every event triggers one judge request, one formal Stage3 arm means
24,000 paid calls. With 500--1,500 billable input tokens and 30--80 output tokens per event, GPT-4o
2024-08-06 has an approximate inference envelope of USD 37.20--109.20 per arm before image-token
variation, retries and OpenRouter's pay-as-you-go credit-purchase fee. The required 0.3/0.4/0.5
sensitivity would approximately triple that envelope if each value is a separately trained arm.

This is only a planning range. A non-test, non-training smoke must record actual image and text
usage before a formal maximum budget is frozen. Raw judge events can be reused for deterministic
offline aggregation checks, but cannot generally be reused across independently trained
sensitivity arms because their evolving policies generate different completions.

## Required decisions before any call

1. Confirm whether the manuscript `Claude-4.5` row means Claude Sonnet 4.5.
2. Decide whether to run only exactly recoverable historical rows, or additionally include a
   separately labelled contemporary replacement table.
3. Approve GPT-4o 2024-08-06 as the primary Stage3 reconstruction judge.
4. Select Claude Sonnet 4.5 or Gemini 2.5 Pro for the independent secondary subset.
5. Freeze the Stage3 JSON event schema, aggregation, timeout/retry/fallback policy and the exact
   meaning of the 0.3/0.4/0.5 sensitivity.
6. Provide an OpenRouter credential outside Git and approve separate maximum budgets for baseline
   evaluation, Stage3 judge calls and independent re-scoring.

Stage3 remains unstarted and paid APIs remain uncalled.
