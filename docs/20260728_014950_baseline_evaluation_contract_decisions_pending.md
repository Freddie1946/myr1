# Baseline evaluation contract: decisions pending

Created: 2026-07-28 01:49:50 +08:00

Status: proposal only; `formal_result: false`. No model inference, provider call, validation
selection or test evaluation is authorized by this document.

## Recommended common PathMMU rerun

1. Use the frozen 385-record validation split only to validate adapters, freeze the common prompt,
   parser and generation cap, and diagnose failures.
2. After every local and hosted baseline has passed the same predeclared smoke, run the corrected
   999-record PathMMU v2 test once per model. Report this as a contemporary full-test rerun, not an
   exact reproduction of the manuscript's unrecovered historical 500 IDs.
3. Use the same image, question and option order for every model. Ask for concise image-grounded
   reasoning followed by exactly `<answer>X</answer>`.
4. Default local decoding proposal: BF16 weights, no quantization, greedy decoding,
   `do_sample=false`, `temperature=0` where the API accepts it, and `max_new_tokens=1024`.
   Preserve cap hits and do not silently retry them with a larger cap.
5. Use parser v2 for the primary multiple-choice accuracy. Preserve every raw response, parsed
   choice, invalid/ambiguous reason, latency, retry and provider response identifier.
6. Primary reporting: exact accuracy, correct/total and Wilson 95% CI. Pairwise model comparisons
   use paired McNemar tests on the same sample IDs with a predeclared multiplicity correction.
   Format-pass and invalid-response rates are secondary. Reasoning-quality scores are separate
   secondary endpoints and must not replace rule-based accuracy.
7. PathVQA and OmniMedVQA OOD results remain separate from PathMMU. Their model scope and
   open-ended scoring contract must be frozen separately; they must not be pooled into PathMMU
   accuracy.

## Local-model proposal

- Use only the exact Hugging Face revisions in
  `protocol/manuscript_baseline_metadata_manifest_20260725_013154.json`.
- Add LLaVA-Med v1.5 as an explicitly additional reviewer-response baseline at
  `microsoft/llava-med-v1.5-mistral-7b`,
  revision `91bb16c122001ddc9cf1fd36ce1dae09448943a2`.
- Use BF16/unquantized weights. The 90B Llama model may use tensor parallelism across the A100s,
  but it may not be silently replaced by an 8-bit/4-bit or third-party conversion.
- PLIP and CONCH are image-text matching baselines. UNI is a visual encoder. They must use a
  separately declared discriminative/representation protocol and must not be presented as
  generative VQA models.
- PathChat remains unavailable as a verified public end-to-end release and cannot be claimed as a
  completed baseline without a new authoritative source.

## Hosted-model identity issues

The names in the submitted table are insufficient to reproduce a 2026 provider call:

- Anthropic documents `claude-3-5-haiku-20241022` as retired on 2026-02-19 and recommends
  `claude-haiku-4-5-20251001`. A rerun with the replacement must be labelled contemporary rather
  than as the historical Claude 3.5 Haiku result.
- `Claude-4.5` does not specify Haiku, Sonnet or Opus. If the intended historical row was Sonnet,
  the proposed pinned Claude API ID is `claude-sonnet-4-5-20250929`; the user must confirm the
  family.
- xAI documents the Grok 4 Fast slugs as retired/redirected. The historical
  reasoning versus non-reasoning variant is unresolved; an explicit current replacement and
  reasoning mode must be selected before calls.
- Alibaba still documents the `qwen-vl-plus` mainline name, but its older dated snapshot has been
  retired. The proposed reproducible contemporary replacement is the dated
  `qwen3-vl-plus-2025-12-19`, unless the user instead requires the moving `qwen-vl-plus` mainline
  for label continuity.
- Volcengine documentation identifies the family as Doubao 1.5 Vision Pro, with examples using
  `Doubao-1.5-vision-pro`/`doubao-1.5-vision-pro-32k`; actual Ark calls can require a dated model ID
  or a user deployment endpoint ID. The exact available endpoint must be read from the user's Ark
  account before freezing the row.

Provider credentials must be injected only through a repository-external mode-600 environment
file or the provider credential store. Before any paid call, freeze a maximum budget and run a
small non-test smoke. For five hosted rows, a full 999-record rerun is 4,995 successful requests
before retries.

## User decisions required before evaluation

1. Approve the 385-validation then one-time 999-test protocol and the proposed common prompt/
   decoding/parser/statistics.
2. Confirm whether `Claude-4.5` means Sonnet 4.5, and approve or replace each contemporary hosted
   model mapping above.
3. Provide the API regions/endpoints and credentials outside Git, plus a maximum paid-call budget.
4. Decide whether every generative baseline runs on every OOD dataset, or whether OOD uses a
   predeclared subset of models.
5. Approve a distinct task/metric for PLIP, CONCH and UNI.
6. Separately define the 4000-SFT additional 1,000 records and transfer the exact seed-42
   n3000 epoch-3/step-1125 parent if continuation is retained.
