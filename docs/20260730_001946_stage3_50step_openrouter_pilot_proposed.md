# Stage3 50-step OpenRouter pilot proposed

Timestamp: `2026-07-30T00:19:46+08:00`

Status: proposal awaiting the user's explicit cost and scientific-contract confirmation.
`formal_result: false`. No Stage3 training or paid API request has started.

## Purpose

The user requested one initial Stage3 training run to inspect whether process reward produces a
useful signal. The first execution should be a bounded engineering/scientific pilot, not a formal
paper result and not the full 0.3/0.4/0.5 sensitivity.

## Proposed fixed pilot

- parent:
  `transferred_checkpoints/outcome_grpo_n1000_seed42_epoch02_step1000`
- data: the frozen image-disjoint RL1000 only; no validation/test answer enters the judge
- seed: 42
- optimizer steps: 50
- global batch size: 8
- generations per prompt: 4
- expected online reward events / maximum judge calls before cache hits: 800
- maximum completion length: 192
- learning rate: `1e-6`
- beta: `0.04`
- trainable/frozen policy: full language model trainable; vision tower and multimodal projector
  frozen
- process penalty: 0.4
- judge: `openai/gpt-4o-2024-08-06`
- provider: Azure OpenRouter endpoint with ZDR required
- model and provider fallback: disabled
- proposed maximum OpenRouter spend: USD 15
- post-pilot evaluation: frozen 385-record validation only
- test999 and external test sets: not accessed for pilot selection

## Proposed judge events and deterministic scoring

The judge must emit strict JSON events and short evidence spans, not a final numerical reward:

1. `image_feature_analysis_present`
2. `option_elimination_present`
3. `medical_knowledge_support_present`
4. `histological_definition_error`
5. `logical_contradiction`
6. `outdated_or_incorrect_pathology_criterion`

No fixed number of eliminated options is required. Local code computes:

```text
integrity = max(0, 1 - 0.4 * missing_integrity_components)
knowledge = max(0, 1 - 0.4 * present_knowledge_error_types)
process_reward = (integrity + knowledge) / 2
```

The GRPO reward list remains accuracy, format and process. Every request/response, exact model and
provider, image/record/completion hash, usage, cost, latency, retry, parsed events and locally
computed score must be cached as append-only JSONL.

## Failure and cost policy

- JSON schema and exact-cache tests run without paid calls.
- A tiny non-training API smoke must pass before GPU training.
- Retry only declared transport/rate-limit failures with bounded backoff.
- A request that still fails after the frozen retry count aborts the run. It does not receive a
  process score derived from final-answer correctness.
- Stop before another call if the locally recorded spend plus a conservative next-request reserve
  could exceed USD 15.
- Preserve partial events and checkpoints on any stop; no automatic parameter or prompt tuning.

## Gates awaiting user confirmation

1. OpenRouter key stored outside Git in
   `/home/dataset-assist-0/czy/wjy/.secrets/openrouter.env`.
2. Explicit approval of the USD 15 maximum spend.
3. Explicit approval of the six-event schema, deterministic 0.4 aggregation and abort-on-failure
   policy.
