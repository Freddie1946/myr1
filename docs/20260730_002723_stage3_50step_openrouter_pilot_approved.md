# Stage3 50-step OpenRouter pilot approved

Timestamp: `2026-07-30T00:27:23+08:00`

Status: approved and frozen before execution. `formal_result: false`. No Stage3 training or paid
model request had started at this timestamp.

## User approvals

The user explicitly approved:

- a client-enforced hard ceiling of USD 15 for the complete pilot, including the API smoke;
- the six-event process-judging contract;
- penalty coefficient `0.4`;
- the previously proposed 50-optimizer-step pilot and validation-only post-check.

The external key gate had already passed with approximately USD 19.64 available credit. The key
has no provider-side dollar limit, so the local ledger is authoritative and must fail closed.

## Frozen execution contract

- parent:
  `/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/transferred_checkpoints/outcome_grpo_n1000_seed42_epoch02_step1000`
- training data: frozen image-disjoint RL1000 only
- seed: `42`
- optimizer steps: `50`
- global batch size: `8`
- generations per prompt: `4`
- maximum completions and Judge calls before cache hits: `400`
- maximum completion length: `192`
- learning rate: `1e-6`
- beta: `0.04`
- reward functions: accuracy, format, process
- maximum total reward audit events: `1,200`
- language model: fully trainable
- vision tower and multimodal projector: frozen
- Judge model: `openai/gpt-4o-2024-08-06`
- provider: OpenRouter Azure only, ZDR required
- model/provider fallback: disabled
- pilot-wide API spend ceiling, including smoke: USD `15`
- post-pilot selection data: frozen validation385 only
- PathMMU test999 and external test sets: not accessed for pilot selection

## Frozen six-event schema

The Judge returns strict booleans and short evidence spans for:

1. `image_feature_analysis_present`
2. `option_elimination_present`
3. `medical_knowledge_support_present`
4. `histological_definition_error`
5. `logical_contradiction`
6. `outdated_or_incorrect_pathology_criterion`

No fixed number of eliminated options is required. Local deterministic aggregation is:

```text
integrity = max(0, 1 - 0.4 * missing(first three events))
knowledge = max(0, 1 - 0.4 * present(last three events))
process_reward = (integrity + knowledge) / 2
```

The reference answer may be supplied only to assess medical correctness of the reasoning. It
must not cause the Judge to award process credit merely because the final answer is correct.

## Failure, audit and budget policy

- Run schema, cache, ledger and mocked-transport tests without paid calls.
- Run one non-training paid API smoke before GPU training.
- Every request and response is content-addressed and cached.
- Record model, provider, hashes, usage, cost, latency, retry, parsed events and local score.
- Reserve a conservative next-call amount under an inter-process file lock before each request.
- Refuse a call when committed spend plus unresolved reservations could exceed USD 15.
- A response with unknown cost or an ambiguous potentially billed failure is retained as an
  unresolved reservation and aborts the run.
- Retry only frozen transient transport/rate-limit failures with bounded backoff.
- Persistent request/schema failure aborts and preserves the run; it never substitutes an
  answer-derived process score.
- Do not tune the prompt, penalty or hyperparameters during this pilot.

## Correction to the proposal

The proposal recorded 800 maximum Judge calls. Inspection of the actual distributed trainer shows
that each optimizer step produces 8 completions globally, not 16. Therefore:

```text
50 steps * 8 completions = 400 maximum Judge calls
50 steps * 8 completions * 3 reward functions = 1,200 reward audit events
```

The earlier Stage2 count included both accuracy and format reward events and was mistakenly reused
as if it counted completions. This correction changes only the call/event accounting; it does not
change any approved training hyperparameter or the USD 15 ceiling.
