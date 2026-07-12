# Stage 3 process reward decisions pending

Timestamp: 2026-07-12 14:45:17 Asia/Shanghai

## Current status

The historical Stage 3 Process Reward artifacts appear incomplete and cannot currently be treated as recoverable ground truth. No reconstructed process reward has been silently introduced. Outcome-reward and validation plumbing are working independently.

## Decisions to agree with the user

1. Process unit: sentence, reasoning step, clause, or model token span.
2. Supervision source: recoverable historical annotations, deterministic rules, human re-annotation, teacher model, or a clearly documented combination.
3. Label space: binary correct/incorrect, ordinal quality, continuous score, or per-error-type labels.
4. Credit assignment: reward every valid step, only first error, prefix correctness, terminal aggregation, or another rule.
5. Process format: numbered steps inside `<think>`, free-form reasoning segmented afterward, or structured JSON.
6. Relationship to outcome reward: additive weights, gated process reward, separate ablation, or curriculum stages.
7. Online implementation: compute process reward directly during generation, call a learned reward model, or score generated traces with deterministic checks.
8. Reconstruction scope: which samples can be legitimately rebuilt and which must be marked unavailable.
9. Leakage control: ensure reconstructed supervision uses SFT/RL material only and never validation/test answers.
10. Formal claims: use “reconstructed/completed” language rather than “recovered” unless provenance proves recovery.

## Proposed engineering contract

Regardless of the chosen definition, the implementation should expose one versioned function:

```text
score_process(problem, image, completion, solution_metadata) -> {
  total_reward: float,
  step_rewards: list[float],
  step_spans: list,
  diagnostics: dict,
  reward_version: str
}
```

The online trainer and offline audit must call the same scoring implementation. Every event should be saved as JSONL so process-reward behavior can be audited independently of training.

## Work that can proceed before the decision

- Directory/config schema.
- Interface and serialization tests with synthetic fixtures.
- Leakage guards and sample-count manifests.
- Migration templates.

Actual labels, weights, and scientific claims must wait for user agreement.

