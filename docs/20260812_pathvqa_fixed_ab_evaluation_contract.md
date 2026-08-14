# PathVQA fixed-A/B evaluation contract (2026-08-12)

> **Superseded reporting priority (2026-08-14):** the A/B generation contract remains valid as a
> supplementary paired interface diagnostic, but original free-generation Yes/No is again the
> primary PathVQA binary contract. See
> `docs/20260814_pathvqa_yesno_primary_ab_diagnostic_contract.md`. Existing raw predictions and
> metrics remain valid under their recorded contract labels.

## Decision

Future PathVQA binary evaluations use a PathMMU-aligned free-generation interface as the primary
performance result:

```text
<question>
Options:
A) Yes
B) No
First output the thinking process in <think> </think> tags and then output the final answer in
<answer> </answer> tags. The answer tag must contain exactly one selected option in the form
A) Yes or B) No.
```

The mapping is always `A = Yes`, `B = No`. Option reversal is not used. This is a complete task
interface transformation, not a post-hoc replacement of the strings `Yes` and `No`.

The generation cap is 2,048 new tokens. Every run reports semantic accuracy, parseable rate,
strict PathMMU-format rate, generation-cap hits, and the pinned mapping. A self-contradictory answer
such as `A) No` or `B) Yes` is unresolved rather than silently scored from its letter.

## Metric hierarchy

1. **Primary PathVQA binary metric:** greedy free-generation accuracy under the fixed-A/B contract.
2. **Secondary interface audit:** original Yes/No free generation on a frozen diagnostic panel when
   a direct interface comparison is needed.
3. **Mechanism-only diagnostic:** forced Yes/No logits for normal/shuffled/blank-image visual
   dependence. It is not the primary semantic PathVQA score.

The A/B interface is more closely aligned with the PathMMU training/output contract and reduces one
known source of parser/interface mismatch. It is not assumed to improve every checkpoint: historical
pilot results showed checkpoint-dependent changes. Therefore earlier PathVQA values retain their
original contract labels and are not silently overwritten; cross-model tables must compare identical
contracts.

## Implementation

- Full PathVQA yes/no test runner contract: `pathvqa_pathmmu_ab_v6_2048` in
  `scripts/run_external_vqa_qwen.py`.
- Frozen PathVQA validation contract: `pathmmu_ab_v1_2048` in
  `scripts/run_pathvqa_yesno_prompt_calibration.py`.
- The standard full-language RL evaluator now records A/B free generation as primary while retaining
  forced-binary normal/shuffle/blank results under a secondary diagnostic namespace.
