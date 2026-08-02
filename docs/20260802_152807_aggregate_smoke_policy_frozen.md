# Aggregate local-baseline smoke policy frozen

Timestamp: `2026-08-02 15:28:07 CST`

## Outcome

The local GPU-baseline smoke gate now checks aggregate adapter behavior only. It does not correct,
rewrite or re-Judge individual answers, and answer accuracy is explicitly excluded from the gate.
Wrong but nonempty and contract-parseable answers remain immutable and pass this engineering check.

The default aggregate thresholds are at least 80% nonempty outputs, at least 80% contract-parseable
outputs and at most 20% generation-cap hits. PathVQA smoke inputs must include both free-form and
yes/no examples. Identity, source/model hashes, prediction hashes, row counts, ordered indices and
per-row source hashes remain fail-closed artifact checks. These thresholds are an engineering
adapter gate, not a performance statistic or a post-hoc test-selection rule.

## Verification

- Five unit tests passed, including an intentionally wrong but parseable PathMMU answer.
- The verifier passed the existing 16-case SFT4000 PathMMU smoke with 100% nonempty/parseable and
  0% cap hits.
- It passed the existing 16-case MedVLM-R1 PathVQA smoke with 100% nonempty, 87.5% parseable and
  0% cap hits; its low answer accuracy was not used by the gate.
- It passed the corresponding OmniMedVQA smoke with 100% nonempty/parseable and 6.25% cap hits.
- No prediction, score, paid request, model inference or test artifact was changed.

The verifier will be applied to the fixed local Llama and DeepSeek-VL2 smokes after GPT-4o Stage3
releases the GPUs. A failed aggregate gate stops the full run for adapter repair, but never
authorizes per-case correction.
