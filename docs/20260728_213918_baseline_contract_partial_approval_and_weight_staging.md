# Baseline contract partial approval and weight-staging completion

Created: 2026-07-28 21:39:18 +08:00

This is a preparation-only decision and asset record. No model inference, validation/test
evaluation, training, Stage 3 run or paid API request was performed. `formal_result` is false.

## User decisions

The user approved the following items from
`20260728_014950_baseline_evaluation_contract_decisions_pending.md`:

1. Use the frozen 385-record PathMMU validation split only for adapter/prompt/parser validation,
   followed by a one-time full 999-record PathMMU v2 test after the complete contract is frozen.
2. Use the common PathMMU output contract: concise image-grounded reasoning followed by exactly
   `<answer>X</answer>`; local inference will be BF16, unquantized, deterministic greedy decoding
   with `max_new_tokens=1024`.
5. Treat PLIP and CONCH as image-text matching baselines and UNI as a separate frozen visual
   representation baseline, rather than mislabelling any of them as a generative VQA model.

Items 3 and 4 remain under discussion:

- primary/secondary metrics, confidence intervals, paired testing, multiplicity and failure scoring;
- the exact model universe and scoring contract for PathVQA and the four OmniMedVQA sources.

Item 6, all hosted-provider calls and model mappings, is deferred. No hosted smoke, credential
check, cost-bearing request or provider call is authorized.

The user will separately resolve:

- Meta Llama 3.2 Vision 11B/90B Hugging Face weight approval;
- transfer of the exact seed-42 n3000 epoch-3/step-1125 checkpoint and the definition of the
  additional 1,000 SFT examples.

## Fixed-revision local weight staging

All non-Llama local generative weights in the active queues have completed. Exact total bytes,
config presence and every model-index shard reference were rechecked:

| Model | Exact weight bytes | Index references | Gate |
|---|---:|---:|---|
| Qwen2.5-VL-7B-Instruct | 16,584,414,560 | 5 | pass |
| Qwen2.5-VL-3B-Instruct | 7,509,337,976 | 2 | pass |
| MedVLM-R1 | 4,418,050,848 | 0, single shard | pass |
| MedGemma-4B-IT | 8,600,277,880 | 2 | pass |
| Lingshu-7B | 16,584,414,544 | 4 | pass |
| DeepSeek-VL2 | 54,961,192,528 | 8 | pass |
| InternVL3-8B | 15,888,831,920 | 4 | pass |
| HuatuoGPT-Vision-7B | 17,583,367,129 | 4 plus its separate visual weight | pass |
| LLaVA-Med-v1.5-Mistral-7B | 15,132,534,160 | 4 | pass |

CONCH, UNI and PLIP weights were already complete under their separately audited pathology
environment. The two Meta Llama repositories are the only remaining local manuscript-weight access
gate. Weight presence is preparation evidence only and does not imply successful model loading,
adapter validation or evaluation.
