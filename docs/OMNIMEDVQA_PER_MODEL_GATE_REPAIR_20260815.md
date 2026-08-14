# OmniMedVQA per-model generation repair (2026-08-15)

## Decision

The shared compatible/native baseline queues were stopped. Their partial files
remain audit artifacts and are not formal results. Formal baseline evaluation
now uses one independent process, state file, log, smoke gate, and full output
directory per model. A failing backend cannot start its own full run and cannot
block or mutate another model's result.

## Frozen data and scoring contract

- Full set: `omnimedvqa_four_sources_8518.json`, SHA256
  `04ba0790410cd5de4c95689d349574b102034a153afa00d952e351d3c13991a7`.
- Smoke set: 64 records, deterministically selected by canonical record hash,
  with 16 records from each of Chest CT Scan, Diabetic Retinopathy, ISIC2020,
  and Retinal OCT-C8. SHA256
  `d30ae81529a84cbeaaa02244e296e7132367d011429e5fb016854669ed265fe7`.
- Default prompt/output contract:
  `omnimed_domain_think_answer_v4_1024`.
- Llama-3.2-Vision, DeepSeek-VL2, and LLaVA-Med use a recorded
  `native_choice_only` prompt adapter after the uniform reasoning prompt caused
  deterministic loops or systematic omission of the final answer. The image,
  question, labeled options, option order, greedy decoding, and score semantics
  remain unchanged. These prompt-style results must be identified explicitly
  in tables and must not be described as an identical-prompt comparison.

## Smoke gate

Accuracy is never used as a generation gate. Each model must instead satisfy:

- exact data/model/prediction hashes and 64 contiguous rows;
- exactly 16 rows from each source subset;
- no invalid JSON;
- no Unicode replacement/control-format corruption;
- no adjacent repeated-token/phrase loop;
- at most one 1024-token cap hit;
- strict, target-blind final-choice coverage of at least 95% under the default
  prompt.

For the documented LLaVA-Med native adapter, the threshold is 75%, and at most
one immediate-EOS empty completion is permitted. Such rows are retained and
counted wrong, never imputed or manually corrected. Both strict-final and the
official OmniMedVQA option-matching metrics must be reported.

## Model-specific findings

- Seven initial models passed the four-source gate after target-blind parser
  fixes: Qwen2.5-VL-3B, Lingshu-7B, MedVLM-R1, MedGemma-4B-IT,
  ScaleReasoner-R1, HuatuoGPT-Vision-7B, and InternVL3-8B.
- Mllama uses `image_token_index == text_vocab_size`; the stock Transformers
  repetition-penalty processor therefore indexes one past the language logits.
  The replacement processor penalizes generated tokens only and never visual
  prompt IDs.
- DeepSeek-VL2 and Llama-11B produced deterministic long-form token loops under
  the reasoning contract. Their failed outputs are preserved; the native
  single-choice adapter passed for DeepSeek and passed after target-blind
  explicit-answer parsing for Llama-11B.
- LLaVA-Med did not reliably follow the reasoning/XML wrapper. Its no-example
  native single-choice adapter is evaluated separately and retains unresolved
  or empty generations as errors.

## Artifacts and execution

- Independent launcher: `scripts/run_single_omnimed_baseline_gated.py`
- Smoke freezer: `scripts/freeze_omnimed_stratified_smoke_panel.py`
- Gate: `scripts/verify_omnimed_stratified_smoke.py`
- Target-blind additive rescorer: `scripts/rescore_external_vqa_predictions.py`
- Formal output root:
  `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/omnimedvqa_per_model_gated_v2_20260815`

As of 2026-08-15 03:41 CST, eight independent full evaluations were active,
Lingshu and MedVLM-R1 had independent GPU waiters, and Llama-90B had an
independent three-GPU smoke waiter. The migration P1 model upload queue remained
active and independent of GPU evaluation.
