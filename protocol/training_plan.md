# Training protocol outline

## Stage 1 SFT

- Framework: LLaMA-Factory.
- Base: pinned Qwen2.5-VL-7B-Instruct revision from `base_model_manifest.json`.
- Data: CoT SFT files from `pathmmu_image_disjoint_v2`. These SFT files are byte-identical to v1;
  v2 only removes one exact-content duplicate QA from test.
- Validation: frozen 385-QA validation split.
- Formal seeds: 42, 43, 44 for the full-data model.
- Scaling subsets: 500, 1000, 2000, 3000 QA; seed 42 for initial scaling runs.

The old 7B full-fine-tuning configuration OOMed on 4x4090 at AdamW state allocation.
The revised formal protocol is no longer method-selective: it requires full-parameter
language-model fine-tuning with the vision tower and multimodal projector frozen.
Use ZeRO-3 and optimizer offload as required by the formal hardware. LoRA/QLoRA may
only be used in separately labelled engineering diagnostics and must not enter the
formal scaling curve or paper results.

Before any formal SFT run, complete the one-step save/reload/resume gate specified in
`AGENTS.md` and record trainable/frozen parameter counts and representative tensor deltas.

## Stage 2 Outcome GRPO

- Parent: validation-selected Stage 1 checkpoint from the same formal seed lineage.
- Rewards: accuracy plus format.
- Scaling subsets: 250, 500, 1000 QA.

## Stage 3 Process GRPO

- Status: reconstructed/completed work whose scientific definition is still pending user agreement.
- Parent, process unit, supervision source, label space, aggregation, and online implementation must be frozen in a new timestamped protocol before training.
- If an external judge is selected, every request, response, parser result, fallback, model version, decoding parameter, and cost record must be cached and auditable.
- No historical-recovery claim is allowed unless provenance proves it.

## Evaluation

- Development decisions use validation only.
- Test is the corrected 999-QA v2 split and is executed only after the protocol and checkpoints are
  frozen. It is never used for prompt, checkpoint, parser, seed, reward, or threshold selection.
- OOD evaluation is stored separately from in-domain test results.
