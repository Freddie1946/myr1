# Training protocol outline

## Stage 1 SFT

- Framework: LLaMA-Factory.
- Base: pinned Qwen2.5-VL-7B-Instruct revision from `base_model_manifest.json`.
- Data: CoT SFT files from `pathmmu_image_disjoint_v1`.
- Validation: frozen 385-QA validation split.
- Formal seeds: 42, 43, 44 for the full-data model.
- Scaling subsets: 500, 1000, 2000, 3000 QA; seed 42 for initial scaling runs.

The old 7B full-fine-tuning configuration OOMed on 4x4090 at AdamW state allocation. Before selecting the final method, run controlled smoke tests for:

1. Full language-model fine-tuning with ZeRO-3 CPU optimizer offload.
2. LoRA SFT with the same frozen vision tower/projector and data protocol.

The formal paper method must use one consistently documented choice. Do not mix full and LoRA runs in the same scaling curve.

## Stage 2 Outcome GRPO

- Parent: validation-selected Stage 1 checkpoint from the same formal seed lineage.
- Rewards: accuracy plus format.
- Scaling subsets: 250, 500, 1000 QA.

## Stage 3 Process GRPO

- Parent: validation-selected Stage 2 checkpoint.
- Rewards: accuracy, format, and online GPT process reward.
- Every judge request and response must be cached and auditable.

## Evaluation

- Development decisions use validation only.
- Test is executed after the protocol and checkpoints are frozen.
- OOD evaluation is stored separately from in-domain test results.

