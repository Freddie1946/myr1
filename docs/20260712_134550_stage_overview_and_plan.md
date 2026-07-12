# PathVLM-R1 revision execution overview

Timestamp: 2026-07-12 13:45:50 Asia/Shanghai

This document is the durable recovery entry for the revision experiments. All formal results must use the frozen data split and pinned base model recorded below. Smoke runs on the 4x RTX 4090 host are engineering checks only and are not paper results.

## Fixed protocol

- Data version: `pathmmu_image_disjoint_v1`.
- Split by image with zero cross-split image overlap: SFT 3000 QA, RL 1000 QA, validation 385 QA, test 1000 QA.
- Nested SFT sizes: 500, 1000, 2000, 3000. Nested RL sizes: 250, 500, 1000.
- `picked.json` is deprecated and excluded.
- Base: `Qwen/Qwen2.5-VL-7B-Instruct`, revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- The test set is frozen and must not be used for model selection, seed selection, threshold tuning, or prompt tuning.
- Validation is used for model/configuration selection. Test is evaluated only after the protocol is fixed.
- Main model target: three independent training seeds when formal compute permits. Inference should normally be deterministic per checkpoint; report any sampling and inference seeds if sampling is used.

## Stage 0: data and protocol freeze

Completed:

- Audited the 5385 usable PathMMU QA records: 3301 PubMed and 2084 EduContent, covering 3809 unique images.
- Created and verified the image-disjoint split and nested sample-size subsets.
- Generated LLaMA-Factory ShareGPT multimodal adapters and manifests.
- Pinned the exact base-model snapshot and audited historical SFT attempts.

Remaining:

- Do not change the split without creating a new explicit data version.
- Preserve exact QA/image counts and answer/source distributions in formal reports.

## Stage 1: SFT

Goal: full-parameter language-model fine-tuning while freezing the vision tower and multimodal projector.

Debug-host plan:

1. Complete a one-step 7B smoke run without touching unrelated GPU processes.
2. Verify trainable/frozen parameter counts, forward/backward, optimizer step, checkpoint save, and resumability.
3. Record peak GPU memory, RAM, swap, wall time, exact environment, and command.
4. Prepare portable formal-training configs; do not treat debug-host outputs as experimental evidence.

Formal-machine plan:

1. Run the agreed SFT size ablation using nested subsets.
2. Select training duration/configuration on validation only.
3. Train the main configuration with seeds 42, 43, and 44 unless later documented otherwise.
4. Save per-run sample counts, seeds, code revision, base revision, metrics, and checkpoints.

## Stage 2: outcome-reward GRPO

Plan:

1. Start from the newly trained SFT checkpoint, not the obsolete historical 7B attempts.
2. Use only the frozen RL split for policy optimization.
3. Conduct the agreed RL data-size ablation with nested 250/500/1000 subsets.
4. Select configurations using validation; keep test sealed.
5. Run the final main setting with the agreed independent training seeds and record online reward details.

## Stage 3: process-reward completion

Status: the original complete artifacts are likely unrecoverable; only a reconstruction/completion is planned.

Plan:

1. Agree on the recoverable supervision format and exact reward definition with the user before implementation.
2. Clearly label reconstructed artifacts and distinguish them from historical runs.
3. Keep data provenance and leakage checks explicit.
4. Do not claim an exact reproduction if original artifacts cannot be recovered.

## Stage 4: evaluation and reviewer-response experiments

Plan:

1. Evaluate validation during development and test only after locking the configuration.
2. Report data-scale ablations, generalization results, and seed variation.
3. Separate training-time online reward from post-hoc offline scoring.
4. Preserve raw predictions and scoring outputs, with sample counts and model/checkpoint identifiers.

## Current blocking constraint

The debug host has four RTX 4090 GPUs, but GPU 3 is occupied by an unrelated VLLM process. As of this timestamp, another user's process also occupies GPU memory on GPUs 0-2. No unrelated process may be terminated or disturbed.

