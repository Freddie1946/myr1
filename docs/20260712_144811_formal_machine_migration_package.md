# Formal-machine migration package

Timestamp: 2026-07-12 14:48:11 Asia/Shanghai

## What has been validated on the debug host

- Frozen image-disjoint data contracts and sample counts.
- LLaMA-Factory multimodal adapter loading.
- Full-parameter language-model SFT with frozen vision tower/projector.
- ZeRO-3 optimizer-only CPU offload with Torch AdamW.
- A gathered checkpoint that reloads in a different environment.
- Online image-conditioned generation, outcome/format reward calls, reference-model/KL plumbing, GRPO backward/optimizer framework, and save.
- Deterministic validation inference, raw output persistence, and offline scoring with the identical reward module.

Stage 3 process reward is not scientifically finalized and is excluded from “validated” claims.

## Formal model and data

- Base: `Qwen/Qwen2.5-VL-7B-Instruct`.
- Revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- SFT sizes: nested 500/1000/2000/3000.
- RL sizes: nested 250/500/1000.
- Main SFT/RL configuration seeds: 42, 43, 44 unless a later timestamped protocol explicitly replaces them.
- Seeds refer to independent training runs. Deterministic inference does not require three seeds per checkpoint.
- Validation may guide configuration selection. Test remains sealed until the final protocol is locked.

## Files to migrate

1. Entire `pathvlm_r1_v1` experiment root, excluding debug checkpoint shards if bandwidth/storage is limited.
2. Entire `pathmmu_image_disjoint_v1` data version, including manifests and nested subsets.
3. All referenced PathMMU images with paths rewritten or mounted consistently.
4. LLaMA-Factory source at the audited commit and the VLM-R1/Open-R1 source files matching `code_hash_manifest_20260712_144811.json`.
5. Exact base-model snapshot or a verified download of the pinned revision.

Do not use the 3B debug checkpoints as parents of formal 7B runs.

## Environment split

SFT overlay:

- Torch 2.6.0 with a CUDA build compatible with the formal host.
- Transformers 4.49.0.
- Tokenizers 0.21.0.
- DeepSpeed 0.15.4.
- LLaMA-Factory 0.9.2 at audited source revision.

GRPO overlay:

- Same Torch/Transformers/Tokenizers/DeepSpeed base.
- TRL 0.15.2.
- Open-R1 trainer source verified by SHA-256.

Keep SFT and GRPO overlays separate because LLaMA-Factory's historical TRL constraint and the checked-in GRPO trainer require different TRL generations.

## Formal SFT execution

Use `configs/sft/formal_7b_full_n3000_seed0042.yaml` as a reviewed reference, not a blind final command. Replace `/path/to` roots, select the DeepSpeed policy for the actual GPU/RAM topology, and create uniquely named configs/manifests for every sample size and seed.

Before full runs:

1. Verify exact base revision and hashes.
2. Validate adapter counts and all image paths.
3. Run one optimizer step using the exact formal environment/hardware.
4. Verify trainable/frozen parameter counts.
5. Save, reload, and resume a checkpoint.
6. Confirm disk requirements for model plus optimizer state and checkpoint retention.

Do not assume the debug-host `10 epochs, 2e-5` reference is optimal. Select duration/configuration using validation-only evidence and document the selection rule before opening test.

## Formal outcome GRPO execution

Use `scripts/launch_formal_outcome_grpo_template.sh` after setting all required environment variables. The template uses a 192-token completion cap because 64 tokens truncated both debug completions; confirm the final cap on validation/runtime constraints.

Preflight requirements:

- Parent is the newly trained formal SFT checkpoint.
- Dataset is the frozen RL subset, never validation/test.
- Global batch is divisible by `num_generations`.
- Reward events show nonzero variance on a diagnostic window; zero variance is legal but yields no learning signal.
- Monitor accuracy reward, format reward, total reward, reward std, KL, completion length, gradient norm, GPU/RAM/swap, and throughput.
- Save online reward JSONL and raw diagnostic completions.

## Required result registry fields

- Run ID, stage, formal/debug flag.
- Data version, split, subset count, adapter hash.
- Base revision and parent checkpoint hash/ID.
- Code hashes and environment package freeze.
- Train/data seed and inference sampling seed when applicable.
- GPU model/count, DeepSpeed config, effective batch.
- Trainable/frozen parameter counts.
- Start/end timestamps, exit status, resume history.
- Validation metrics used for selection.
- Final test metrics and raw predictions only after protocol lock.

## Stop conditions

- Any SFT/RL/validation/test image overlap.
- Test accessed during tuning.
- Wrong base revision or wrong parent checkpoint.
- Vision/projector unexpectedly trainable.
- Reward parser mismatch between online and offline paths.
- Persistent zero reward variance without an explicit diagnosis.
- Missing sample counts, seeds, raw predictions, or code/environment identifiers.
- Inadequate disk space for an atomic checkpoint save.

