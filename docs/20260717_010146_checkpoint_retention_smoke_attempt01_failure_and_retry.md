# Checkpoint-retention smoke Attempt01 failed; readiness retry corrected

Timestamp: `2026-07-17T01:01:46+08:00`

## Attempt disposition

Run directory:

`/home/wjy/pathvlm_r1_v1_formal/runs/stage1_sft_smoke/formal_7b_sft_smoke_n0008_seed0042_20260717_004648`

Attempt01 is an engineering smoke with `formal_result: false`. Its manifest is retained with
`status: failed`; no history was overwritten and test was not accessed.

## What passed

- Eight-GPU ZeRO-2 with GPU-resident fused AdamW and gradient checkpointing initialized correctly.
- Language parameters were trainable and vision/projector parameters were frozen.
- Step 1 completed and produced a full resumable checkpoint.
- Resume from step 1 loaded all eight ZeRO optimizer partitions and reached step 2.
- Resume from step 2 loaded all eight ZeRO optimizer partitions and reached step 3.
- Trainer rotation removed checkpoint 1 after checkpoint 2, then checkpoint 2 after checkpoint 3.
- All GPUs returned to the idle baseline.

## Failed gate and exact cause

The retention watcher treated the appearance of `trainer_state.json` as proof that every model and
tokenizer file was complete. LLaMA-Factory writes processor/tokenizer artifacts after the underlying
Trainer checkpoint routine. During that interval `vocab.json` existed with size zero. The watcher
correctly refused to archive an empty file but incorrectly treated this transient state as a terminal
failure, stopped retrying, and therefore missed the model-only snapshots before Trainer rotation.

The attempt ended fail-closed after step 3. It occupies approximately 133 GiB, consisting primarily
of the final gathered model and the last full resume checkpoint.

## Correction

The watcher now requires the full Qwen model/processor/tokenizer file set. Missing, empty, partially
decoded, or temporarily inconsistent files remain pending and are retried. Unexpected I/O failures
still stop the watcher. Every observed checkpoint step is recorded; if a checkpoint rotates without
a corresponding scientific snapshot, the terminal missing-snapshot gate fails.

A new regression test reproduces an empty `vocab.json`, verifies that no incomplete snapshot is
published, completes the file, and verifies successful retry. Sixteen unit/regression tests pass.

## Next action

After recording and explicitly approving removal of Attempt01's non-scientific model/optimizer
artifacts, rerun the three-step smoke. Do not start the GC confirmation or formal n=3000 SFT until the
retry passes all retention, reload, resume, rotation, backend, freeze, and test-isolation gates.

