# SFT speed smoke Attempt02 completed; checkpoint policy pending

Timestamp: `2026-07-16T23:59:50+08:00`

## Scope and result

This was an engineering-only 20-step throughput/memory smoke. It did not access validation or test
data and is not a formal scientific result.

Run directory:

`/home/wjy/pathvlm_r1_v1_formal/runs/stage1_sft_speed_smoke/sft_speed_smoke_z2_gpu_nogc_fused_20260716_235245`

All execution gates passed. The observed LLaMA-Factory log did not enable gradient checkpointing,
matching the corrected no-GC intent. Language parameters were trainable and the vision tower and
multimodal projector were frozen.

Measured results:

- 20/20 optimizer steps completed;
- `0.085 step/s` (`11.7647 s/step`);
- peak GPU memory `48098 MiB` on GPU 0, `46143 MiB` on GPU 7, and `41803-42523 MiB` on the other
  devices; and
- all eight devices returned to the idle baseline after exit.

The trainer completed and saved the final model, but NCCL process-group destruction logged a CUDA
OOM warning during shutdown. The process still exited successfully and the manifest gates passed.
This warning plus the `48098 MiB` peak means no-GC has insufficient safety margin for a formal long
run.

## Comparison and backend recommendation

Attempt01 actually ran with gradient checkpointing and measured `11.6279 s/step`, with a peak of
`40052 MiB`. Attempt02 correctly disabled it and measured `11.7647 s/step`, with a peak of
`48098 MiB`. Disabling checkpointing therefore provided no measured throughput benefit and removed
approximately 8 GiB of memory margin.

The current candidate for a correctly labeled confirmation smoke is therefore
`z2_gpu_gc_fused`: eight GPUs, global batch size 8, ZeRO-2, no CPU/NVMe offload, fused torch AdamW,
and gradient checkpointing enabled. No formal long run is authorized by this conclusion alone.

## Checkpoint-policy issue raised before formal relaunch

Over-SFT may reduce the usefulness of an SFT checkpoint as an Outcome-GRPO parent even when its SFT
loss continues to fall. The old `save_total_limit: 2` policy pruned earlier epoch checkpoints and is
not sufficient for duration/parent selection.

Full resumable ZeRO checkpoints are approximately 100 GiB each in the completed runs, while a
model-only 7B snapshot is approximately 16 GiB. Keeping ten full checkpoints for every scale is not
compatible with the available storage and would waste capacity needed by RL.

Before formal relaunch, implement and smoke-test a two-tier policy:

1. retain one or two newest full optimizer/scheduler checkpoints for interruption recovery;
2. retain a model-only snapshot at each epoch boundary for validation and later RL-parent analysis;
3. retain the final model and exact trainer state;
4. never prune a model snapshot until raw validation generations and the parent-selection decision
   are recorded; and
5. select with validation only, never test.

The exact retention implementation and the choice of which SFT scale receives the full ten-epoch
duration sweep remain pending user review. No new formal SFT run should start before this is frozen.

