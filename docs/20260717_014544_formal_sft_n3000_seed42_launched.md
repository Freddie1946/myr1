# Formal SFT n=3000 seed-42 duration sweep launched

Timestamp: `2026-07-17T01:45:44+08:00`

## Launch

The fresh formal n=3000, seed-42, ten-epoch SFT duration sweep launched through the audited runner:

`/home/wjy/pathvlm_r1_v1_formal/runs/stage1_sft/n3000_seed0042/formal_sft_n3000_seed0042_20260717_014544`

The runner repeated and passed its startup disk, physical-GPU occupancy, model identity, adapter
count/path, frozen preflight, configuration, DeepSpeed-backend, and master-port gates. Free space at
run creation was 1,821,179,666,432 bytes. All eight physical RTX 4090 GPUs were at the 18-MiB idle
baseline with no compute processes before launch. Test was not accessed.

## Observed initialization and first updates

- Exact base revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- 3,000 examples, global batch 8, 375 steps/epoch, 3,750 total optimizer steps.
- Full fine-tuning with 7,615,616,512 trainable of 8,292,166,656 total parameters.
- Vision tower and multimodal projector freeze messages were observed.
- Gradient checkpointing, bfloat16, ZeRO stage 2, fused AdamW, and no CPU offload were observed.
- The first four losses were finite (`2.0176`, `1.7170`, `2.0562`, `2.0066`).
- The first four gradient norms were finite and nonzero (`31.2638`, `28.6754`, `40.1124`,
  `36.9121`).
- Initial steady step time approached approximately 11.8 seconds/step; all eight GPUs reached
  approximately 39.2 GiB used and 100% utilization during sampled training intervals.

At this initial rate, optimizer steps project to approximately 12.3 hours. Including ten epoch
checkpoint saves, scientific-snapshot hashing, final reload, and tensor-delta gates gives a current
completion window of approximately 14:30--16:30 Asia/Shanghai on 2026-07-17, with 18:00 retained as
a conservative shared-storage-load bound.

## Post-run boundary

No later SFT run or Stage 2 run is chained blindly. On successful completion, first require every
formal gate and validate all ten n=3000 epoch snapshots on validation only while retaining raw
generations. For a fixed ten-epoch sample-scale comparison, the valid existing n=500 and n=1000
final models remain usable, the failed/pruned n=2000 point must be rerun, and this run supplies the
n=3000 point. If validation selects an earlier common epoch for the scale experiment, n=500,
n=1000, and n=2000 must be retrained under the new per-epoch snapshot policy because their old runs
do not contain the complete duration curve. Test remains evaluation-only.
