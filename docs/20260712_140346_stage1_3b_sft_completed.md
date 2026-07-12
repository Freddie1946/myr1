# Stage 1 3B SFT completed

Timestamp: 2026-07-12 14:03:46 Asia/Shanghai

## Outcome

E1 completed successfully on physical GPUs 1 and 2 using the pinned complete 3B snapshot `319ccfdc6cd974fab8373cb598dfe77ad93dedd3`.

- Full language-model fine-tuning: yes.
- Vision tower frozen: yes.
- Multimodal projector frozen: yes.
- Total parameters: 3,754,622,976.
- Trainable parameters: 3,085,938,688 (82.1904%).
- Frozen parameters: 668,684,288.
- Data: 8 SFT samples from `pathmmu_image_disjoint_v1`; test not accessed.
- Effective batch size: 2 across two GPUs.
- Optimizer: Torch AdamW with ZeRO-3 optimizer-only CPU offload.
- Attention: SDPA.
- Steps: 1.
- Loss: 1.642995834350586.
- Gradient norm: 25.186206817626953.
- Optimizer-step training runtime: 36.6896 seconds.
- Total wall time including load and save: 133.43 seconds.
- Exit status: 0.

## Resource observation

- Observed GPU memory during checkpoint save was approximately 7.5 GiB on each training GPU; both GPUs returned to approximately 0.4 GiB baseline afterward.
- Host maximum resident set size reported by `/usr/bin/time -v`: 45,739,288 KiB for the launcher process tree measurement.
- Swap rose from roughly 23 GiB to 30 GiB and remained far from exhaustion.
- Output size: 7.1 GiB.
- Disk headroom after save: approximately 30 GiB.

## Checkpoint handoff

`/home/wjy/revision_runs/10_training/pathvlm_r1_v1/debug_e2e/qwen2_5_vl_3b/stage1_sft/n0008/seed0042/attempt02/output`

The directory contains a ZeRO-3-gathered two-shard model plus config, tokenizer, processor, and training metrics. It is the required parent checkpoint for the debug outcome-reward stage.

## Next action

Create an 8-sample deterministic GRPO debug adapter from the frozen RL split. Use online generation and in-training accuracy/format rewards, starting from this SFT output. Keep validation and test out of RL training.

