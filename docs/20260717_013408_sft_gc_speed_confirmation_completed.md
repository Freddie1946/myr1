# Correctly labelled SFT GC speed confirmation completed

Timestamp: `2026-07-17T01:34:08+08:00`

## Outcome

The correctly labelled `z2_gpu_gc_fused` 20-step engineering speed smoke completed with
`status: completed` and `formal_result: false`. All recorded gates passed, and test was not
accessed.

Run directory:

`/home/wjy/pathvlm_r1_v1_formal/runs/stage1_sft_speed_smoke/sft_speed_smoke_z2_gpu_gc_fused_20260717_012431`

## Configuration and observed execution

- Eight physical RTX 4090 GPUs, one sample per device, no gradient accumulation.
- DeepSpeed ZeRO stage 2 with optimizer offload disabled.
- Full language-model fine-tuning with the vision tower and multimodal projector frozen.
- Gradient checkpointing was both requested and observed in the LLaMA-Factory log.
- Optimizer was `adamw_torch_fused`.
- Peak selected-GPU memory was 41,226 MiB.

The run completed all 20 requested steps in 371.8437 seconds. Recorded throughput was 0.054
steps/second, or 18.5185 seconds/step, with final reported training loss 0.7138. This is slower
than the earlier actual-GC but incorrectly labelled Attempt01 measurement of 11.6279 seconds/step.
The current result is the conservative planning measurement; no claim is made that the difference
is caused by a specific component because the host and storage/cache state differed.

## Formal-run projection

The n=3000 run has 375 optimizer steps per epoch and 3,750 steps over ten epochs. At the current
confirmation rate, training steps alone project to approximately 19.3 hours. Allowing for ten full
checkpoint saves, scientific-snapshot hashing, final save/reload, and tensor comparison gives a
conservative end-to-end estimate of approximately 21--24 hours.

After the speed smoke's final 16-GiB model save, free space was 1,191,458,734,080 bytes. The formal
runner requires 1,157,493,686,272 bytes at startup, leaving 33,965,047,808 bytes (approximately
31.6 GiB) above the start gate. The budget includes a 550-GiB hard reserve, three transient full
resume checkpoints, ten epoch snapshots, and the final model. The run must fail closed if the
550-GiB reserve cannot be maintained.

An out-of-sandbox read-only `nvidia-smi` check after completion showed GPUs 0--7 at the 18-MiB idle
baseline with no compute processes.

## Next action

After committing this record, perform the runner's fresh atomic GPU, disk, model, data, preflight,
configuration, and port gates and launch only the fresh n=3000, seed-42 formal SFT duration sweep.
Do not resume the old scale supervisor. Stage 2 and Stage 3 remain outside the current launch.
