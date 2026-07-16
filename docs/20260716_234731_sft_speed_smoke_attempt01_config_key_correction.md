# SFT speed smoke Attempt01: configuration-key correction

Timestamp: `2026-07-16T23:47:31+08:00`

## Scope

This is an engineering throughput smoke only. It is not a formal scientific result and did not
access validation or test data.

## Attempt01 retained result

Run directory:

`/home/wjy/pathvlm_r1_v1_formal/runs/stage1_sft_speed_smoke/sft_speed_smoke_z2_gpu_nogc_fused_20260716_233956`

The 20 optimizer steps completed with eight GPUs, ZeRO-2, GPU-resident fused AdamW, global batch
size 8, and the required language-trainable/vision-projector-frozen policy. Measured throughput was
`0.086 step/s` (`11.6279 s/step`) and peak device memory was `40052 MiB` on GPU 0 and approximately
`39167-39267 MiB` on the other GPUs.

## Gate failure and cause

The resolved YAML requested `gradient_checkpointing: false`, but the training log explicitly
reported `Gradient checkpointing enabled.` LLaMA-Factory v0.9.2 controls this model-level behavior
with the negative model argument `disable_gradient_checkpointing`; the generic Transformers field
alone did not disable it.

Attempt01 is therefore relabeled as a completed measurement with a failed configuration-intent
gate. Its speed and memory measurements remain useful engineering evidence, but it is not evidence
for a no-gradient-checkpointing configuration. The original run and log are retained.

## Correction

The speed-smoke launcher now:

- writes `disable_gradient_checkpointing: true` for the intended no-GC variant;
- retains the generic `gradient_checkpointing` field for consistency;
- derives the observed state from the LLaMA-Factory log;
- fails the run gate if the observed state differs from the variant intent; and
- exposes a correctly named `z2_gpu_gc_fused` fallback variant.

## Next authorized action

Run the corrected `z2_gpu_nogc_fused` 20-step smoke first. If it fails from CUDA OOM, retain that
failure and run the correctly labeled `z2_gpu_gc_fused` fallback. Do not launch a formal long run
until the resulting backend is frozen and reviewed.

