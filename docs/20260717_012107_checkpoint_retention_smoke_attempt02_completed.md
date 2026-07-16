# Checkpoint-retention smoke Attempt02 completed

Timestamp: `2026-07-17T01:21:07+08:00`

## Outcome

Attempt02 completed all 18 engineering gates with `status: completed` and
`formal_result: false`. Test was not accessed.

Run directory:

`/home/wjy/pathvlm_r1_v1_formal/runs/stage1_sft_smoke/formal_7b_sft_smoke_n0008_seed0042_20260717_010639`

## Save, resume, and rotation evidence

- Step 1 completed with loss `1.9121` and gradient norm `43.0792`.
- The complete eight-rank ZeRO-2 optimizer checkpoint reloaded and resumed to step 2.
- Step 2 completed with loss `0.8095` and gradient norm `13.3279`.
- The complete eight-rank ZeRO-2 optimizer checkpoint reloaded and resumed to step 3.
- Step 3 completed with loss `0.1889` and gradient norm `5.7532`.
- With smoke `save_total_limit: 1`, checkpoint 1 was rotated after checkpoint 2 and checkpoint 2
  was rotated after checkpoint 3; only the latest complete checkpoint remains.

## Scientific snapshots

Model-only snapshots for steps 1, 2, and 3 all survived source-checkpoint rotation and independently
reloaded. Each snapshot is marked non-resumable and contains exact file sizes and SHA-256 hashes.
Their gathered model payloads are approximately 16.60 GB each. No optimizer state appears in a
scientific snapshot.

The sampled language tensor changed from base to step 1 and from step 1 to step 2. The sampled visual
tensor was exactly equal across both comparisons. LLaMA-Factory logged 7,615,616,512 trainable
parameters out of 8,292,166,656 and both required visual freezes.

## Backend evidence

All three training processes logged:

- gradient checkpointing enabled;
- torch bfloat16 ZeRO stage 2 optimizer;
- CPU offload false; and
- the full language-model trainability and visual/projector freeze policy.

Selected-GPU peak memory was at most 43,051 MiB during resume. All GPUs returned to the idle
18-MiB baseline after the smoke.

## Storage

Attempt02 currently occupies approximately 164 GiB: three scientific snapshots, the latest complete
resume checkpoint, and the final gathered model. Free space after completion is 1,208,059,944,960
bytes, still above the projected formal-start requirement before the small GC confirmation run.

## Next action

Run the correctly labeled 20-step `z2_gpu_gc_fused` throughput confirmation. If its configuration,
throughput, memory, and freeze gates pass and the formal projected disk reserve still passes, record
the result and launch only the fresh n=3000, seed-42 formal SFT duration sweep. Stage 2 and Stage 3
remain out of scope.

