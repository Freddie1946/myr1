# Formal 7B SFT smoke Attempt02 completed

Timestamp: `2026-07-14 01:08:13 Asia/Shanghai`

## Outcome

Attempt02 completed the full formal-machine engineering gate with exit status 0. It used the exact
Qwen2.5-VL-7B base revision, eight frozen SFT records, seed 42, and physical GPUs 1--4. The language
model was fully trainable while the vision tower and multimodal projector were frozen. This is an
engineering smoke and remains correctly labelled `formal_result: false`.

Run directory:

`/home/wjy/pathvlm_r1_v1_formal/runs/stage1_sft_smoke/formal_7b_sft_smoke_n0008_seed0042_20260714_005945`

## Save, reload, and resume gates

- Step 1 completed with loss `1.8846` and gradient norm `45.9486083984375`.
- Gathered `checkpoint-1` independently loaded as Qwen2.5-VL with 8,292,166,656 parameters.
- Resume from the locally generated DeepSpeed checkpoint reached global step 2.
- Step 2 completed with loss `1.1769` and gradient norm `17.422452926635742`.
- Gathered `checkpoint-2` independently loaded.
- The narrow PyTorch 2.6 compatibility override was applied only to the trusted local resume
  subprocess. It was not enabled for downloads or arbitrary checkpoints.

## Trainability and tensor gates

LLaMA-Factory reported 7,615,616,512 trainable parameters out of 8,292,166,656 and logged both the
vision-tower and multimodal-projector freezes.

- Base to checkpoint 1: the sampled language tensor changed in 3,758,274 elements; the sampled
  visual tensor had zero changed elements and was exactly equal.
- Checkpoint 1 to checkpoint 2: the sampled language tensor changed in 3,764,940 elements; the
  sampled visual tensor had zero changed elements and was exactly equal.

All nine manifest gates passed: optimizer step, both checkpoint saves, both independent reloads,
resume, both tensor-delta checks, and the trainability/freeze check. Test was not accessed.

## Resources and storage

- Peak selected-GPU memory was about 9.45 GiB.
- Minimum available host memory was 70,203,020 KiB during resume.
- Peak swap use was 11,144,652 KiB during resume.
- Attempt02 occupies approximately 220 GiB.
- Failed Attempt01 remains preserved separately at approximately 118 GiB; its history was not
  rewritten or deleted.

The formal save/reload/resume gate is therefore closed successfully. A fresh eight-GPU occupancy
check and the audited scale launcher remain mandatory before a long run.
