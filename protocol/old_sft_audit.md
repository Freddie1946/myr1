# Historical LLaMA-Factory SFT audit

Audited: 2026-07-11

## Successful local run found

The directory below is named as a 7B experiment, but its saved `config.json` proves that it is Qwen2.5-VL-3B-Instruct (`hidden_size=2048`, 36 layers):

`/home/wjy/LLaMA-Factory/saves/qwen2_5_vl-7b-3.24.0300_withoutcot/full/sft`

The corresponding W&B run is:

`run-20250324_030316-ujmhkibd`

Recovered parameters:

- Base: `Qwen/Qwen2.5-VL-3B-Instruct`.
- Dataset: `sft_without_cot_data_sample1000`.
- Dataset size: 1000.
- Fine-tuning: full language-model fine-tuning.
- Vision tower: frozen.
- Multimodal projector: frozen.
- DeepSpeed: ZeRO-3 without CPU optimizer offload.
- Four RTX 4090 GPUs.
- Per-device batch size: 1.
- Gradient accumulation: 1.
- Effective global batch: 4.
- Cutoff length: 1024.
- Learning rate: `2e-5`.
- Scheduler: cosine.
- Epochs: 10.
- BF16: enabled.
- Validation: disabled.
- Steps: 2500.
- Runtime: 5411 seconds.
- Final training loss: approximately 0.0691.

This run is legacy evidence only. Its directory name must not be cited as proof of a 7B model.

## Historical 7B attempts

Five W&B runs on 2025-03-28 reference the real `Qwen/Qwen2.5-VL-7B-Instruct` model. Their runtimes were only several seconds and no successful output checkpoint was found.

The last attempt used:

- Dataset: `sft_with_cot_data_sample3000`.
- Full language-model fine-tuning.
- Frozen vision tower and multimodal projector.
- DeepSpeed ZeRO-3 without CPU optimizer offload.
- Four RTX 4090 GPUs.
- Per-device batch size 1, gradient accumulation 1.
- Cutoff length 512.
- Image maximum pixels 65536.
- Learning rate `2e-5`.
- Ten epochs.
- Gradient checkpointing enabled.
- Seed 42.

It failed during the first optimizer step while allocating AdamW state, with a CUDA out-of-memory error requesting approximately 3.76 GiB on GPU 0.

Conclusion: no successful local LLaMA-Factory 7B SFT run was found. The revised 7B pipeline must be established anew and validated with a smoke test before formal training.

## Historical software evidence

- LLaMA-Factory commit recorded by W&B: `ef5f1c1def3da62ee2d5e6ba933f9d7d6aab4340`.
- Historical Python: 3.10.16.
- Historical Transformers: 4.49.0.
- Historical host: four RTX 4090 GPUs; CUDA reported as 12.2.
- The current `vlm-r1` environment has drifted and should not be treated as the historical lockfile.

