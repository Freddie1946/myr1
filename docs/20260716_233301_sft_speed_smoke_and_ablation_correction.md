# SFT throughput smoke and ablation correction

Timestamp: `2026-07-16 23:33:01 Asia/Shanghai`

## Correction to checkpoint interpretation

The completed n=500 and n=1000 gathered final checkpoints remain intact and loadable. Automatic
`save_total_limit: 2` rotation removed their early per-epoch optimizer checkpoints. Therefore the
SFT data-scale study remains recoverable, but a retrospective epoch/early-stopping curve is not.
Future duration selection must retain explicitly selected model-only checkpoints or validate them
before pruning.

## Superseded n=2000 attempt

The user chose to measure and adopt a faster eight-GPU backend before continuing the formal scale
curve. The old CPU-optimizer-offload n=2000 run was interrupted through its torchrun parent. Its
runner recorded `status: failed`, `formal_result: false`, and the fail-closed scale supervisor
stopped without launching validation or n=3000. This history is retained and must not be rewritten.

## First throughput candidate

The first candidate deliberately tests the fastest configuration that can plausibly fit eight
48-GiB RTX 4090 GPUs while preserving the scientific optimizer and global batch:

- eight workers, per-device batch 1, global batch 8;
- full language-model fine-tuning with frozen vision tower and multimodal projector;
- BF16 and Torch fused AdamW;
- DeepSpeed ZeRO-2 with optimizer and gradients on GPU;
- communication overlap enabled;
- gradient checkpointing disabled;
- the same 512-token cutoff and 65,536 image-pixel cap;
- 64 records from the frozen SFT training adapter, seed 42, 20 optimizer steps;
- no validation or test access; `formal_result: false`.

Unsharded DDP is not an executable candidate: replicated BF16 model weights, gradients, and FP32
Adam master/moment states require roughly 115 GiB per GPU before activations. A non-DeepSpeed
alternative would still require state sharding, such as PyTorch FSDP.

If this candidate OOMs or fails, later timestamped attempts may re-enable gradient checkpointing,
then fall back to ZeRO-3 without optimizer offload. No fallback may run concurrently.
