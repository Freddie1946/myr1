# Stage 1 SFT debug log

Timestamp: 2026-07-12 13:45:50 Asia/Shanghai

## Objective

Validate that Qwen2.5-VL-7B can execute one full-parameter language-model SFT optimizer step through LLaMA-Factory on the 4x RTX 4090 debug host. The vision tower and multimodal projector remain frozen.

## Environment

- Repository: `/home/wjy/LLaMA-Factory`, commit `ef5f1c1def3da62ee2d5e6ba933f9d7d6aab4340` (working tree is dirty; unrelated changes are preserved).
- Isolated overlay venv: `/home/wjy/revision_runs/10_training/pathvlm_r1_v1/env/pathvlm-sft-venv`.
- LLaMA-Factory 0.9.2, Transformers 4.49.0, Tokenizers 0.21.0, Torch 2.6.0+cu118, DeepSpeed 0.15.4.
- System CUDA toolkit is 12.1 while Torch was built for CUDA 11.8. Therefore DeepSpeedCPUAdam JIT compilation is not used.
- RTX 4090 distributed runs require `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` on this host.

## Attempt 01

Result: failed during forward execution.

- Confirmed the real 7B model: 8,292,166,656 total parameters.
- Confirmed 7,615,616,512 trainable language-model parameters (91.84%).
- Confirmed vision tower and multimodal projector are frozen.
- Replaced DeepSpeedCPUAdam with Torch AdamW due to the CUDA build mismatch.
- FlashAttention then failed with a rotary-embedding dtype mismatch (`float32` versus `bfloat16`).

## Attempt 02

Result: timed out because of CPU offload and swap pressure.

- Switched attention to SDPA; model initialization and forward/backward proceeded beyond the Attempt 01 failure.
- ZeRO-3 offloaded both parameters and optimizer state to CPU, using Torch AdamW.
- GPU memory stayed around 4.8 GiB per worker, but host virtual memory reached roughly 115 GiB and swap peaked around 89 GiB.
- The one-step run did not complete within approximately 15 minutes.
- The timed-out SSH session left no LLaMA-Factory or torchrun worker behind; later inspection showed memory recovered.

## Attempt 03

Status at this timestamp: prepared locally, not yet launched.

Change from Attempt 02:

- Keep ZeRO-3 model parameter partitions on GPU.
- Offload only optimizer states to CPU.
- Continue using Torch AdamW and SDPA.
- Use GPUs 0, 1, and 2 only; GPU 3 is excluded because of an unrelated VLLM process.

Expected tradeoff: much lower RAM/swap pressure at the cost of substantially higher GPU memory. The run must not start until current GPU occupancy is mapped and enough headroom is confirmed.

## Recovery procedure

1. Check exact GPU index/UUID mapping, running compute processes, free memory, RAM, swap, and disk.
2. Never kill or alter unrelated processes.
3. Sync Attempt 03 configuration, DeepSpeed JSON, manifest, and this timestamped documentation to the remote experiment root.
4. Launch with `CUDA_VISIBLE_DEVICES` restricted to safe GPUs.
5. Monitor logs and resource use; stop only our exact PIDs if the host becomes unstable.
6. Update a new timestamped execution log with the final outcome and next configuration.

## Attempt 03 command template

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HUB_OFFLINE=1 \
WANDB_MODE=disabled \
/home/wjy/revision_runs/10_training/pathvlm_r1_v1/env/pathvlm-sft-venv/bin/python \
  -m torch.distributed.run \
  --nproc_per_node=3 \
  --master_port=29571 \
  /home/wjy/LLaMA-Factory/src/llamafactory/launcher.py \
  /home/wjy/revision_runs/10_training/pathvlm_r1_v1/configs/sft/sft_smoke_full_z3_optimizer_offload_3gpu_a03.yaml
```

