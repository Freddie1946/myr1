# Formal GPU allocation and machine-local configuration

Timestamp: `2026-07-13 20:49:27 Asia/Shanghai`

## Allocation

The user confirmed that all eight physical NVIDIA GPUs on `YiwuServer`, indexed `0-7`, are allocated
to the PathVLM-R1 formal work after a collaborator stops the existing process on GPU 0. The process
was still present at the time of the earlier inventory, so no CUDA training may start until a fresh
`nvidia-smi` confirms that it has exited. No other user's process may be stopped by Codex.

## Machine-local configuration

Created gitignored `formal_machine.env` with:

- `INSTALL_ROOT=/home/wjy/pathvlm_r1_v1_formal`;
- frozen split root under the current Git checkout;
- online gated PathMMU acquisition;
- exact Qwen2.5-VL-7B revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`;
- `/opt/miniconda3/condabin/conda`;
- PyTorch 2.6.0 and torchvision 0.21.0 from the official cu124 wheel index;
- physical GPUs `0,1,2,3,4,5,6,7` and `NPROC_PER_NODE=8`;
- GRPO per-device batch 1, giving global batch 8, divisible by four generations.

The file passed Bash syntax, Git-ignore, path/value, and GRPO divisibility checks. It contains no
token. Bootstrap remains blocked on successful Hugging Face authentication for gated PathMMU access
and the fresh eight-GPU occupancy check.
