# Stage 2 Attempt 04 ready, waiting for GPUs

Timestamp: 2026-07-12 18:43:22 Asia/Shanghai

## Completed before the wait

- Parser v2 is synchronized to the remote experiment root.
- Local and remote reward regression tests pass.
- Attempt 03 is marked invalid for scientific use because its policy update used the old false-positive accuracy parser.
- Attempt 03 audit artifacts are retained: logs, online rewards, corrected offline reward audit, parameter delta, command, manifest, and model/tokenizer metadata.
- Attempt 03's two invalid model shards were deleted after approval.
- Disk now has approximately 711 GiB free because other shared users also removed substantial data concurrently; our deletion accounted for only about 7.1 GiB.
- Attempt 04 command and manifest are synchronized remotely.
- Attempt 04 output directory does not yet exist; it has not launched.

## Current resource state

- GPU 0: unrelated `zxc` process, about 8.8 GiB plus high utilization.
- GPU 1: unrelated `zw` OAGA-FastWAM evaluation, about 13.3 GiB.
- GPU 2: unrelated `zw` OAGA-FastWAM evaluation, about 14.1 GiB.
- GPU 3: unrelated VLLM service, about 22.2 GiB.

No unrelated process was modified or terminated.

## Why the run is paused

Attempt 04 requires policy and reference models plus a global batch of four on two GPUs. Running on GPUs 1 and 2 while these evaluations are active would interfere with other users and is likely to exceed memory. A one-GPU CPU-offload fallback would add substantial RAM/swap risk and would not be a cleaner validation than waiting for two free GPUs.

## Resume procedure

1. Confirm there are no stale PathVLM processes.
2. Check physical GPUs 1 and 2; both should have roughly 20 GiB or more free.
3. Confirm Attempt 04 output directory is absent.
4. Re-run `test_pathmmu_rewards.py`.
5. Launch the saved Attempt 04 command on GPUs 1 and 2.
6. Audit the eight online reward events with parser v2.
7. Require positive reward variance and gradient norm.
8. Compare a language tensor and a visual tensor against the original SFT parent.
9. Only mark Stage 2 complete if parser correctness and parameter-delta gates all pass.

