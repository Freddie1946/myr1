# Strengthened formal preflight rerun completed

Timestamp: `2026-07-14 00:20:42 Asia/Shanghai`

After installing the two declared Open-R1 dependencies and adding strict adapter reuse validation,
the formal preflight was rerun against the existing installation. The refreshed report is
`/home/wjy/pathvlm_r1_v1_formal/reports/preflight_report.json`, written at approximately 00:20 local
time. It reports `passed: true` and all 21 gates passed.

The passing gates include exact Qwen2.5-VL 7B model identity, revision and audited config/index/
tokenizer hashes; official LLaMA-Factory v0.9.2 commit and launcher hash; complete frozen-split
verification with zero pairwise top-level image overlap; `picked_json_used: false`; exact adapter
source counts; pinned SFT and GRPO environments; LLaMA-Factory and Open-R1 imports; reward regression
tests; and clean `pip check` results in both environments.

The same preflight captured eight visible NVIDIA GeForce RTX 4090 devices with 49,140 MiB each. The
previous GPU-0 Python training process was absent. GPUs 1--7 showed only the 4 MiB Xorg allocation;
GPU 0 showed Xorg plus NX remote-desktop and Firefox allocations (approximately 561 MiB total).
This observation does not authorize training and is not a substitute for a fresh occupancy check
immediately before any GPU work.

No download, optimizer step, smoke run, or formal training was performed during this correction and
preflight rerun.
