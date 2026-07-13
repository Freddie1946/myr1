# Fixed-model download concurrency correction

Timestamp: `2026-07-13 21:43:52 Asia/Shanghai`

## State before correction

The standard HTTP retry completed PathMMU successfully:

- resolved dataset revision: `054e64e56e599e9636024f1471d49ecae4a2784f`;
- official `images.zip` size: 1,931,564,255 bytes;
- archive SHA-256: `9799d35d7d5dace0ccb34d71aea1c13900287fca50e8b9b51d282ea23b80eb36`;
- required/extracted/valid images: 3,809/3,809/3,809;
- missing images: 0.

The fixed 7B model snapshot then used the Hugging Face default concurrent downloader for five large
weight shards. Through the formal-machine proxy, simultaneous shard transfers repeatedly produced SSL
EOF and 10-second read timeouts. About 6.7 GiB of resumable local snapshot data existed when Codex
interrupted only its own bootstrap process. No completed shard was removed.

## Correction

- Set `HF_HUB_DOWNLOAD_TIMEOUT=120`.
- Set the formal snapshot download to `max_workers=1`.
- Expose the worker count as `HF_HUB_MAX_WORKERS=1` in the example and machine-local configs.
- Preserve Hugging Face's local incomplete files so the next run resumes rather than restarts.

This is a transfer-stability correction only. Model ID, immutable revision, expected config/index/
tokenizer hashes, and all preflight identity gates remain unchanged. No training has run, and GPU 0
remained occupied at the last snapshot.
