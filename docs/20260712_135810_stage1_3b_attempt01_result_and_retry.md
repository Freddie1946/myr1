# Stage 1 3B Attempt 01 result and retry

Timestamp: 2026-07-12 13:58:10 Asia/Shanghai

## Attempt 01 result

Failed before tokenizer/model loading. The configured revision `66285546d2b821cf421d4f5eb2576359d3770cd3` is the current cache `refs/main` target but contains only `generation_config.json` and `preprocessor_config.json`. It has no `config.json`, tokenizer, or weight shards.

- No forward/backward or optimizer step occurred.
- No checkpoint was created.
- Peak resident host memory was under 1 GiB for the launcher.
- GPUs 1 and 2 returned immediately to their prior idle state.
- Test data was not accessed.

## Cache audit

Two older snapshots, `1b989f2c...` and `319ccfdc...`, both contain the complete config/tokenizer/processor set and two weight shards. Attempt 02 pins `319ccfdc6cd974fab8373cb598dfe77ad93dedd3`; it does not rely on the mutable/broken `refs/main` pointer.

## Attempt 02

The training method, data, seed, and GPUs remain unchanged. Only the base snapshot path and output/log directory change. Before launch, load `AutoConfig`, tokenizer, and processor offline from the pinned snapshot and verify both model shard targets exist.

