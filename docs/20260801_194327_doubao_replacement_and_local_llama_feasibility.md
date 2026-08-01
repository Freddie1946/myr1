# Doubao replacement and local Llama feasibility

Recorded at: `2026-08-01T19:43:27+08:00`

## Doubao

The historical `doubao-1.5-vision-pro-250328` AIGCBest endpoint remains unusable (terminal 404), and the catalog alias `Doubao-1.5-vision-pro-32k` serves a different dated identity. Two exact contemporary Doubao vision candidates were therefore subjected to the same fixed four-case behavioral smoke and one-case GPT-5 mini PathVQA semantic-judge smoke:

- `doubao-seed-1-6-vision-250815`: exact identity, 4/4 transport/scoring paths passed, semantic-judge path passed. This is selected as the continuity replacement because it is a fixed dated vision-specific successor to the 1.5 Vision family. Its PathMMU test999 full evaluation has started with two workers.
- `doubao-seed-2-0-mini-260428`: exact identity, 4/4 transport/scoring paths passed, semantic-judge path passed. It remains an optional stronger/current multimodal-reasoning baseline and must not be relabelled as Doubao 1.5. No full evaluation has started for it.

Smoke accuracy was not used to select the continuity replacement. The decision is based on lineage, task identity and reproducible dated model identity.

## Llama 3.2 Vision local deployment

Local deployment is technically feasible but currently license/access blocked.

- Hugging Face account: `Freddie1946`.
- Both official repositories expose metadata, but authenticated HEAD requests for `config.json` and the first safetensors shard return `GatedRepoError 403` stating that the access request was rejected by the repository authors.
- The local 11B and 90B directories contain only README/LICENSE material, approximately 144 KiB each; no weights are present.
- Frozen revisions remain `9eb2daaa8597bf192a8b0e73f848f3a102794df5` for 11B and `e305d2a43a4adc6987308fe7d896fb8ec5f1a5d8` for 90B.
- Transformers-format BF16 weight sizes are 21,340,560,870 bytes for 11B and 177,186,907,758 bytes for 90B. Repository totals are approximately double because the repositories also contain original-format duplicate weights; only the Transformers files should be staged.
- The isolated `llama32_vision` environment has PyTorch 2.6.0+cu124, Transformers 4.49.0, Accelerate 1.4.0 and `MllamaForConditionalGeneration` support. System vLLM 0.24.0 is installed, although it is not installed inside the isolated environment.
- 11B BF16 should fit on one 80GB A100 for batch-one deterministic evaluation. 90B BF16 requires multi-GPU sharding; four A100-80GB devices are a practical minimum and eight are preferred for throughput/headroom.
- Current disk (about 5 TiB free) and host RAM (about 811 GiB available at audit time) are sufficient.

No Llama download or inference was started. The accepted paths are: obtain official gated access, or copy an author-owned legally downloaded snapshot from another machine and verify the fixed revision/file hashes. Third-party mirrors will not be treated as the official baseline.
