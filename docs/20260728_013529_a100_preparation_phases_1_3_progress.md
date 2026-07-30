# A100 preparation phases 1--3 progress

Created: 2026-07-28 01:35:29 +08:00

This is a preparation-only record. No training, model generation, paid API request, validation
selection, or test evaluation was run. `formal_result` is false.

## Machine and workspace

- Workspace: `/home/dataset-assist-0/czy/wjy`
- Repository branch: `codex/a100-stage3-eval`
- Handoff commit: `e483e06adf2380923ace0998ee9211c300df59a7`
- Hardware audit: eight idle NVIDIA A100-SXM4-80GB GPUs, driver `580.65.06`, no MIG, no GPU
  processes at the preparation gate.
- The workspace shell now exposes local Git, Hugging Face CLI, micromamba and the project cache
  paths without relying on the machine-wide Codex or another user's home directory.

## Formal bootstrap state

- SFT and GRPO environments exist and import with Torch `2.6.0+cu124`.
- LLaMA-Factory is detached at `e2299e261be852304bb1d370515078193ab12bd8`; the frozen
  launcher hash passed.
- PathMMU revision `054e64e56e599e9636024f1471d49ecae4a2784f` was downloaded.
  The archive is 1,931,564,255 bytes with SHA-256
  `9799d35d7d5dace0ccb34d71aea1c13900287fca50e8b9b51d282ea23b80eb36`.
- Exactly 3,808 required images were extracted. All split hashes, physical image hashes,
  pairwise basename/content-disjointness, nested-subset, removed-test-image, parent-file,
  `picked.json` absence and test-blinding gates passed.
- The first resumed bootstrap then stopped before the base-model snapshot because the outbound
  proxy returned an intermittent Hugging Face TLS `UNEXPECTED_EOF_WHILE_READING`. The exact
  Qwen 7B snapshot download is being resumed with standard verified HTTPS, fixed revision and one
  worker; certificate verification has not been disabled.
- Formal preflight remains pending until that exact base snapshot is complete.

## Evaluation environments

All environments are under
`/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/envs` and are isolated from formal SFT
and GRPO:

| Environment | Core frozen stack | Preparation gate |
|---|---|---|
| `pathology_clip` | Torch 2.5.1+cu124, Transformers 4.49.0, timm 0.9.8 | CONCH, UNI and PLIP CPU imports pass; `pip check` passes |
| `qwen_vl` | Torch 2.6.0+cu124, Transformers 4.49.0, qwen-vl-utils 0.0.14 | Qwen2.5-VL class import passes; `pip check` passes |
| `llama32_vision` | Torch 2.6.0+cu124, Transformers 4.49.0 | Mllama class import passes; `pip check` passes |
| `deepseek_vl2` | Torch 2.0.1+cu117, Transformers 4.38.2, xformers 0.0.21 | DeepSeek-VL2 class import passes; `pip check` passes |
| `medgemma` | Torch 2.6.0+cu124, Transformers 4.57.3 | Gemma 3 image-text class import passes; `pip check` passes |
| `internvl3` | Torch 2.0.1+cu117, Transformers 4.37.2, timm 0.9.12 | InternVL chat class import passes; `pip check` passes |
| `huatuo_llava` | Torch 2.0.1+cu117, Transformers 4.37.2 | Huatuo Qwen2/LLaVA class import passes; `pip check` passes |
| `llava_med` | Torch 2.0.1+cu117, Transformers 4.36.2 | LLaVA-Med Mistral class import passes; `pip check` passes |
| `api_clients` | OpenAI 1.97.1, Anthropic 0.57.1, DashScope 1.23.9, Volcengine SDK 4.0.25 | imports pass; no endpoint call was sent |

The migration guide suggested a combined Huatuo/LLaVA environment. Both upstreams expose a
top-level `llava` package, so combining them can silently select the wrong source tree. They were
split into `huatuo_llava` and `llava_med`.

Compatibility corrections are explicit:

- The pathology Torch build is cu124 rather than the old cu121 command, matching the audited A100
  driver/runtime.
- DeepSeek-VL2 NumPy is pinned to 1.26.4 because Torch 2.0.1/torchvision 0.15.2 emit an ABI
  incompatibility under NumPy 2.x.
- InternVL's optional `decord==0.6.0` wheel is rejected by `pip check` on this platform. It was
  removed because this environment is for static-image VQA, not video inference.
- Huatuo's upstream `tokenizers==0.13.3` conflicts with its own Transformers 4.37.2 requirement.
  Tokenizers 0.15.1 is used. Timm 0.9.12 supports both the Huatuo and LLaVA-family evaluators.
- FlashAttention is not installed. Future BF16 smoke must explicitly use a supported eager/SDPA
  path; no quantized substitute is authorized.

Exact `pip freeze` files and their hashes are stored outside Git under
`pathvlm_revision_eval_a100/reports/manuscript_baselines`.

## Evaluation assets and access

- PathVQA fixed test revision `1685832883334b5bb5beaf4e4b333fdeecaa4ad9` is local.
  Its three shards contain 2,240, 2,240 and 2,239 rows, and all three expected SHA-256 values pass.
- OmniMedVQA fixed archive download has started and is resumable.
- Authenticated metadata checks pass for the exact revisions of Qwen 3B/7B, DeepSeek-VL2,
  Llama 11B/90B Vision, MedVLM-R1, MedGemma 4B, Lingshu 7B, InternVL3 8B,
  HuatuoGPT-Vision 7B, CONCH and UNI, plus public PLIP.
- Model weights remain preparation assets, not evidence that an adapter or end-to-end evaluation
  is complete.

## Checkpoint and experiment gates

- The required formal 3000-SFT parent is the seed-42 epoch-3/step-1125 checkpoint from the old
  machine. It is not present on this A100 machine.
- `Freddie1946/cot-3000-7b` is epoch 5/step 1250 and is not a valid substitute for that parent.
- The repository contains 3,000 SFT examples and a separate 1,000-example RL split. It does not
  define an “additional 1,000 SFT” set. The RL split must not be silently relabelled as SFT.
- Therefore 4000-SFT remains blocked on an exact parent transfer and a user-approved data/control
  definition. Stage 3 remains explicitly unstarted.
- Baseline evaluation remains blocked on freezing the common prompt, decoding, parser, sample
  universe, statistics, exact hosted API identities/credentials and cost authorization.
