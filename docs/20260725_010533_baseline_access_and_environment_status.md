# Baseline access and environment status audit

Audited: `2026-07-25T01:05:33+08:00`

This is a read-only status audit with `formal_result: false`. No weight was downloaded, no
environment was modified, and no model inference, training, GPU allocation or test scoring occurred.

## Newly proposed pathology/medical baselines

| Baseline | Source | Environment/import | Weight access | Weight local | Formal adapter | Status |
|---|---|---|---|---|---|---|
| PLIP | pinned | pass through Transformers/source import | public | no | no | environment ready; model not runnable yet |
| CONCH | pinned | editable install and CPU import pass | authenticated HEAD pass | no | no | access resolved; download/adapter pending |
| UNI | pinned | editable install and CPU import pass | authenticated HEAD pass | no | no | access resolved; positioning protocol pending |
| LLaVA-Med v1.5 | pinned source only | dedicated environment absent | not re-audited in this step | no | historical script only, not formal-ready | not configured |
| PathChat | no verified public source package | absent | unresolved | no | no | writing comparison only unless an official release is found |

The authenticated fixed-weight checks now pass:

- CONCH revision `f9ca9f877171a28ade80228fb195ac5d79003357`,
  `pytorch_model.bin` size 802,235,437 bytes, ETag
  `40a9644b9ba0e83a74576e0a5e5f7313599fa9c9cdaf3c20f8a3e271b0e9ae7c`.
- UNI revision `b55a5ec6cade1a39edfe6534189a9b8ca7a022f0`,
  `pytorch_model.bin` size 1,213,527,781 bytes, ETag
  `56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40`.

This resolves the external license-access block only. It does not by itself make either baseline
runnable. Exact weight download, local hash verification and a frozen task-appropriate adapter are
still required.

## Baselines already listed in the manuscript

The manuscript Table II names:

- API/hosted large models: Doubao-1.5-vision, Qwen-VL-Plus, Claude-3.5-Haiku, Grok-4-Fast and
  Claude-4.5; the historical execution path also treated some other large models through an
  OpenAI-compatible gateway.
- Local/open-weight candidates: DeepSeek-VL2, LLaMA3.2-90B, LLaMA3.2-11B, MedVLM-R1,
  MedGemma-4B, Qwen2.5-VL-3B, Qwen2.5-VL-7B, Lingshu-7B and InternVL3-8B.
- Table IV additionally names HuatuoGPT-Vision-7B.

Only the pinned Qwen2.5-VL-7B baseline is complete on this formal machine: exact model snapshot,
formal SFT environment and deterministic validation infrastructure are present and already
validated.

The other listed local/open-weight baselines do not have verified local weight snapshots or isolated
reproducible environments on this machine. The previously documented paths
`/home/wjy/LLaMA-Factory/saves`, `/home/wjy/HuatuoGPT-Vision`,
`/home/wjy/revision_runs/04_baselines/repos` and
`/home/wjy/revision_runs/04_baselines/outputs` are absent. Historical scripts exist under
`/home/wjy/PathVLM-R1/src/eval`, but they contain stale machine-specific paths, inconsistent model
classes/prompts/parsers and no complete raw-output provenance. They must not be treated as configured
formal baselines.

The common `wjy` environment has `openai==2.44.0`, but not the Anthropic or DashScope SDKs. More
importantly, the historical gateway credentials were invalidated by the user, so the API baselines
are not runnable. A fresh provider/endpoint/model-version contract and credentials outside Git are
required before they can be reproduced.

## Current conclusion

“Environment configured” is currently true for:

1. formal Qwen2.5-VL-7B;
2. the shared PLIP/CONCH/UNI dependency environment and import layer.

“End-to-end baseline runnable and auditable” is currently true only for Qwen2.5-VL-7B. PLIP, CONCH
and UNI still need weights and adapters; LLaVA-Med and all other manuscript local baselines still
need dedicated environment/weight preparation. API baselines need new credentials and frozen
provider contracts.

The next safe preparation order is PLIP weight plus option-matching adapter, CONCH weight plus the
same matched adapter, then a separate decision for UNI positioning and LLaVA-Med. Original
manuscript baselines should be triaged into a small mandatory rerun set rather than downloading every
model in Table II without raw-provenance and comparability plans.
