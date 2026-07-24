# Manuscript baseline metadata, storage and gated-access audit

Audited: `2026-07-25T01:31:54+08:00`

This is a metadata/access preparation audit with `formal_result: false`. Official Hugging Face model
metadata was queried and three small `config.json` access checks were attempted under `/tmp`. No
weight was downloaded, no environment was modified, no inference/training was run, no GPU was
allocated and no validation/test record was opened.

## Fixed metadata for local manuscript baselines

The official repository heads and required `.safetensors`/`.bin` weight bytes observed in the
metadata response are:

| Manuscript row | Official repository candidate | Revision | Weight bytes |
|---|---|---|---:|
| DeepSeek-VL2 | `deepseek-ai/deepseek-vl2` | `f363772d1c47f4239dd844015b4bd53beb87951b` | 54,961,192,528 |
| LLaMA3.2-90B | `meta-llama/Llama-3.2-90B-Vision-Instruct` | `e305d2a43a4adc6987308fe7d896fb8ec5f1a5d8` | 177,186,907,758 |
| LLaMA3.2-11B | `meta-llama/Llama-3.2-11B-Vision-Instruct` | `9eb2daaa8597bf192a8b0e73f848f3a102794df5` | 21,340,560,870 |
| MedVLM-R1 | `JZPeterPan/MedVLM-R1` | `d256f2cfdf98c6872c1dc9f20b7dd52f49374fe9` | 4,418,059,032 |
| MedGemma-4B | `google/medgemma-4b-it` | `290cda5eeccbee130f987c4ad74a59ae6f196408` | 8,600,277,880 |
| Qwen2.5-VL-3B | `Qwen/Qwen2.5-VL-3B-Instruct` | `66285546d2b821cf421d4f5eb2576359d3770cd3` | 7,509,337,976 |
| Lingshu-7B | `lingshu-medical-mllm/Lingshu-7B` | `b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9` | 16,584,414,544 |
| InternVL3-8B | `OpenGVLab/InternVL3-8B` | `853e3a797a661694b1b8ece0cb72dc2b23e3dac9` | 15,888,831,920 |
| HuatuoGPT-Vision-7B | `FreedomIntelligence/HuatuoGPT-Vision-7B` | `34dfcdbb7728ff38da865839f342b88c4cf6ef39` | 17,583,367,129 |

The formal Qwen2.5-VL-7B snapshot remains separately pinned at
`cc594898137f460bfe9f0759e9844b3ce807cfb5` and occupies 16,595,999,395 bytes locally.

These revisions freeze the preparation target observed on this date. They are not yet a claim that
every candidate matches the exact unidentified historical snapshot used for the submitted table.

## Storage result

The nine not-yet-local manuscript models above total:

- required recognized weight files: 324,072,949,637 bytes = 324.07 decimal GB = 301.82 GiB;
- every repository sibling without an allowlist: 523,100,720,447 bytes = 523.10 decimal GB =
  487.18 GiB.

The large difference is mainly caused by repositories that publish multiple weight formats. A
naive full snapshot download is forbidden. Every model needs a reviewed allowlist that selects one
official weight format plus required configuration/tokenizer/processor/custom-code files.

At audit time `/home` had 542,882,775,040 bytes available (about 506 GiB), already below the formal
550-GiB reserve. Even the selective 301.82-GiB weight set would leave only about 204 GiB before
environments, caches and outputs. Bulk download is therefore blocked by the existing storage gate
until additional storage is assigned or explicitly approved cleanup/retention changes are made.

## Gated access result

Authenticated fixed-revision `config.json` checks produced:

- `google/medgemma-4b-it`: pass; the small config was downloaded to `/tmp`;
- `meta-llama/Llama-3.2-11B-Vision-Instruct`: fail with HTTP 403, account not authorized;
- `meta-llama/Llama-3.2-90B-Vision-Instruct`: fail with HTTP 403, account not authorized.

The previously approved CONCH and UNI access remains separate and passed in the earlier audit.
The current Hugging Face account must request/receive access on both Meta Llama 3.2 Vision model
pages before those two mandatory rows can be prepared. A pending request is not equivalent to
access.

## Consequences

1. All fifteen manuscript baselines remain mandatory.
2. Exact model revisions can now be frozen for the nine missing local candidates, subject to
   historical-identity review.
3. MedGemma setup may proceed after its isolated environment is pinned.
4. Meta Llama 11B/90B setup is access-blocked.
5. No bulk download should begin before a selective-file manifest and storage destination are
   approved.
6. Hosted/API row identity and fresh credentials remain unresolved and are unaffected by Hugging
   Face access.

