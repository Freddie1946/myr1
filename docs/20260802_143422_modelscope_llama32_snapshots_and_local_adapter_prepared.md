# ModelScope Llama 3.2 Vision snapshots and local adapter prepared

Recorded at: `2026-08-02 14:34:22 CST`

## Outcome

The earlier statement that Llama 3.2 Vision weights were absent is no longer current.  Complete
ModelScope-sourced directory snapshots are present locally for both manuscript rows:

- Llama 3.2 11B Vision Instruct: 5 indexed shards, 21,340,560,870 weight bytes;
- Llama 3.2 90B Vision Instruct: 37 indexed shards, 177,186,907,758 weight bytes.

Both totals exactly match the weight-byte totals recorded earlier from official Hugging Face
metadata.  Each local configuration declares `MllamaForConditionalGeneration`; every indexed
shard exists; the tokenizer, processor, chat template, Meta license and use-policy files are
present.  This proves structural completeness and size agreement, not byte-for-byte equivalence
to the access-gated official HF revision.  Full per-shard SHA-256 hashing remains a pre-execution
provenance step.

The official HF repositories still return the earlier account-gated response.  These local
snapshots must therefore be reported transparently as ModelScope-sourced mirrors, not as an
official-HF access success.

## Adapter preparation

The two frozen local inference runners now accept a `mllama` backend:

- `scripts/run_pathmmu_qwen_diagnostic.py` for the already disclosed PathMMU test999 development
  diagnostic;
- `scripts/run_external_vqa_qwen.py` for full PathVQA and OmniMedVQA evaluation.

The backend loads `MllamaForConditionalGeneration` in BF16 with no quantization.  It uses
Accelerate device mapping and an explicit 76-GiB-per-visible-GPU ceiling so the 90B row can span
the eight A100s without CPU/disk offload.  Input tensors are placed on the first mapped model
device; the existing deterministic prompts, source hashes, append-only prediction rows and frozen
scoring contracts are unchanged.

No weight was loaded and no GPU inference/test record was accessed during this code preparation.
After GPT-4o Stage3 and its validation finish, each Llama row must first pass fixed adapter and
response-contract smokes.  Full PathMMU/PathVQA/OmniMedVQA runs may start only after those smokes
and the full snapshot hash manifests pass.

## Other downloaded supplemental models

The previously requested snapshots are also structurally present:

- `Qwen/Qwen3.5-4B`, fixed local directory revision label
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, two indexed shards totaling 9,319,828,096 bytes;
- `nvidia/LocateAnything-3B`, fixed local directory revision label
  `c32291ca5e996f5a7a485845b4f57a233936bba0`, two indexed shards totaling 7,661,427,376 bytes.

They are supplemental candidates, not original manuscript baselines.  LocateAnything is a
localization/grounding architecture and must not be assigned the generative VQA scoring contract
without a separate task justification and adapter smoke.

## Verification

- Both real Llama snapshot structural verifications passed.
- Mllama imports under the isolated environment (`transformers 4.49.0`, `torch 2.6.0+cu124`).
- Both modified runners parse `--help` under that environment.
- Thirteen relevant unit tests passed.
- Python compilation and `git diff --check` passed.

