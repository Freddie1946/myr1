# GPT-4o Stage3 selected, uploaded, and local GPU baselines started

Recorded at: `2026-08-03 01:16:01 CST`

## GPT-4o Stage3 completion and validation selection

The formal GPT-4o process-GRPO execution reached all 1,500 optimizer steps. Its audited Judge
ledger closed at `$63.12758000000005` for 12,275 completed unique requests, with no open
reservations or budget breach. The bounded structural fallback was used twice in total and zero
times consecutively at completion.

All three model-only epoch snapshots passed file-by-file size and SHA-256 verification. The frozen
PathMMU validation385 results were:

| Step / epoch | Correct | Accuracy | Strict format | Cap hits |
| --- | ---: | ---: | ---: | ---: |
| 500 / 1 | 234/385 | 60.78% | 385/385 | 0 |
| 1000 / 2 | 247/385 | 64.16% | 385/385 | 0 |
| 1500 / 3 | 245/385 | 63.64% | 385/385 | 0 |

The predeclared rule therefore selected step 1000 / epoch 2 by validation accuracy. PathMMU
test999 was not accessed by this selection pipeline.

## Private Hugging Face model backup

The selected 16.6-GB model-only snapshot was uploaded to the private model repository
`Freddie1946/PathVLM-R1-Process-GRPO-GPT4o-n1000-seed42-epoch2` at immutable revision
`3ade3cffd46b64abc864ed9f271b47632810ec9c`.

Remote verification covered all 18 local files and 16,600,833,380 bytes. The four model shards and
other LFS objects were checked against remote LFS SHA-256 values; ordinary Git objects were
re-downloaded at the fixed revision and SHA-256 checked. The verification record is
`/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/stage3_process_grpo/gpt4o_full3epoch_seed42_20260801/hf_remote_verification.json`
(SHA-256 `7b690a04eb5e2fd5095ce2a5cae1e2aa5c9688d90c33a1fb021724aad8154048`).
Optimizer/scheduler checkpoints were not uploaded.

## Local-baseline provenance and smoke gates

Full SHA-256 lists were computed for every ModelScope-mirror Llama weight shard:

- 11B: 5/5 shards; list SHA-256
  `1e66b76d15507c98edaf53e808be17408c461fb4c40c24cd160125477b551108`;
- 90B: 37/37 shards; list SHA-256
  `f83427eb07d33b2786206034f0789d7723b206522eadcb8d4ac83a77aa376b41`.

The frozen 16-case aggregate smokes produced these outcomes:

| Model/task | Nonempty | Parseable | Cap hits | Gate |
| --- | ---: | ---: | ---: | --- |
| Llama 3.2 11B / PathMMU | 16/16 | 13/16 | 0/16 | pass |
| Llama 3.2 11B / PathVQA | 16/16 | task types covered | 9/16 | **fail** |
| Llama 3.2 11B / OmniMedVQA | 16/16 | 16/16 | 3/16 | pass |
| DeepSeek-VL2 / PathVQA | 16/16 | 14/16 | 0/16 | pass |
| DeepSeek-VL2 / OmniMedVQA | 16/16 | 16/16 | 0/16 | pass |

Accuracy was not used as a smoke gate and no answer was modified or re-Judged. Llama 11B
PathVQA failed because its 56.25% generation-cap-hit rate exceeded the predeclared 20% maximum;
that full run is not launched. Llama 11B PathMMU test999 development diagnostics and full
OmniMedVQA, plus DeepSeek-VL2 full PathVQA and OmniMedVQA, are running on separate GPUs. The prior
DeepSeek-VL2 PathMMU response-contract failure remains terminal. Llama 90B smokes and eligible full
runs remain queued until GPUs 0/1 are released, because that model requires all eight A100s.

The fresh Kimi Stage3 arm remains last in the execution order. The user's OpenRouter top-up was
verified as `$27.312309725` available, which covers the retained `$26.71782984` Kimi authorization.

