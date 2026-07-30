---
library_name: transformers
license: other
base_model: Freddie1946/PathVLM-R1-SFT-n3000-seed42-epoch3
tags:
- qwen2_5_vl
- pathology
- visual-question-answering
- full-parameter-finetuning
---

# PathVLM-R1 SFT4000 control — seed 42, epoch 2

This private repository stores the fixed model-only checkpoint for the exposure-matched SFT4000
control. It is a Qwen2.5-VL-7B pathology model continued from the selected SFT3000 parent on the
exact 1,000 records allocated to the Outcome-GRPO branch, using ordinary supervised loss.

This is not a fresh-base SFT run on 4,000 jointly shuffled records. “SFT4000” means 3,000 distinct
records seen by the parent plus the additional disjoint RL1000 records used here as supervised
targets.

## Fixed checkpoint

- parent: `Freddie1946/PathVLM-R1-SFT-n3000-seed42-epoch3`
- parent revision: `97aae0969edd2824191f7bf15cd67835c1794ecf`
- seed/data seed: `42`
- additional records: `1,000`
- additional epochs: `2`
- global batch size: `8`
- optimizer steps: `250` (`125` per epoch)
- learning rate: `2e-5`
- scheduler: cosine with warmup ratio `0.03`
- dtype: BF16
- language model: fully trainable
- vision tower and multimodal projector: frozen
- checkpoint role: fixed epoch-2/step-250 evaluation control
- format: model-only Transformers snapshot; not resumable optimizer state

Snapshot-manifest SHA-256:

`d966b80a86a74495f35e0a2f34e2c66e1347a674f6db8fa96ac922afa8dc5e1c`

Model index SHA-256:

`3067e9b0f35596ff3426a0d0ec8c982a51fa1e110c4fc30dcf3be9ea37409df6`

The model index references four safetensor shards and 729 tensors. All indexed tensors and shard
headers were checked before upload.

## Recorded evaluations

These results are supplied for provenance and must retain their stated roles:

- PathMMU test999 development diagnostic: `595/999` (`59.56%`), not an untouched final test;
- PathVQA external test, normalized exact match: `868/6719` (`12.92%`);
- OmniMedVQA four-source external test, official option accuracy: `4030/8518` (`47.31%`).

The PathVQA run used a 64-token generation cap and had 2,396 cap hits; the OmniMedVQA run had 6,645
cap hits. These limitations must accompany any interpretation of those external scores.

## Intended role

Use this checkpoint as the causal control for comparing continued SFT on the RL1000 records against
Outcome-GRPO from the same selected SFT3000 parent. It was not selected using PathMMU test999,
PathVQA or OmniMedVQA.

The full training protocol, logs, diagnostic metrics and integrity records are preserved on the
GitHub branch `codex/a100-stage3-eval`.
