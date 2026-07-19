# Formal n=500/n=1000 comparability and storage audit

Timestamp: `2026-07-19T21:14:17+08:00`

## Existing results are valid but final-only

The existing seed-42 n=500 and n=1000 runs are genuine formal 7B SFT results:

- exact pinned Qwen2.5-VL-7B base;
- 10 epochs, learning rate `2e-5`, seed/data seed 42;
- full language-model training with vision tower/projector frozen;
- finite nonzero gradients, language tensor changed, visual tensor exactly unchanged;
- final checkpoint saved and reloaded;
- test not accessed.

Their deterministic 385-QA validation results and raw predictions also pass their original gates:

| Size | Retained point | Accuracy | Format | Current parser mismatches |
| ---: | --- | ---: | ---: | ---: |
| 500 | epoch 10 final | 49.87% | 99.48% | 0/385 |
| 1000 | epoch 10 final | 51.43% | 95.84% | 0/385 |

The v1 and v2 rewritten validation files have the same SHA-256, so the data correction does not
invalidate these validation generations.

## Why they are not comparable to the selected n=2000/n=3000 curve

The n=500/n=1000 runs used the earlier ZeRO-3 optimizer-offload configuration, `save_steps: 100`,
and retained only the last two resumable checkpoints plus the final output. Their available resume
points are approximately epoch 9.52/10.0 for n=500 and 9.6/10.0 for n=1000. Earlier epoch models no
longer exist.

The n=2000/n=3000 runs used the later frozen ZeRO-2 GPU fused-AdamW backend and the two-tier policy
that preserved all ten model-only epoch snapshots. Attempt02 found their best validation points at
epoch 5 and epoch 3; their epoch-10 accuracies were 5.97 and 4.94 percentage points below their peaks.
It is therefore unsafe to assume that the n=500/n=1000 final checkpoints are their best points.

Two scientifically distinct analyses are possible:

1. A fixed-10-epoch endpoint table can truthfully report 49.87%, 51.43%, 51.95%, and 55.58% for
   n=500/1000/2000/3000. This is duration-matched by epoch, but backend/numerical implementation is
   not fully matched and it does not answer best-validation performance per scale.
2. The preferred primary scaling curve reruns n=500/n=1000 under the exact n=2000/n=3000 backend,
   checkpoint-retention policy, seed, and v2 data contract, then applies the same frozen selection
   rule over epochs 1--10.

The second option is recommended for the reviewer response.

## Time and storage implications

Historical wall times were approximately 5.4 hours for n=500 and 10.8 hours for n=1000. A cautious
budget for matched reruns plus validation is approximately 17 hours total; actual speed may differ.

Current `/home` free space is approximately 931 GiB. The audited runner requires about 1.08 TiB free
at each start, so it correctly cannot launch yet. Four old redundant resumable-state directories
consume about 408 GiB:

- n=500 `checkpoint-600` and `checkpoint-630`: about 102 GiB each;
- n=1000 `checkpoint-1200` and `checkpoint-1250`: about 102 GiB each.

These are optimizer/resume states, not the final 16.6-GB model outputs, raw validation predictions,
or the scientific epoch snapshots needed by the new curve. Because deletion is irreversible, none
was removed during this audit.

If the user approves matched reruns, the storage-safe sequence is:

1. hash/record then remove only the four old redundant resume directories; preserve old final models,
   manifests, logs, tensor gates and raw predictions;
2. run matched n=500 with ten scientific epoch snapshots;
3. after completion/reload/tensor gates, remove only its two optimizer resume checkpoints while
   preserving final model plus all ten epoch snapshots;
4. run matched n=1000 and apply the same post-completion optimizer-state policy;
5. validate base/n=500/n=1000 epochs with the same v2 data, template, parser, decoding and selection
   rule; never access test.

This requires a new fail-closed manifest/supervisor before execution. It must not reuse or relabel the
old results, and every deletion must receive explicit approval and an immutable pruning record.
