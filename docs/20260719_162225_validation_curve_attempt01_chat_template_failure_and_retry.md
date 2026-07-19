# Validation curve Attempt01 chat-template failure and retry

Timestamp: `2026-07-19T16:22:25+08:00`

## Attempt01 outcome

The full hash/GPU preflight passed, and Attempt01 created:

`/home/wjy/pathvlm_r1_v1_formal/runs/stage1_validation_curves/sft_base_n2000_n3000_seed0042_20260719_155520`

It ended with `status: failed_gate`, `formal_result: false`, and `test_accessed: false`. The runner's
stop event prevented every job that had not already started from launching.

- Base and n=2000 epochs 1--5 and 7 completed 385/385 predictions and their per-job gates.
- n=2000 epoch 6 failed before producing any prediction.
- The remaining 13 jobs were never started.
- All running jobs were allowed to finish normally; no process was killed.
- All GPUs returned to the 18-MiB idle baseline.
- Attempt01 outputs remain an engineering failure record and must not be merged into Attempt02.

## Root cause

The two-tier archiver correctly retained every file that existed in each source checkpoint, but its
required scientific model-file set did not require `chat_template.json`. LLaMA-Factory wrote that
file inconsistently across epoch checkpoints. It is absent from:

- n=2000 epoch 6 (step 1500); and
- n=3000 epochs 1, 5, and 9 (steps 375, 1875, and 3375).

The other 16 snapshots contain `chat_template.json` with one identical file hash. Parsing the JSON
shows that their `chat_template` string is exactly equal to the frozen base model's template string.
The file hashes differ only because the base JSON is pretty-printed and the snapshot JSON is compact.

Attempt01 relied on `AutoProcessor.from_pretrained(checkpoint)` to discover a checkpoint-local chat
template. At n=2000 epoch 6, `processor.chat_template` was `None`, so
`processor.apply_chat_template` raised before generation. The original preflight verified model,
processor, tokenizer, trainer metadata, sizes, and hashes, but failed to assert a usable template.

## Correction

All Attempt02 jobs now receive one explicit immutable template file:

`/home/wjy/pathvlm_r1_v1_formal/models/Qwen2.5-VL-7B-Instruct-cc594898137f460bfe9f0759e9844b3ce807cfb5/chat_template.json`

SHA-256:

`ad60d90252ed0b0705ba14e2d0ad0fec0beac1ea955642b54059b36052d8bc96`

The inference program loads its `chat_template` string and assigns it to every processor, including
base and snapshots that already carry the same semantic template. Each metrics file and job manifest
must record the exact template path and hash, and offline audit rejects a mismatch.

Seven isolated curve regression tests still pass. A real processor smoke using the actual missing-
template n=2000 epoch-6 snapshot confirms that it initially has no template and successfully builds
the frozen multimodal prompt after explicit injection. No validation answer was generated in this
smoke.

## Retry rule

Attempt02 must start from a new curve directory and rerun all 21 jobs. It may not recover or reuse the
seven completed Attempt01 jobs. Before launch it must repeat code hashes, all 20 snapshot hashes,
validation-data hash, parser tests, explicit template hash, and the eight-GPU idle gate. Test remains
untouched.
