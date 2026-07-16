# Two-tier SFT checkpoint retention implemented

Timestamp: `2026-07-17T00:23:23+08:00`

## Authorization and scope

The user approved implementation and execution of the storage plan on 2026-07-17. This change does
not itself authorize Stage 2 or Stage 3. Test data remains forbidden.

## Retention policy

Formal SFT now saves a complete resumable checkpoint at every epoch boundary while Transformers
retains the latest two. A concurrent audited archiver hard-links the gathered model files from every
completed checkpoint into an independent `epoch_snapshots/checkpoint-N` directory before the source
checkpoint can be rotated.

Each model-only snapshot contains its gathered safetensors, model/processor/tokenizer configuration,
trainer state, and a `snapshot_manifest.json` with file sizes and SHA-256 hashes. It excludes
DeepSpeed optimizer shards and is explicitly marked `resumable: false`. Hard links avoid copying the
model data while the source checkpoint exists; when Trainer removes an old full checkpoint, the
scientific snapshot remains allocated and independently loadable.

The formal policy is:

- ten model-only epoch snapshots, never automatically scientifically pruned;
- the latest two complete resume checkpoints;
- the final gathered model;
- a 550 GiB free-space reserve; and
- launch budgeting for three transient full checkpoints because Trainer writes the new checkpoint
  before rotating the oldest retained checkpoint.

## Training-backend correction

Generated formal SFT configs now use the measured candidate backend:

- eight GPUs and global batch size 8;
- DeepSpeed ZeRO-2;
- no optimizer or parameter offload;
- torch fused AdamW;
- gradient checkpointing explicitly enabled with both the positive Transformers argument and the
  LLaMA-Factory `disable_gradient_checkpointing: false` control; and
- epoch-boundary checkpointing.

The formal runner requires log evidence that gradient checkpointing actually ran, exactly ten model
snapshots, the latest two full checkpoints, the disk reserve, trainability/freeze policy, final model
reload, finite loss/nonzero gradients, a changed language tensor, and an identical visual tensor.

## Tests completed before GPU execution

Fifteen unit/regression tests pass. The new tests verify atomic model-only archiving, SHA-256
validation, hard-link preservation after simulated source rotation, exclusion of optimizer state,
retention of every scientific snapshot, and refusal to archive incomplete checkpoints. Supervisor
gates were strengthened to require the new backend and all ten snapshot manifests.

## Next gates

1. render the new machine-resolved configs and verify their hashes;
2. run the formal 7B three-step save/resume/rotate smoke with three model-only snapshots and one
   retained full checkpoint to bound smoke storage;
3. run the correctly labeled GC throughput confirmation smoke; and
4. only if all gates pass and projected disk remains above the reserve, launch the fresh n=3000,
   seed-42 duration-sweep SFT.

Existing checkpoints and model weights are not deleted by this implementation.

