# Stage3 post-training validation pipeline prepared

Recorded at: `2026-08-02 14:28:34 CST`

## Outcome

The formal GPT-4o Stage3 supervisor correctly owns bounded training recovery, but it intentionally
does not run the post-training scientific selection.  A fail-closed post-training pipeline is now
prepared so completion cannot be confused with checkpoint selection.

No model file was read, no GPU inference ran, no paid request was made and no test record was
accessed while preparing this pipeline.  The active GPT-4o arm was not modified or restarted.

## Snapshot integrity verifier

`scripts/verify_checkpoint_snapshot.py` verifies the model-only snapshot manifest and every
declared file by exact size and SHA-256.  It additionally requires:

- `model_only=true` and `resumable=false`;
- exact expected optimizer step and epoch when supplied;
- no undeclared or missing top-level file;
- a nonempty model weight map whose shards all exist in the manifest;
- agreement of `trainer_state.json` step/epoch with the snapshot manifest;
- presence of the model index, configuration and fixed chat template.

This restores the previously referenced checkpoint-snapshot verifier on the A100 repository and
also applies it to the new Stage3 epoch snapshots.

## Three-epoch validation and selection

`scripts/run_stage3_formal_validation.sh` accepts only a completed run below the formal Stage3 run
root.  It verifies and then evaluates the model-only snapshots at steps 500, 1000 and 1500 using
the existing eight-way deterministic validation385 runner.  Existing output is never overwritten
or silently reused.

`scripts/select_stage3_validation_checkpoint.py` validates all three metrics/prediction pairs,
including the fixed validation-data hash, count 385, `test_accessed=false`, exact model identity
and prediction SHA-256.  It then applies the frozen tie-break in this exact order:

1. higher validation correct count;
2. higher validation strict-format correct count;
3. earlier epoch.

The selector writes an explicit `selection.json`; validation does not access PathMMU test999.

## Verification

- Bash syntax checks passed for the new wrapper and inherited validation runner.
- Python compilation passed.
- Seven new unit tests passed, covering valid/tampered/unexpected/mismatched snapshots and all
  selection/tie/test-access gates.
- `git diff --check` passed.

Code SHA-256:

- `scripts/verify_checkpoint_snapshot.py`:
  `70c88474f8b8b9ccc5308bebea38cb9b3e1425e4f1302ee7ad3d200fe4467271`;
- `scripts/select_stage3_validation_checkpoint.py`:
  `ac6f81c4ccf4b3316230b3b2917931536dc2c38dfc31398e0b987c3c6330793c`;
- `scripts/run_stage3_formal_validation.sh`:
  `f624977737711a83d4beba4cf5a035f0e479171ecdb8cafd8362ed99d5175c4c`.

