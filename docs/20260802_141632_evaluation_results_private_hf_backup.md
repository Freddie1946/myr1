# Evaluation-results private Hugging Face backup

Timestamp: `2026-08-02 14:16:32 CST`

## Outcome

The completed/static evaluation artifacts on the A100 machine now have an off-machine backup in
the private Hugging Face dataset repository
`Freddie1946/PathVLM-R1-Revision-Evaluation-Results`.

The frozen snapshot is rooted at `snapshots/20260802_initial_static/` and contains 241 files with
an exact total size of 338,397,446 bytes.  The verified repository revision is
`8e4348a96eb7164de0b39eb5d5a7018203674fc8` and the repository API reports `private=true`.

The aggregate SHA-256 below is computed over sorted rows of
`<run-root>\t<relative-path>\t<size>\t<file-sha256>\n` for all 241 included files:

`c6f5779c7256c47e506f2d5c6deb4c6aab52fa7cabbdea799dfe7a71d9f1a867`

## Included run roots

- `external_vqa_full_20260729_224503`: 66 files, 177,399,822 bytes.
- `hosted_baseline_behavioral_smoke_20260801`: 25 files, 118,564 bytes.
- `hosted_baseline_full_20260801`: 47 static files, 141,427,679 bytes.
- `pathmmu_diagnostic`: 85 files, 14,810,633 bytes.
- `pathmmu_sft4000_stage2_20260729_221640`: 18 files, 4,640,748 bytes.

These directories preserve raw predictions, scoring outputs, summaries, logs and bad-case
artifacts already present in the run trees.  They are evaluation/development evidence, not model
weights and not an untouched-test claim.

## Deliberate exclusions and incremental policy

The two active PathVQA GPT-5-mini semantic-judge passes were still appending records at snapshot
time.  Their `gpt-5-mini_semantic_judgments*.jsonl` and `judge_retry_20260802.log` files were
therefore removed from the current static snapshot.  They will be uploaded in a new immutable
increment after each judge pass finishes and is validated.  Six accidentally included mutable
files were removed in HF commit `8e4348a96eb7164de0b39eb5d5a7018203674fc8`; earlier HF history
remains recoverable.

GitHub remains the authoritative home for source, small manifests, hashes, summaries and recovery
instructions.  Raw generated evaluation outputs remain outside Git history and are backed up to
the private HF dataset instead.

## Model/checkpoint backup status

The following private HF model backups were independently resolved through the HF API:

- SFT3000: `Freddie1946/PathVLM-R1-SFT-n3000-seed42-epoch3`, revision
  `97aae0969edd2824191f7bf15cd67835c1794ecf`.
- Stage2: `Freddie1946/PathVLM-R1-Outcome-GRPO-n1000-seed42-epoch2`, revision
  `6496331a597246bda84e8945a3f554c92a3ccc84`.
- SFT4000 control: `Freddie1946/PathVLM-R1-SFT-n4000-control-seed42-epoch2`, revision
  `31ecd18b9dc9de5c4118efde586bb625e4a6da06`.

The in-progress GPT-4o Stage3 arm is not yet mirrored to HF.  At this timestamp, its model-only
step-500 epoch snapshot is about 16 GiB and its rolling full recovery checkpoints at steps 700 and
800 are about 102 GiB each on the formal-machine storage.  Uploading those while the eight-GPU run
is active would add avoidable storage/network contention.  The retained policy is to upload a
validated model-only formal snapshot plus its manifest after the next formal milestone/final
completion; full optimizer checkpoints remain rolling local recovery artifacts unless separately
approved.

## Monitoring boundary

The user changed the human audit cadence to approximately once every 30 minutes.  No foreground
sleep loop is used.  The existing Stage3 supervisor continues immediate classified-failure
handling; periodic Codex audits check progress, bounded retry/fallback ledgers, checkpoint health,
disk headroom and completion sequencing.

