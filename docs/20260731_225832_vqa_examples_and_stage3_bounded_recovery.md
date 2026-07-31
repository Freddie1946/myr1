# VQA concrete cases and bounded Stage3 recovery

Recorded at: 2026-07-31 22:58:32 CST

## Concrete external-VQA cases

The common external-VQA run root is:

`/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/external_vqa_full_20260729_224503`

PathVQA is the dataset in this pair that actually contains open-ended answers.  A
representative Stage2 row is index 1:

- question: `how are the histone subunits charged?`
- reference: `positively charged`
- answer type: `free_form`
- completion: `Each histone subunit is positively charged.`

This is semantically correct, although literal whole-string exact match is false.  It is a
concrete reason not to use raw completion-versus-reference exact match as the sole free-form
metric.

A contrasting PathVQA row is index 25:

- question: `what is aids?`
- reference: `acquired immunodeficiency syndrome`
- answer type: `free_form`
- completion: `Aids is a condition related to the immune system, characterized by the presence
  of CD4-positive lymphocytes. It is not a condition associated with high blood pressure, as
  mentioned in the description. Therefore, the answer is incorrect.`

This completion never supplies the requested expansion and should not be rescued merely because
it contains related medical words.

OmniMedVQA is not open-ended in the frozen 8,518-row evaluation set; each row is four-option
multiple choice.  A representative Stage2 row is index 4:

- question: `What imaging modality was employed to obtain this image?`
- options: A ECG; B nuclear medicine scan; C ultrasound; D CT
- target: D (`CT`)
- completion ends with: `<answer>D) CT</answer>`

The original whole-response SequenceMatcher selected A because it compared the entire reasoning
string with short option strings.  Target-blind final-answer extraction selects D.  This is an
evaluation-parser failure in the original report, not a model error.

OmniMedVQA index 17 illustrates the generation-cap edge case:

- target: A (`Adenocarcinoma of the left lower lobe, T2 N0 M0, Stage Ib`)
- completion ends with: `<answer>A) Adenoc`
- generated tokens: 64; EOS: false; generation cap reached: true

The target-blind parser can recover the explicitly emitted A, while the original whole-response
similarity selected C.  The truncation flag must still be retained for audit.

## Stage3 save and failure contract

- Save a full DeepSpeed checkpoint every 100 optimizer steps.
- Keep only the latest two full checkpoints (`save_total_limit=2`).
- Preserve model-only epoch snapshots at steps 500, 1000 and 1500.
- Retry only explicit transient router statuses 408, 429, 500, 502, 503 and 529, with bounded
  delays of 15, 45 and 90 seconds.
- Accept an `IncompleteRead` body only if its received bytes parse as one complete JSON object.
- Do not resend a connection-level failure or an HTTP-200 truncated/invalid JSON response in the
  same process: its billing state is ambiguous.
- On a failed training segment, conservatively settle every unresolved reservation at its full
  reserved amount, then resume from the newest structurally complete 100-step checkpoint.
- If failure occurs before the first checkpoint, preserve the partial output and restart from the
  frozen Stage2 parent while retaining successful judge cache records.
- Allow at most three recoveries (four launches total).  Repeated deterministic failure therefore
  stops instead of entering an infinite restart loop.
- Never fabricate a judge event or substitute a default process reward.

The existing pilot checkpoint at step 50 passed the new structural validator: four safetensor
shards, eight optimizer-state files, eight model-state files and eight RNG states.  CUDA-free
regression coverage contains 25 passing tests.  No paid smoke or formal Stage3 restart was made as
part of this change.

Each full pilot checkpoint occupies approximately 102 GiB.  With two retained full checkpoints,
the steady full-checkpoint footprint is approximately 204 GiB; the filesystem currently has
approximately 5.4 TiB available.  The 100-step policy is therefore safe for capacity and halves
the checkpoint frequency relative to 50-step saving.

## AIGCBest key gate

The local AIGCBest secret file has mode 0600.  An authenticated model-catalog request returned
HTTP 200 and 1,342 model identifiers.  This was a read-only authentication check; no paid model
request and no evaluation image were sent.
