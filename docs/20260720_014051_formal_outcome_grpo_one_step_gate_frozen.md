# Formal Outcome-GRPO one-step gate frozen

Timestamp: 2026-07-20 01:40:51 Asia/Shanghai

## Scope and authorization boundary

The user authorized continuation after deciding not to rerun the n=500/n=1000 SFT jobs. This change
prepares exactly one Outcome-GRPO optimizer-step engineering gate. It does not authorize a formal
long RL run, an automatic retry, a reward/prompt/seed change, Stage 3, test inference, or deletion of
the resulting gate checkpoint.

The gate is always labelled `formal_result: false`. Even if it passes, a later formal long run must
restart from the selected SFT parent, not continue from the one-step gate output.

## Frozen parent and data

- Parent lineage: formal seed-42 SFT n=3000.
- Validation selection: epoch 3, global step 1125, validation accuracy 60.52%.
- Parent snapshot manifest SHA-256:
  `6c8708dd677f5ab1563e77317d86cd1b53cb9b969ebda09e4a312581a2b9f1f1`.
- Validation-curve manifest SHA-256:
  `e7170be9ddd3050b8421a02c3a12a3e59d9762b6296a26f246abe445ea06ff24`.
- Data: exact frozen v2 `pathvlm_rl_smoke_n0008`, eight QA from the RL split only.
- Validation and corrected 999-QA test are not training inputs; `picked.json` remains forbidden.

The selected snapshot directory does not contain `Qwen2.5-VL` in its pathname, while the vendored
trainer dispatches model classes using pathname substrings. The runner therefore creates a run-local
symlink named `parent_Qwen2.5-VL-7B-Instruct` and verifies that its real path is the exact selected
snapshot. No parent file is changed.

## Frozen one-step settings

- GPUs 0–7, only if all eight have no compute process at the launch check.
- Seed/data seed 42; maximum one optimizer step.
- Eight-device global batch, per-device batch 1, four generations per prompt.
- Accuracy parser v2 plus format reward; beta 0.04; learning rate 1e-6.
- Maximum completion length 192; SDPA; bf16; gradient checkpointing.
- Full language-model parameters trainable; vision tower and projector frozen.
- DeepSpeed ZeRO-3 with CPU optimizer offload.
- Offline mode and proxy variables removed from the child environment; no download is allowed.
- Preserve at least 550 GiB free disk.

## Fail-closed audit

The wrapper now writes one JSONL per distributed rank. Every record includes rank, local rank, PID,
reward type, call/item indices, completion, solution, and reward. Per-rank files avoid concurrent
JSONL corruption. The post-run auditor requires exactly eight accuracy and eight format events,
pairs them by rank/call/item, recomputes both rewards offline with the same parser, reconstructs the
two four-generation reward groups, and requires positive group reward variance.

The run also must show exactly one optimizer step, finite loss, finite nonzero gradient norm,
positive trainer reward standard deviation, fully trainable language parameters, frozen visual and
projector parameters, a loadable saved Qwen2.5-VL checkpoint, a changed representative language
tensor, and an exactly equal representative visual tensor. Any failed gate preserves the run and
stops without automatic retry.

## Static verification completed before launch

- Parser v2 regression: pass.
- CUDA-independent distributed reward-log/trainability regression: pass.
- Synthetic post-run reward/state auditor regression: pass.
- GRPO environment `pip check`: pass.
- Selected 17-file parent snapshot: all sizes and SHA-256 values pass.
- v2 smoke adapter: exact eight frozen RL rows; all image paths exist.
- v2 basename and exact-content primary-split overlaps: all zero.
- Current free disk at preparation: about 930 GiB; available RAM about 235 GiB.

The exact protocol is frozen in
`protocol/outcome_grpo_gate_manifest_20260720_013054.json`; executable source hashes are frozen in
`protocol/outcome_grpo_gate_code_manifest_20260720_013054.json`.

## Reviewer mapping

- R2-4: strengthens exact online/offline outcome-reward execution and raw event provenance; it does
  not answer the separate GPT-4o/Stage-3 judge questions.
- R3-3: establishes a valid engineering prerequisite for an outcome-only RL arm; no process-reward
  ablation result is claimed.
- R3-5: prerequisite for the later frozen RL 250/500/1000 scale runs; no RL-scale result exists yet.
- R3-6: preserves seed and raw per-completion evidence, but this one-step engineering gate is not a
  multi-seed scientific result.
