# Data-ratio remote backup and safe tail handoff started

Timestamp: 2026-08-09 18:28:42 Asia/Shanghai

## Backup correction

The data-ratio implementation and launch records were previously committed only locally. The
formal run outputs also had no independent remote copy. This gap is now being corrected without
stopping the active training child:

- GitHub branch `codex/a100-stage3-eval` is synchronized through commit
  `a25b36f70532f8e42ad897f2f72c0790996ab640`.
- Frozen prepared data, protocol/code snapshots, completed-task logs, training metrics and all
  completed rule-reward audit events were uploaded to the private dataset repository
  `Freddie1946/PathVLM-R1-Revision-Evaluation-Results` under
  `snapshots/20260809_data_ratio_ablation_live/`.
- The remotely verified dataset revision after those uploads is
  `e22b50bf497dd25b64689b5d30ae8bcd800a4cfa`; 86 files exist under the new prefix.
- A detached low-priority model backup worker is sequentially uploading the five completed
  model-only outputs to five private model repositories. Each upload includes loadable weights,
  processor/tokenizer metadata, training metrics and a content-hash manifest. Optimizer and RNG
  state are deliberately excluded; the two rolling full checkpoints remain local for recovery.

Model backup state:

`/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/reports/data_ratio_model_hf_backup_20260809/state.json`

## Safe optimization boundary

The active `sft0250_rl0750_rule_rl` child continues unchanged with its original command and all
eight workers. Only the old Python supervisor is paused. A detached handoff process waits for the
child to finish naturally, verifies step 1125, model structure, trainability and rule-reward audit,
then retires the old supervisor.

After that boundary, the handoff runs bounded 20-step engineering smokes on the fixed 0+4000
training data. The scientific contract remains unchanged: per-device batch 1, eight GPUs,
gradient accumulation 1, four generations, rule accuracy+format rewards, learning rate, seed,
image limits and formal step counts are fixed.

FlashAttention 2 is not installed and is explicitly excluded. The only compute candidates are:

1. ZeRO-3 with optimizer state on GPU and gradient checkpointing disabled;
2. the same GPU-resident ZeRO-3 with gradient checkpointing retained;
3. current optimizer-offload ZeRO-3 with gradient checkpointing disabled.

A candidate must finish with finite metrics, a passing trainability/freeze audit, complete
accuracy/format reward events, no process/Judge events and at least 3% higher measured step
throughput than the observed baseline. Otherwise the tail automatically reuses the current
optimizer-offload, gradient-checkpointed SDPA stack.

Checkpoint-only optimization is independent of candidate selection: Stage2 continued rule-RL
saves every 250 steps, the 50-step gate saves at step 50, and the 6,000-step stress arm saves every
500 steps. All retain the latest two full recovery checkpoints. Stage2 continues to use the same
training stack as Stage3; only its save cadence changes.

## Live identities

- Original formal supervisor: PID `2715867`, intentionally stopped in state `T` while its child
  continues.
- Active unchanged training child at handoff admission: PID `2872541`.
- Detached handoff supervisor: PID `2884951`.
- Detached low-priority model backup worker: PID `2884852`.
- Training progress at record time: step `171/1125`.
- Test accessed: `false`.

The throughput smoke and optimized tail are not claimed complete by this record. Fail-closed
fallback preserves the existing training stack if the optimization path is not demonstrably safe.
