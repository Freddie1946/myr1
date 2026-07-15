# Fail-closed formal SFT scale supervisor

Timestamp: `2026-07-15 23:50:00 Asia/Shanghai`

## Scope

The user authorized a persistent supervisor after warning that an incorrect automatic transition
could corrupt a sequence of expensive experiments. The supervisor is therefore deliberately
fail-closed. It attaches to the already running formal n=500 seed-42 SFT run and may perform only:

1. validate the completed n=500 SFT manifest and checkpoint gates;
2. run deterministic frozen `validation_0385` for n=500;
3. sequentially run n=1000, n=2000, and n=3000 seed-42 SFT;
4. run the same deterministic validation after each successful SFT;
5. stop after n=3000 validation.

It cannot launch seed 43/44, Outcome GRPO, Stage 3, test inference, alternative data, or an alternative
GPU topology.

## Transition gates

Every SFT-to-validation transition independently requires a completed `stage1_sft` manifest with
`formal_result: true`, exact data version/count/seed, exact base revision, eight physical GPUs,
full-language training, frozen vision/projector, all required gates true, final checkpoint inside its
own run directory, and required checkpoint metadata files present.

Every validation-to-next-SFT transition independently requires a completed `stage1_validation`
manifest with `formal_result: true`, exact parent path and SHA-256, exact parent checkpoint, frozen
385-QA validation data, deterministic 192-token decoding, all gates true, test-access flags false,
metrics and raw predictions inside the validation run, and exactly 385 non-empty prediction records.

Before every new process, the supervisor re-verifies the core code manifest, its own sequence code
manifest, a clean and unchanged Git commit, at least 500 GiB free disk, the exact eight-GPU inventory,
and zero compute processes. If GPUs are occupied, it waits and records their PIDs; it never terminates
them or changes topology.

## Duplicate and recovery safety

- An exclusive filesystem lock permits only one supervisor.
- Exactly one new manifest must appear after a launch.
- Multiple existing/new manifests stop the sequence.
- Failed, failed-gate, aborted, cancelled, or unknown status stops the sequence.
- Failed jobs are never retried automatically.
- A launch intent is fsync-persisted before process creation. If the supervisor is interrupted before
  it can recover exactly one manifest, the unresolved intent forbids an automatic retry.
- Supervisor-launched child processes use independent sessions so an interrupted supervisor does not
  kill an otherwise valid training process. A restarted supervisor may attach only to a unique
  recorded/existing manifest.
- State changes use atomic replace after fsync.

## Persistent observability

Machine-local files are stored outside Git and model outputs:

- state: `/home/wjy/pathvlm_r1_v1_formal/control/sft_scale_seed0042/sequence_state.json`;
- event log: `/home/wjy/pathvlm_r1_v1_formal/control/sft_scale_seed0042/supervisor.jsonl`;
- daemon log: `/home/wjy/pathvlm_r1_v1_formal/control/sft_scale_seed0042/daemon.log`;
- per-command logs under `command_logs/`;
- exclusive lock: `supervisor.lock`.

The state includes current phase/count/manifest, training progress heartbeat, child PID, validation
metrics, every transition event, expected Git commit, both code manifests, and `test_accessed: false`.
It provides durable monitoring and automatic continuation but does not claim to deliver an external
chat notification.

## Verification performed before activation

The Python and Bash launchers pass syntax checks. Twelve isolated regression tests pass with warnings
treated as errors. They cover valid SFT/validation manifests, false SFT gates, test access, wrong GPU
topology, checkpoint path escape, validation parent-hash mismatch, validation test access, 384 rather
than 385 predictions, exclusive-lock rejection, atomic state writes, and unresolved SFT/validation
launch intents refusing retry. The wrapper also rejects a current manifest outside the exact n=500
formal run layout.

The supervisor must not be activated until this document and its sequence hash manifest are committed,
the repository is clean, the current training process is rechecked, and the fixed daemon launch
receives explicit system approval.
