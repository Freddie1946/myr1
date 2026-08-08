# Data-ratio rule-RL ablation formal sequence started

Timestamp: 2026-08-09 03:40:26 Asia/Shanghai

## Completion boundary

The preparation goal is complete. The sparse data-ratio ablation protocol, deterministic data,
recoverable sequential runner and real multi-task smoke were completed before this launch. The
formal queue is now running under a detached supervisor. This record verifies the launch and the
first real optimizer updates; it does not claim that the nine-task queue or any downstream
evaluation has completed.

## Frozen launch identity

- Repository commit: `8f5eb1d380117c39029da216ff4c6610a364607e`
- Protocol SHA-256:
  `941c959179c1d6a6ab75872ae5e75d2aeee34b415b519467f789ddbc20831a82`
- Prepared-data manifest SHA-256:
  `2761b64e10eec83ab67fd3e74973558996a6c1084bb5cc09630f2eeed3d6ae79`
- Formal preflight SHA-256:
  `ac214bb60420124235f5afcebfb0c01ecf45e6158c1f3be977cf6b69ed04892e`
- Test accessed: `false`

The formal preflight passed with the fixed base revision, exact Stage2 checkpoint, all eight idle
A100 80-GB GPUs and approximately 4.11 TB free disk before launch.

## Detached-process verification

- Supervisor PID: `2715867`
- Parent PID: `1`
- Session ID: `2715867`
- Formal state: `running`
- Current task: `sft0750_rl0250_sft`
- Attempt: `1`

The parent/session values verify that the supervisor is detached from the launching Codex shell.
The first task created its immutable task contract, resolved YAML and dedicated training log.

The training log then reported:

- `***** Running training *****`;
- 750 examples and 10 epochs;
- effective total batch size 8;
- 940 optimization steps;
- full-language training with the vision tower and projector frozen;
- multiple completed optimizer updates with finite loss and gradient norm.

At final admission verification, all eight GPUs were at 100% utilization and used approximately
39.2--39.4 GiB each. Therefore the first formal task entered real training rather than merely
starting a wrapper or loading weights.

## Recovery and handoff

The supervisor owns a state file and a process lock. It validates completed outputs before skipping
them, resumes partial tasks from the latest structurally complete checkpoint, and stops the queue
on a recorded failure instead of retrying indefinitely. Formal checkpoints are written every 100
steps with the latest two retained.

Handoff paths:

- State:
  `/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/data_ratio_rule_rl_ablation_v1/formal_sequence/state.json`
- Supervisor log:
  `/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/data_ratio_rule_rl_ablation_v1/formal_sequence/supervisor.log`
- PID file:
  `/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/data_ratio_rule_rl_ablation_v1/formal_sequence/supervisor.pid`
- First-task log:
  `/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/data_ratio_rule_rl_ablation_v1/formal_sequence/tasks/sft0750_rl0250_sft/train_attempt01.log`

Per the user's instruction, this goal ends after launch verification. Continuous monitoring,
completion auditing, checkpoint selection and evaluation are separate future goals.
