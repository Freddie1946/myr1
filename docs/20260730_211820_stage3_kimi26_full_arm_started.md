# Full Kimi 2.6 Stage3 arm started

Timestamp: `2026-07-30T21:18:20+08:00`

Status: running. `formal_result: false`.

The user authorized starting the complete Kimi Stage3 validation arm after the separately prepared
USD 30 hard ceiling was presented. This arm is a contemporary Judge-specific validation, not a
claim that the historical Stage3 Judge was recovered.

## Frozen execution

- parent: selected Stage2 Outcome-GRPO epoch-2/step-1000;
- data: frozen image-disjoint RL1000;
- seed/data seed: 42;
- three epochs represented as exactly 1,500 optimizer steps;
- eight A100 GPUs, global completion batch 8 and four generations per source prompt;
- full language-model training with frozen vision tower and multimodal projector;
- learning rate `1e-6`, beta `0.04`, maximum completion length 192;
- reward order: accuracy, strict format, process;
- full checkpoints every 500 steps and model-only epoch snapshots at steps 500/1000/1500.

The run started from the frozen Stage2 parent. It did not resume the mixed-contract 50-step pilot.

## Judge and budget

- model: `moonshotai/kimi-k2.6`;
- provider: Inceptron only, fallback disabled;
- structured six-event output and local coefficient 0.4;
- `max_tokens=1024`;
- evidence target 160 characters, absolute ceiling 512;
- ZDR required, data collection denied and distillable-text enforcement enabled;
- reasoning disabled;
- global request starts separated by at least four seconds;
- retries bounded to 15/45/90 seconds;
- maximum requests: 12,001, consisting of one synthetic smoke plus at most 12,000 training calls;
- client-side hard ceiling: USD 30.

The exact-contract synthetic smoke passed before training:

- generation ID: `gen-1785417110-jhJKpFOYmYIWDwy6icKr`;
- served provider: Inceptron;
- cost: USD `0.00072115`;
- process score: `0.8`;
- unresolved reservations after smoke: zero.

## Startup evidence

All source hashes, disk, port, secret-permission, route, cache and GPU-idle gates passed. DeepSpeed
loaded the frozen parent on all eight ranks. The trainability audit passed:

- language trainable: `7,615,616,512 / 7,615,616,512`;
- vision trainable: `0 / 676,550,144`;
- projector trainable: `0 / 44,574,464`.

The first optimizer step completed with finite nonzero gradients:

- gradient norm: `5.103901386260986`;
- completion length: `72.125`;
- accuracy reward: `1.0`;
- strict-format reward: `1.0`;
- process reward: `0.7125000953674316`;
- total reward: `2.7125000953674316`;
- KL: `0.0`.

At this record timestamp the process was still healthy on all eight GPUs. The budget ledger had
committed USD `0.03597462` across 32 completed unique requests, had one in-flight reservation and
had not breached the USD 30 ceiling.

Run root:

`/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/stage3_process_grpo/kimi26_full3epoch_seed42_20260730`

Final accuracy, checkpoint integrity and formal completion status must not be inferred from this
startup record. They will be recorded only after termination and post-run validation.
