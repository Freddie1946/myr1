# Stage3 Kimi 2.6 pilot prepared, waiting for external gates

Timestamp: `2026-07-30T01:07:53+08:00`

Status: the user selected Kimi 2.6 for an initial Stage3 version. The 50-step pilot is prepared
but not started. No Kimi paid call, reward event or model update has occurred.

## Frozen pilot delta

The previously approved 50-step Stage3 pilot remains unchanged except for the Judge route:

- Judge model: `moonshotai/kimi-k2.6`;
- initial fixed provider candidate: Inceptron;
- image input and strict JSON Schema output;
- `provider.enforce_distillable_text=true`;
- Zero Data Retention required;
- provider data collection denied;
- provider fallback disabled;
- Judge output ceiling: 320 tokens through `max_tokens`;
- six events and deterministic penalty 0.4 unchanged;
- USD 15 client-side hard ceiling unchanged;
- one synthetic smoke plus at most 400 training requests;
- Stage2 epoch-2/step-1000 checkpoint, RL1000, seed 42, 50 steps, global batch 8,
  four generations, LR `1e-6`, beta `0.04` unchanged.

OpenRouter's read-only model filter returns Kimi 2.6 when image input, distillable output reuse and
ZDR are required together. The live route advertises structured output support. Current filtered
list pricing is approximately USD 0.646/M prompt tokens and USD 2.72/M completion tokens.

## Implementation

`scripts/stage3_openrouter_judge.py` now makes the model, frozen provider set, token parameter,
token ceiling and ZDR requirement part of the request and cache contract. It still fails closed
if distillation enforcement is disabled or if the served provider/model differs.

New entry points:

- `scripts/run_stage3_kimi26_smoke.py`: one synthetic, non-training paid compatibility call,
  sharing the production Judge ledger and writing a verified smoke marker;
- `scripts/launch_stage3_kimi26_pilot.sh`: validates the smoke marker, RL1000 content and image
  hashes, all eight idle GPUs, free disk, secret permissions and the master port before launching.

The launcher refuses to overwrite an existing output directory and saves checkpoints at steps 25
and 50.

## Verification

- system Python: 16/16 Judge tests passed;
- actual GRPO Python 3.10 environment: 16/16 Judge tests passed;
- Kimi payload dry check passed;
- Python compilation and Bash syntax checks passed.

Code SHA-256:

- `scripts/stage3_openrouter_judge.py`:
  `dd46867169b732c0a9fc95d961f6ffa02a75a0cdaac11074d7f809051da5c0c8`
- `scripts/test_stage3_openrouter_judge.py`:
  `dcc0486b547b492bb5106f8429bad449452c73918ef69a9786ae47bda04317d1`
- `scripts/run_stage3_kimi26_smoke.py`:
  `098029d99776a25837a8e012e607ee91169f1ba2018a66bef4493350018f5b26`
- `scripts/launch_stage3_kimi26_pilot.sh`:
  `a7ec56e839fd63b06aded6550c7b90b93794e8df652241b8667084f6886e4a4a`

## Remaining gates

1. Revoke the OpenRouter key exposed by the failed redaction and install a replacement in the
   existing mode-0600 secret file. The old key also appears in the untracked
   `scripts/keepintouch.txt` and must not be reused.
2. Run one Kimi 2.6 paid synthetic smoke. If Inceptron cannot satisfy the frozen ZDR/structured
   output/distillable contract, select and document another qualifying Kimi provider before any
   training.
3. Wait for GPU 0's authorized InternVL OmniMedVQA evaluation to finish. At this timestamp it had
   written 4,641/8,518 predictions; GPUs 1--7 were idle.
4. Launch only after all eight GPUs are idle and the smoke marker exists.

The existing evaluation process must not be interrupted to accelerate Stage3.
