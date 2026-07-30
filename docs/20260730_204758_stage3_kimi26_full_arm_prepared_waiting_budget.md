# Full Kimi 2.6 Stage3 arm prepared, waiting for separate budget approval

Timestamp: `2026-07-30T20:47:58+08:00`

Status: prepared but not started. No new paid smoke, training request, optimizer step or model
mutation has occurred for this arm. `formal_result: false`.

## Scope

The user requested a complete Kimi 2.6 Stage3 validation arm and stated that later Stage3 arms may
use GPT or other Judge models. This arm is a contemporary model-specific reconstruction and must
not be described as recovery of the historical Stage3 Judge.

The arm restarts from the frozen Stage2 parent rather than resuming the mixed-contract 50-step
engineering pilot:

`/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/transferred_checkpoints/outcome_grpo_n1000_seed42_epoch02_step1000`

## Frozen training contract

- data: frozen image-disjoint RL1000;
- seed/data seed: `42`;
- full language model trainable; vision tower and multimodal projector frozen;
- three epochs represented as exactly `1,500` optimizer steps;
- global completion batch: `8`; generations per source prompt: `4`;
- maximum online Judge requests: `12,000`, plus one synthetic contract smoke;
- maximum completion length: `192`;
- learning rate: `1e-6`;
- beta: `0.04`;
- reward order: accuracy, strict format, process;
- six process events and deterministic coefficient `0.4` unchanged;
- full checkpoints every 500 steps with a rolling limit of two;
- model-only epoch snapshots retained at steps 500, 1000 and 1500.

## Frozen Judge contract

- model: `moonshotai/kimi-k2.6`;
- provider: `Inceptron`;
- provider fallback disabled;
- ZDR required and provider data collection denied;
- distillable-text enforcement enabled;
- reasoning disabled;
- structured JSON Schema output;
- `max_tokens=1024`;
- evidence target 160 characters and absolute ceiling 512 characters;
- minimum global request-start interval: four seconds;
- bounded retry delays: 15, 45 and 90 seconds;
- new cache, audit and budget ledger isolated from the 50-step pilot;
- fail closed on route, schema, accounting, source-hash or request-limit mismatch.

## Cost gate

The 501 successful cached pilot calls cost USD `0.58212086`, a mean of USD
`0.001161917884` per call. Mean-cost projection for 12,000 calls is USD `13.9430`; projecting every
call at the pilot P95 cost gives USD `24.4945`.

A USD `30` client-side ceiling is recommended to cover cost variance and bounded retries. It is a
hard stop, not a spending target. The user's earlier USD 15 approval applied to the 50-step pilot
and is not silently reused. The exact new ceiling must be explicitly approved before the new paid
smoke or training launch.

## Prepared entry points

- `scripts/run_stage3_kimi26_formal_smoke.py`
- `scripts/launch_stage3_kimi26_formal.sh`

Static verification completed:

- Bash syntax passed;
- Python compilation passed;
- existing Judge regression tests passed: `16/16`;
- no API secret is embedded in either new entry point.

Code SHA-256:

- formal smoke:
  `0f6f300ac689674a83412a3c9302b0f976a717cb6816b676448d923717d7cdd5`;
- formal launcher:
  `e76941082e3e71a924cc1d0b457c1287a5a942150eb2bd1c1431ec40fdd88bd9`;
- Judge implementation:
  `045e9422adbc0724d455d1c87f8df5fab517c6f01b53fdf71891fa196a6b54c1`;
- audited trainer wrapper:
  `3c20993b29420ff061246935422a4a44d3959fcb3bd34b8f62d79bc4702741ee`.

## Remaining gate

Explicit approval of the separate full-arm budget ceiling. After approval, run the exact-contract
synthetic smoke and launch from the frozen parent only if all eight GPUs, disk, port, source hashes,
secret permissions and budget ledger pass.
