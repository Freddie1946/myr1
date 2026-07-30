# Kimi 2.6 Stage3 50-step pilot completed

Completed: `2026-07-30T10:21:42+08:00`

Status: engineering pilot completed; `formal_result: false`.

## Outcome

The seed-42 Stage3 pilot completed 50 optimizer steps on the frozen image-disjoint RL1000 split.
The final process used eight A100 GPUs, full language-model training, a frozen vision tower and
multimodal projector, and the unchanged accuracy/format/process reward aggregation. The successful
process exited with status 0 and saved both a full DeepSpeed checkpoint and a gathered final model.

Final output:

`/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/stage3_process_grpo/kimi26_50step_seed42_20260730/output_attempt03`

Recoverable checkpoint:

`/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/stage3_process_grpo/kimi26_50step_seed42_20260730/output_attempt03/checkpoint-50`

Checkpoint-50 contains four model safetensor shards, eight ZeRO optimizer-state shards, eight model
state shards, scheduler, RNG states and trainer state. The model index maps exactly 729 tensors and
all 729 tensor headers are present.

## Final-trajectory training audit

The final model trajectory consists of steps 1-25 from attempt03 and steps 26-50 from the successful
attempt07 recovery. Exactly 1,200 final-trajectory reward events were retained:

| reward | events | minimum | maximum | mean |
|---|---:|---:|---:|---:|
| accuracy | 400 | 0.0 | 1.0 | 0.685 |
| strict format | 400 | 1.0 | 1.0 | 1.000 |
| process | 400 | 0.0 | 1.0 | 0.646 |

There are 400 completions over 100 unique source records, as expected from four generations per
prompt. All frozen image SHA-256 fields are present.

The final trainer audit contains 50 finite metric rows:

- gradient norm: mean `4.8851`, range `3.4789` to `6.0561`;
- KL: mean `0.0001551`, maximum `0.0009766`;
- completion length: mean `81.1625`, range `67.5` to `96.5`;
- total reward: mean `2.3310`, range `1.4250` to `2.9750`;
- train loss: `4.1740069e-06`;
- final learning rate: `0`;
- trainability gate: language `7,615,616,512 / 7,615,616,512` trainable; visual
  `0 / 676,550,144` trainable; projector `0 / 44,574,464` trainable.

## Judge and recovery disclosure

The approved GPT-4o Judge could not be used through the available OpenRouter route. At the user's
direction the pilot used `moonshotai/kimi-k2.6`, provider `Inceptron`, with fallback disabled,
ZDR required, data collection denied, distillable-text enforcement enabled and reasoning disabled.
The six booleans and the local penalty coefficient `0.4` never changed.

This engineering pilot is not a single-contract formal Stage3 result. Steps 1-25 used a 640-token
Judge output cap and a strict 160-character evidence cap. A real response hit a structured-decoding
dead end at step 33. Recovery retained the 160-character target but used a 512-character absolute
evidence ceiling and a 1,024-token Judge output cap for steps 26-50. The exact previously failing
sample passed the revised contract before recovery. This response-cap change does not alter the
six-event scoring formula, but it can alter Judge outputs and must be disclosed. A later formal run
should restart from the frozen parent with one fixed Judge contract from step 1.

The first checkpoint recovery also exposed PyTorch 2.6 versus DeepSpeed 0.15.4 deserialization
behavior. `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` was scoped to recovery of the locally generated,
trusted ZeRO checkpoint. Inceptron's shared upstream pool later returned HTTP 429 under eight
simultaneous starts. The successful recovery globally staggered request starts by four seconds and
used bounded `15,45,90`-second retry delays. Attempt07 recorded no HTTP 429.

## API accounting

The fail-closed ledger ended with:

- hard ceiling: USD `15`;
- ledger entries: `512`;
- committed conservative spend: USD `0.88919683`;
- unresolved reservations: `0`;
- budget breached: `false`;
- six interrupted/ambiguous reservations conservatively settled at their full USD `0.05`
  reservation rather than released.

The OpenRouter key endpoint and account-credit delta both report actual key usage of
USD `0.59393869`. The local ledger therefore intentionally overstates actual spend by
USD `0.29525814`.

## Frozen validation385 post-check

The final model was evaluated on the frozen 385-item validation split only, using the historical
BF16, unquantized, greedy, 1,024-token-cap runner, unchanged chat template and unchanged parser.
Eight GPUs processed contiguous shards with per-GPU batch size 1; outputs were merged in original
index order and rescored offline. The first 19 predictions from an earlier single-GPU attempt match
the parallel outputs exactly in completion text, token count, extracted choice and rewards.

| model | correct | accuracy | strict format | choice extracted | cap hits |
|---|---:|---:|---:|---:|---:|
| Stage2 parent (historical same protocol) | 241/385 | 62.60% | 100.00% | 385/385 | 0 |
| Stage3 Kimi pilot, step 50 | 243/385 | 63.12% | 100.00% | 385/385 | 0 |

The pilot point estimate is `+2/385`, or `+0.52` percentage points. This is not evidence of a
material improvement by itself. The full historical Stage2 validation predictions are not present
on this machine, so a paired transition table or McNemar test was not fabricated. PathMMU test999
and external test sets were not accessed for this post-pilot selection.

Validation artifacts:

`/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/stage3_process_grpo/kimi26_50step_seed42_20260730/validation_0385_postpilot_parallel_20260730_101722`

## Key hashes

- checkpoint-50 model index:
  `3067e9b0f35596ff3426a0d0ec8c982a51fa1e110c4fc30dcf3be9ea37409df6`
- checkpoint-50 trainer state:
  `1949fd7ef18cf97953b04c15f0182b141f9f65c8e07db981ed29591cdc24f779`
- final train-state audit:
  `7d9bdf1afe77d75ff103fbaacdd1dd6d812b9efd0087d87c7b959506e4690ed5`
- validation predictions:
  `146fb4a9c788accda041ed333c66f44f963a7e91682b54a0e303ce8877c70d13`
- validation metrics:
  `e8194c732c1bb869888c73c94b7894ee1fc073ebd0b06a1f90a379879a458f65`
- final conservative Judge ledger:
  `b8d69f580ab0d807286c2f60625b60e2718b81a060a9e6443ead24602624e3a2`
