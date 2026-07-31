# Kimi 2.6 Stage3 attempt 01 failure and attempt 02 restart

## Outcome

The first formal Kimi 2.6 Stage3 process-GRPO run did **not** complete.  It
stopped at optimizer step 274/1500 (epoch 0.55) at `2026-07-31 02:12:22
Asia/Shanghai`.  Rank 3 received 176 valid response-body bytes but the upstream
HTTP server omitted the terminal chunk, causing `http.client.IncompleteRead`.
The distributed launcher then terminated all ranks.  There was no CUDA OOM,
NaN, gradient failure or model failure.

The final recorded step was healthy: accuracy reward `1.0`, format reward
`1.0`, process reward `0.9125`, total reward `2.9125`, gradient norm `4.0313`,
and KL `0.006195`.

Attempt 01 has no resumable checkpoint because its first scheduled save was at
step 500.  It therefore cannot truthfully be resumed from step 274.

## Budget settlement

Attempt 01 committed USD `2.63313963`.  Three requests were unresolved when the
distributed run stopped.  In the absence of billing evidence, all three were
conservatively settled at their full USD `0.05` reservation.  Attempt 01 is
therefore closed at USD `2.78313963`, with zero unresolved reservations and no
budget breach.

The user's previously approved overall ceiling remains USD `30.00`.  Attempt
02 has a separate client-side hard cap of USD `27.21686037`; the closed attempt
01 amount plus the attempt 02 ceiling is exactly USD `30.00`.

## Engineering-only correction

The response reader now accepts an `IncompleteRead.partial` body only when the
partial bytes already form a complete, strictly parseable JSON document.  A
genuinely truncated JSON document still fails closed and retains its budget
reservation.  Eighteen judge/ledger/scoring tests pass.

The launcher now accepts `PATHVLM_STAGE3_SAVE_STEPS`, defaulting to the original
500.  Attempt 02 uses 100 so a future transport interruption loses at most 99
steps.  The value must divide 500, preserving the frozen epoch model snapshots
at 500, 1000 and 1500.  No optimization, reward, data, seed or model
hyperparameter changed.

## Attempt 02

- Run root: `/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/stage3_process_grpo/kimi26_full3epoch_seed42_attempt02_20260731`
- Parent: fixed Stage2 Outcome-GRPO epoch 2, step 1000
- Judge: `moonshotai/kimi-k2.6`, provider-only `inceptron`
- Paid contract smoke: passed, cost USD `0.00072115`
- Static preflight: passed
- Training start: `2026-07-31 21:49 Asia/Shanghai`
- First optimizer step: completed with total reward `2.7125`, gradient norm
  `5.0491`, and KL `0.0`
- Full target: 1500 steps; this document is a live-start record, not a
  completion claim

The original failed run and its complete judge/reward audit files remain
preserved.  Attempt 02 writes to a new root and does not reuse attempt 01's
judge cache.

