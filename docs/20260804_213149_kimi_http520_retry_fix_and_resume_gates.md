# Kimi Stage3 HTTP 520 retry fix and resume gates

Recorded at: `2026-08-04 21:31:49 CST`

## Outcome

The Kimi 2.6 Stage3 run stopped after optimizer step 255 because rank 2 received a generic
Cloudflare-style `HTTP 520` response. The response body was only `error code: 520`; it contained
no structured OpenRouter error, generation ID or served-provider identity. The failing image was
752x575 and 637,627 bytes, within the normal PathMMU request range. This was therefore classified
as a transient router/origin failure rather than a sample, model, GPU, storage or numerical error.

The Judge previously retried only statuses 408, 429, 500, 502, 503 and 529. HTTP 520 was treated
as terminal on its first occurrence, and the supervisor did not recognize its exception text.
The correction:

- adds HTTP 520 to the explicit transient-status whitelist;
- retains the frozen four-attempt schedule with 15/45/90-second delays;
- retains the total-24/consecutive-4 structural fallback limiter after retry exhaustion;
- classifies an exhausted `OpenRouter HTTP 520` as recoverable at the supervisor layer;
- audits only safe tracing headers (`cf-ray`, request IDs, generation ID and `retry-after`) while
  excluding cookies and credentials.

The correction does not change the Judge model/provider, six events, penalty 0.4, prompt, response
schema, Stage2 parent, data, optimizer, seed, checkpoint cadence or fallback reward.

## Verification

The OpenRouter Judge/checkpoint recovery suites passed 43/43 CUDA-free tests. The AIGCBest shared
transport suite passed 10/10 tests. A fresh synthetic paid smoke then passed against
`moonshotai/kimi-k2.6` served by `Inceptron` in one physical request:

- cost: `$0.00072797`;
- generation ID: `gen-1785850126-lYJqjWSzjtMujySl4Ltk`;
- process reward: `0.8`;
- smoke directory:
  `/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/stage3_process_grpo/kimi26_http520_fix_smoke_20260804`.

The smoke uses a synthetic image and does not access train, validation or test data.

## Resume state and gates

`checkpoint-200` was revalidated with global step 200, four model shards, eight optimizer-state
files, eight model-state files and eight RNG-state files. Steps 201-255 are not retained and must
be recomputed. The filesystem has approximately 4,280 GiB free.

Training was not restarted in this action because two independent gates remain:

1. GPU 0 is occupied by an unrelated process (PID 533179) using approximately 69,426 MiB. The
   process was not stopped or modified. The formal launcher requires all eight GPUs idle.
2. The shared physical-request ledger already contains 2,223 settled requests. From step 200,
   10,400 logical judgments remain. The old 12,361 cap would leave only 10,138 requests and must
   stop 262 requests before completion even with zero new retries. Of the existing requests, 615 belong to
   failed/discarded trajectory portions or interrupted calls. Preserving the original 360-attempt
   retry allowance therefore requires the minimum revised shared cap `12,976 = 12,361 + 615`.
   This cap change requires explicit user approval and must not be applied silently.

The retained scientific trajectory used 1,607 physical attempts through checkpoint 200 for 1,600
logical judgments, leaving 353 of the original 360 retry slots under the proposed 12,976 cap.
The USD ledger ceiling remains `$26.71782984`; this record does not authorize changing it.

## Code hashes

- `scripts/stage3_openrouter_judge.py`:
  `5a42a8c8f000a94c47f264824a0c2fb0983e6487f9f4f891b179e7b58f06cb48`
- `scripts/stage3_checkpoint_recovery.py`:
  `1b6242d9bf484864f724d9084ac838fa2608bdd72d4505e30a518faea6a04aee`
- `scripts/test_stage3_openrouter_judge.py`:
  `69a4ddccb8cad20b9f03d8e2c9041827f86149b9f40e1925a54bc9a31e95e190`
- `scripts/test_stage3_checkpoint_recovery.py`:
  `8d845c65dd702c44430487dcc68af7cf25c5261732e8d6ba3640d5fde4516650`
