# Corrected 1024-token validation Attempt01 stopped on degenerate cap hits

Timestamp: `2026-07-21T22:54:42+08:00`

## Disposition

Attempt01 is a preserved failed-gate run (`formal_result: false`). It created ten of the 26 jobs,
saved 3,850 raw predictions, completed eight jobs, failed two jobs after all 385 predictions were
generated, and did not start the remaining sixteen jobs. The fail-closed stop behaved as frozen.
No automatic retry occurred, all GPUs returned to the 18-MiB idle baseline, and test was not
accessed.

Run root:

`/home/wjy/pathvlm_r1_v1_formal/runs/validation_1024_all_candidates/validation_1024_20260721_184726`

## Exact failures

- SFT n=2000 epoch 1, validation index 148 generated exactly 1,024 tokens without EOS. The output
  entered a repeated `karyolysis, karyosis` loop, never emitted a final answer, and scored accuracy
  0 / format 0.
- SFT n=2000 epoch 5, validation index 257 generated exactly 1,024 tokens without EOS. It repeatedly
  emitted malformed think/answer tags and the same rationale. Its parsed choice was wrong and it
  scored accuracy 0 / format 0.

These are model degeneration failures, not coherent reasoning truncated slightly before a final
answer. Raising the cap to 2,048 would extend the loops rather than recover a scientifically useful
completion.

## Protocol issue exposed

The frozen rule that any single 1,024-token cap hit invalidates the entire 26-job run is too strict
for rare degenerate loops. A fixed evaluation timeout should preserve and score such a completion
as model failure, while reporting cap-hit counts separately. It should invalidate the evaluation
configuration only when cap hits indicate systematic coherent truncation, as occurred for the Base
under the old 192-token protocol.

No correction or Attempt02 is authorized by this record. The user must confirm whether to freeze a
new policy that keeps 1,024 as the output timeout, counts each cap-hit output under the unchanged
parser, reports it as a format/generation failure, and reruns all 26 candidates from scratch.

