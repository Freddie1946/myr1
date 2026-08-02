# Local GPU baselines to Kimi fail-closed sequence

Recorded at: `2026-08-03 01:35:00 CST`

The user added USD 6 to OpenRouter. The account's previously checked available credit is
`$27.312309725`, which covers but does not enlarge the retained Kimi authorization of
`$26.71782984`. The Kimi arm therefore keeps the existing remaining ceiling.

`scripts/sequence_local_gpu_baselines_then_kimi.sh` encodes the already approved remaining order:

1. wait for and integrity-check the four running Llama 11B / DeepSeek-VL2 full evaluations;
2. require all eight GPUs to be idle;
3. run independent 16-case Llama 90B smokes for PathMMU, PathVQA and OmniMedVQA under the frozen
   80% nonempty, 80% parseable, 20% generation-cap policy;
4. run only the Llama 90B tasks whose own smoke passed, in sequence on all eight GPUs;
5. run one fresh paid Kimi contract smoke in the fresh run directory, then launch the bounded
   Kimi Stage3 supervisor with 100-step checkpoints and at most three recognized recoveries;
6. stop the sequence after the first Stage3 segment-start event is verified. It does not monitor
   Kimi after launch.

The sequence fails closed if a currently running task disappears without final metrics, a data or
model hash differs, a completed artifact has a count/index/hash mismatch, an incomplete 90B output
directory already exists, the secret contract fails, or Kimi does not reach a supervisor segment
start. It never changes or re-Judges an individual answer and never converts a failed smoke into a
full run. Kimi's old smoke markers are not reused because their budget and request-cap contracts
differ from the fresh arm.

The full-run verifier is `scripts/verify_local_baseline_full.py`; it checks metrics, run config,
prediction count/order, source-hash presence and the predictions-file SHA-256 before a task is
accepted as complete.
