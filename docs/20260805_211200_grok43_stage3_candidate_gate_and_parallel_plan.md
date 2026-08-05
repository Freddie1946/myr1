# Grok 4.3 Stage3 candidate gate and parallel evaluation plan

Recorded at: `2026-08-05 21:12 CST`

## Candidate result

The selected replacement candidate is `grok-4.3` through the AIGCBest OpenAI-compatible gateway.
It is not an OpenRouter route and must be reported as `Grok 4.3 via AIGCBest`; the returned model
string does not independently prove the upstream route.

Eight independent, zero-retry PathMMU-validation Judge smokes passed. Every request returned HTTP
200, exact served identity `grok-4.3`, a unique response ID and a valid strict six-event JSON
object. Latency ranged from 9.34 to 22.74 seconds. The eight estimated request costs sum to
`$0.02914`. The immutable aggregate gate is:

`/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/reports/aigcbest_stage3_candidate_smoke_20260805/grok-4.3-stability-gate.json`

Its SHA-256 is `c9881552154ffbe927ab8aced7265448629ec8fc9d6851d49b7a712b98282752`.

Claude Sonnet 4.6 failed its first strict-JSON smoke after an HTTP response, and the pre-correction
failure recorder did not preserve that response body. It was not retried. Gemini 2.5 Pro All
timed out after 180 seconds without a retry. Neither candidate is eligible for the current formal
arm. The smoke client now preserves a received response even when schema validation subsequently
fails.

## Frozen Grok arm

- parent: Stage2 Outcome-GRPO epoch 2 step 1000;
- dataset: frozen image-disjoint RL1000;
- Judge: `grok-4.3` via AIGCBest, exact identity required;
- coefficient: 0.4;
- training/data seed: 42/42;
- extent: 1,500 optimizer steps, four generations per prompt;
- checkpoint interval: 100 steps, with epoch snapshots at 500/1000/1500;
- physical request ceiling: 12,360;
- USD hard ledger ceiling: 50;
- bounded response/transport retries and 24-total/4-consecutive structural fallbacks inherited
  from the audited GPT-4o contract;
- independent cache, budget ledger, audit and run directory.

## Parallel GPU plan

All eight GPUs were idle at the execution boundary. Grok Stage3 loads first under the existing
eight-GPU admission gate. After its first training step proves resident memory and Judge transport,
the completed GPT-4o selected checkpoint may run PathMMU, PathVQA yes/no and OmniMedVQA inference
on three explicitly assigned GPUs, and the GPT-4o visual-fidelity arm may use a fourth GPU. Each
secondary process receives its own memory admission smoke; an OOM or abnormal training-memory
increase stops the secondary launch, not the training arm.

The visual run is an independently verified GPT-4o arm on the already-frozen 24-case panel. It is
not labelled as the completed five-arm comparison.

## Additional local baseline preparation

Qwen3.5-4B is selected as the next lightweight general multimodal baseline. Its isolated
environment pins Transformers commit `0104732151118872214a2bcdc5badd32d938ab38`; CPU configuration
and processor loading passed. Full inference remains gated on a 16-case behavior smoke. NVIDIA
LocateAnything-3B is retained as a specialized localization candidate and is not reported as a
direct VQA baseline.

Llama 3.2 Vision 11B is already locally deployed. Its missing PathVQA value is caused by the prior
mixed-answer smoke producing 9/16 generation-cap hits, not by missing weights or an API endpoint.
A separately frozen yes/no-only short-answer adapter can be tested later without rewriting the
old failed gate.
