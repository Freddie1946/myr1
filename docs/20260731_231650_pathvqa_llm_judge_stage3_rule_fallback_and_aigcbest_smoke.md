# PathVQA LLM judge, bounded Stage3 rule fallback and AIGCBest smoke

Recorded at: 2026-07-31 23:16:50 CST

## External-VQA scoring split

The frozen PathVQA test set has 6,719 rows: 3,357 free-form and 3,362 yes/no.  Only the
3,357 free-form rows are eligible for the supplemental LLM semantic judge.  Yes/no retains its
deterministic accuracy metric.  OmniMedVQA remains a four-option task and does not use an LLM
judge.

The new PathVQA semantic judge uses `claude-sonnet-4-6` through the AIGCBest OpenAI-compatible
endpoint.  It sees question, reference answer and target-blind extracted candidate answer, but not
the evaluated image.  Its strict JSON result contains a boolean, a short reason and one frozen
error category.  Exact/paper metrics remain reported alongside this supplemental metric.

Two paid calibration cases passed:

- PathVQA index 1, `Each histone subunit is positively charged.`: correct.
- PathVQA index 25, the AIDS-related description without the requested expansion: incorrect,
  `omission`.

Observed usage was 351 input + 27 output tokens and 376 input + 42 output tokens.  With the public
default-group Sonnet 4.6 ratios (approximately USD 3/M input and USD 15/M output), this is an
estimated average of USD 0.001608 per judged answer.  Provisional totals are therefore USD 5.40
per complete model, USD 16.20 for the SFT3000/SFT4000/Stage2 triad, or USD 107.96 for twenty full
model outputs.  No full judging run is authorized by this smoke.

## Stage3 outage behavior

Cross-segment judge cache reuse is disabled.  A restarted training segment gets a fresh cache
namespace, while the shared budget ledger remains cumulative.

When the remote judge is unavailable, the local fallback is deliberately weaker and bounded:

- It reads only the generated completion, never the reference answer.
- It awards structural evidence for a non-empty reasoning/answer pair, image-grounding language,
  comparison/elimination language and medical-support language.
- It cannot judge medical correctness and is capped at process reward 0.5.
- At most 24 completions may use the rule in the entire training run.
- At most four may use it consecutively.  A fifth consecutive unavailable response stops the
  segment and invokes checkpoint recovery.
- A successful LLM judge resets only the consecutive counter, not the total counter.
- Every ambiguous failed request is first charged at the full USD 0.05 local reservation; no
  default event labels or hidden zero/one reward are fabricated.

These defaults correspond to at most 0.4% of the nominal 6,000 Stage3 judge decisions and one
four-generation GRPO group consecutively.  They are implemented and tested but remain subject to
explicit approval before the formal Stage3 restart.

## AIGCBest vision smoke and provisional baseline cost

One paid `qwen-vl-plus` smoke used OmniMedVQA index 4.  It served the requested model, returned
`D) CT.`, reached `stop`, and took 3.94 seconds.  Usage was 177 input and 5 output tokens.  The
public pricing row had model ratio 0.4 and completion ratio 2.5; under the default group this maps
to approximately USD 0.80/M input and USD 2.00/M output.  The estimated request cost was
USD 0.0001516.

Scaling that observed Omni request and a provisional 150-input/10-output PathVQA request gives
approximately USD 2.23 for both complete datasets with Qwen VL Plus, or USD 2.57 including a 15%
contingency.  A USD 3.00 hard cap is therefore recommended for the first Qwen VL Plus full
baseline.  Other providers can tokenize images differently and require their own smoke before a
reliable full-run cap is set.

The API token reports `unlimited_quota=true`; its granted/available counters changed from 0 to
-20 while `total_used` remained 0.  These counters are not treated as an auditable currency bill.
The cost estimates above use the public price ratios and response token usage instead.

Smoke artifacts:

- `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/aigcbest_vision_smoke_20260731/qwen-vl-plus_omnimed_index0004.json`
  (SHA-256 `7a2a3f3bfddb67d2e57176b42ac401c5e291366e2c849616a54c52b78c040d61`)
- `/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/runs/aigcbest_pathvqa_judge_smoke_20260731/claude-sonnet-4-6_indices1_25.jsonl`
  (SHA-256 `38cc164dda85ec3c712eda7324ceba6e29a1bfd585ae24e219cc29a709d9084c`)

Forty-three CUDA-free regression tests passed.  No full external baseline, full PathVQA LLM
judging run or formal Stage3 restart was started.
