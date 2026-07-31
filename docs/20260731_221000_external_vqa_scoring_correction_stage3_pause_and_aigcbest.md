# External-VQA scoring correction, Stage3 pause, and AIGCBest gate update

## Stage3 attempt 02 paused

Attempt 01 failed at step 274 because an upstream chunked response ended
abnormally.  The user had already authorized the formal Kimi arm and a USD 30
overall cap, so the failure was initially treated as an in-scope operational
recovery.  That was not sufficient justification for an immediate restart:
there was no checkpoint, the run had to start over, and the transport correction
cannot guarantee that a future response body will not be genuinely truncated.
The failure evidence and residual risk should have been presented first.

After the user questioned the restart, attempt 02 was stopped with SIGINT.  It
had reached step 11/1500 and had no checkpoint.  All GPUs are idle.  Eight
in-flight reservations were conservatively settled at USD 0.05 each, leaving
zero unresolved reservations.

- Attempt 01 conservatively settled: USD `2.78313963`
- Attempt 02 conservatively settled: USD `0.49903053`
- Total conservatively accounted: USD `3.28217016`
- Remaining under the original USD 30 authorization: USD `26.71782984`

The transport correction handles one specific failure mode: it accepts partial
bytes only if those bytes already form a complete JSON object and only the HTTP
chunk terminator is missing.  A genuinely truncated JSON object still fails
closed and can still terminate training.  Step 274 itself is not known to be a
bad dataset item; the observed failure is transport-level and stochastic.

No third formal attempt will start until the user approves a recovery policy.
The scientifically conservative options are:

1. retain fail-closed judge behavior, save every 100 steps, and add a bounded
   supervisor that resumes only from a complete checkpoint after a transport
   failure; or
2. convert a judge transport failure into an audited neutral/zero reward and
   continue the optimizer step, which changes the reward contract and therefore
   requires explicit scientific approval.

Option 1 preserves reward semantics and is recommended, but reduces rather than
eliminates wasted work and cannot protect steps before the first checkpoint.

## PathVQA scoring correction

The PathVQA paper requires different reporting by answer type: accuracy for
yes/no questions, and exact match, macro token F1 and BLEU for open-ended
questions.  The prior evaluator incorrectly collapsed all 6,719 questions into
one normalized whole-completion exact-match percentage.  The repository also
contains a function named `calculate_exactmatch` whose implementation is token
coverage rather than literal equality; it is now retained only as a clearly
labelled repository diagnostic, not silently presented as paper accuracy.

The corrected scorer preserves strict exact match, adds target-blind extraction
for explicit answer tags/yes-no answer markers, and reports the repository token
F1/coverage values separately.  Existing inference outputs were rescored without
regeneration.

| Model | Yes/no contract-aligned accuracy | Open strict exact | Open macro token F1 |
|---|---:|---:|---:|
| SFT3000 | 51.40% | 0.06% | 3.02% |
| SFT4000 | 30.34% | 0.00% | 2.86% |
| Stage2 RL | 49.64% | 0.09% | 3.15% |

The corresponding open-ended sentence BLEU-1/2/3 values are `1.52/0.14/0.04`
for SFT3000, `1.09/0.12/0.02` for SFT4000, and `1.66/0.20/0.04` for Stage2.

The correction confirms that the previous single PathVQA number was unsuitable,
but it also confirms a real open-ended weakness: answer extraction alone does
not turn the free-form result into a strong score.

## OmniMedVQA scoring correction

The official question-answering evaluator applies `SequenceMatcher` to the model
response and candidate option texts.  It assumes the prompt's instruction that
the response contains only one candidate answer.  For models that emit a
`<think>...<answer>...` wrapper, the corrected contract reports both:

- the untouched raw-response official Question-answering Score; and
- a target-blind contract-aligned score that extracts `<answer>`/final-answer
  content and then applies the same option matcher.

| Model | Raw-response score | Contract-aligned final-answer score |
|---|---:|---:|
| SFT3000 | 47.07% | 44.88% |
| SFT4000 | 47.31% | 43.14% |
| Stage2 RL | 47.30% | 45.48% |

The aligned score can be lower because the raw thought text can accidentally be
more similar to the correct option even when the explicit final answer is wrong.
Both directions of disagreement are retained.  Hosted APIs generally do not
expose per-option token likelihoods, so the paper's Prefix-based Score cannot be
reconstructed for those APIs; they will be reported using the Question-answering
track only.

Rescored output root:

`/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/reports/external_vqa_rescored_v2_20260731`

## AIGCBest eligibility and catalog

The user clarified that they are located in Taiwan.  The provider's published
mainland-China exclusion therefore does not, by itself, exclude this user.  The
previous location inference is withdrawn.

The public catalog currently contains exact or close-enough runnable identities
for Qwen-VL-Plus, Doubao 1.5 Vision Pro, Grok 4 Fast non-reasoning, Claude Haiku
4.5, and Llama 3.2 Vision 11B/90B.  Claude 3.5 Haiku was not found in the current
catalog and must be marked unavailable/retired unless another exact provider is
used.

No paid AIGCBest request has been made.  Before a smoke test, the remaining
gates are a separate credential, an explicit budget, and provider account/data
terms.  Public catalog visibility alone does not authenticate the actual model
route or guarantee image support, so each frozen slug needs a one-item paid
identity/vision smoke before full evaluation.
