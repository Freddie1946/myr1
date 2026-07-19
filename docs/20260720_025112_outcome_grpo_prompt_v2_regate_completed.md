# Outcome-GRPO prompt-contract v2 re-gate completed

Timestamp: 2026-07-20 02:51:12 Asia/Shanghai

## Result

The user-approved prompt-contract-v2 one-step Outcome-GRPO re-gate completed and passed every frozen
gate.

- Run ID: `outcome_grpo_prompt_v2_gate_n0008_seed0042_20260720_024745`.
- Run manifest:
  `/home/wjy/pathvlm_r1_v1_formal/runs/stage2_outcome_grpo/gate_n0008_seed0042/`
  `outcome_grpo_prompt_v2_gate_n0008_seed0042_20260720_024745/run_manifest.json`.
- Run-manifest SHA-256:
  `caf3acecb3c1e84c8d1080722103026f0b55cda6eb93679f3443e41b07a67549`.
- Status `completed`; `gate_passed: true`; `formal_result: false`.
- Created 02:47:45 and completed 02:50:39 Asia/Shanghai.
- Test was not accessed; `picked.json` was not used; no long GRPO was authorized.

Both the earlier gate and this corrected re-gate remain preserved. Neither one-step child checkpoint
may parent a formal long run.

## Prompt and reward evidence

The only prompt change was removal of the contradictory trailing JSON-format sentence. All eight
raw completions now end at `</answer>` without a trailing JSON/XML-like block.

- Raw reward events: 16/16 present and valid JSONL.
- Online/offline parser-v2 and format-reward consistency: 16/16 exact.
- Accuracy rewards: 6/8 positive; mean `0.75`.
- Format rewards: 8/8 positive; mean `1.0`.
- Total-reward groups: `[2,2,2,2]` and `[1,2,2,1]`.
- Group sample standard deviations: `0.0` and `0.5773503`; mean `0.2886751`.
- Trainer reward standard deviation: `0.2886751`.

The format branch is now demonstrably aligned with the model prompt. In this eight-completion sample
all formats are correct, so format reward itself has no within-group variance; the second prompt's
accuracy outcomes provide the nonzero total-reward advantages. This is acceptable plumbing evidence,
not an estimate of formal-run reward distributions.

## Update and freeze evidence

- Exactly one optimizer step; distributed return code 0.
- Finite gradient norm: `3.3764441`.
- Finite step loss `-0.0`; aggregate trainer loss approximately `-7.45e-09`.
- Full language trainability: 7,615,616,512 / 7,615,616,512.
- Visual trainability: 0 / 676,550,144.
- Projector trainability: 0 / 44,574,464.
- Representative language tensor: 225,514 / 12,845,056 elements changed; maximum absolute delta
  approximately `1.91e-06`.
- Representative visual tensor: 0 / 1,638,400 elements changed; exact equality true.
- Saved Qwen2.5-VL model/processor offline reload gate passed.

The re-gate used the same exact formal SFT n=3000 epoch-3/step-1125 parent and frozen v2 RL-smoke
data. All parent, selection, data, image-isolation, GPU-idle and disk-reserve checks passed.

## Storage and next boundary

The corrected gate occupies about 16 GiB. `/home` has about 900 GiB free after completion, above the
550-GiB reserve. No automatic pruning is allowed. The earlier and corrected gate artifacts together
consume about 32 GiB and are retained for audit.

The Outcome-GRPO engineering prerequisites now pass. Before formal long Stage 2 work, a new protocol
must freeze the RL scale/combination design, maximum steps or epochs, checkpoint/save policy,
validation-only selection rule, seed plan, storage budget, failure/restart behavior and whether runs
are sequential or parallel. Formal runs must restart from the selected SFT parent, not either gate
child. No long training was started in this step.

## Reviewer mapping

- R2-4: outcome-reward execution, exact prompt/parser behavior and raw events now have valid formal-
  hardware engineering evidence; GPT-4o/Stage-3 questions remain open.
- R3-3: the outcome-only arm is technically ready for a separately frozen formal run; no matched
  process-reward ablation exists.
- R3-5: the RL-scale experiment remains pending, but its one-step prerequisite is now satisfied.
- R3-6: seed and provenance are fixed for this engineering gate; it is not a multi-seed formal result.
