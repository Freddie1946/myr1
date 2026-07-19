# Formal Outcome-GRPO one-step gate completed

Timestamp: 2026-07-20 02:19:19 Asia/Shanghai

## Outcome

The frozen eight-GPU, one-optimizer-step Outcome-GRPO engineering gate completed successfully:

- Run ID: `outcome_grpo_gate_n0008_seed0042_20260720_021525`.
- Run manifest:
  `/home/wjy/pathvlm_r1_v1_formal/runs/stage2_outcome_grpo/gate_n0008_seed0042/`
  `outcome_grpo_gate_n0008_seed0042_20260720_021525/run_manifest.json`.
- Run-manifest SHA-256:
  `a82a462dbd68ece09dfdeb38fa730fda1aa940b8aa01020000d3211ed18e0592`.
- Status: `completed`; `gate_passed: true`; `formal_result: false`.
- Created 02:15:25 and completed 02:18:34 Asia/Shanghai.
- Test was not accessed; `picked.json` was not used; no long GRPO was authorized.

This is valid engineering evidence only. It is not a paper result, not a formal RL-scale point, and
its one-step child checkpoint must not parent a later formal long run.

## Parent, data and resource gates

- Exact formal seed-42 n=3000 epoch-3/step-1125 SFT parent; all 17 snapshot files passed size and
  SHA-256 verification.
- Validation selection evidence: 60.52% accuracy and 99.74% format on the frozen 385-QA validation
  split; all selected-result gates true.
- Frozen v2 RL-smoke adapter: exact 8 records, 3 unique image contents, every row an exact member of
  the 1000-QA RL split, and every image path valid.
- All basename and exact-content overlaps among SFT/RL/validation/test remain zero.
- All eight RTX 4090 GPUs had no compute process at the launch check. No process was terminated.
- The run occupied about 16 GiB. Free `/home` space after completion was about 915 GiB, above the
  frozen 550-GiB reserve.

## Optimizer and freeze evidence

- Exactly one optimizer step; return code 0.
- Logged gradient norm: `2.9295966625213623` (finite and nonzero).
- Logged reward standard deviation: `0.25` (positive).
- Logged step loss: `0.0`; trainer aggregate loss: approximately `7.45e-09`; both finite.
- Full language parameters trainable: 7,615,616,512 / 7,615,616,512.
- Visual parameters trainable: 0 / 676,550,144.
- Multimodal projector parameters trainable: 0 / 44,574,464.
- Representative language tensor `model.layers.0.self_attn.q_proj.weight`: 227,735 / 12,845,056
  elements changed; maximum absolute delta about `1.91e-06`.
- Representative visual tensor `visual.blocks.0.attn.proj.weight`: 0 / 1,638,400 elements changed;
  exact equality true.
- Saved Qwen2.5-VL config and processor reload checks passed in offline mode.

## Reward and raw-generation audit

Each rank produced one accuracy and one format JSONL record: 16/16 events and 8/8 completions were
present. Pairing by rank/call/item succeeded. Offline parser-v2 recomputation exactly matched every
online reward.

- Accuracy rewards: seven correct and one incorrect; mean `0.875`.
- Format rewards: all eight were `0.0`; mean `0.0`.
- Reconstructed total-reward groups: `[1,1,1,1]` and `[1,1,1,0]`.
- Group sample standard deviations: `0.0` and `0.5`; mean `0.25`.

The second group supplied genuine total-reward variance and produced the nonzero language update.
The first group had zero within-group advantage, which is expected for four identical total rewards.
Raw completions and reward events are preserved under the run directory for bad-case analysis.

## Newly exposed prompt/reward contract conflict

The all-zero format reward is a real protocol issue before formal long Outcome GRPO. The vendored
PathMMU question template ends with `Output the final answer in JSON format.` The generated outputs
usually contained a correct `<think>...</think><answer>...</answer>` followed by an additional
`<json>...</json>`/`<JSON>...</JSON>` block. Parser v2 could still extract the explicit answer, but
the strict format reward correctly rejected the extra trailing block because its frozen contract is
the exact think/answer structure.

Therefore:

- the frozen one-step gate remains passed under its predeclared requirements (positive total reward
  variance, parser consistency, nonzero update, save/load and freeze evidence);
- the format-reward branch was ineffective on this gate sample;
- no formal long Outcome-GRPO run may start until the prompt/reward contract is explicitly chosen,
  frozen in a new timestamped protocol/code manifest, and rechecked with another engineering gate;
- no automatic retry or silent prompt/reward change was performed.

The cleanest proposed correction is to remove the contradictory JSON-format sentence from the
question template and keep the strict `<think>...</think><answer>...</answer>` reward. Alternatives
(accept trailing JSON or remove format reward) change the scientific method differently and require
the user's explicit choice.

## Reviewer mapping

- R2-4: completes a formal-hardware engineering audit of outcome reward online/offline consistency,
  raw calls and effective language update, while also documenting the format-contract defect. It
  does not answer the GPT-4o/Stage-3 questions.
- R3-3: establishes the outcome-only plumbing prerequisite but no valid long outcome-only or process-
  aware comparison yet exists.
- R3-5: the gate is prerequisite evidence for RL 250/500/1000 scaling; no formal RL-scale point has
  been produced.
- R3-6: seed 42 and all per-completion events are preserved, but this engineering run is not a
  multi-seed formal result.
