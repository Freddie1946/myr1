# Corrected rule-RL formal launch

The corrected 150-step rule-only GRPO run started at `2026-08-11T15:59:38+08:00` from the
preselected L-SFT3000 step-80 parent. This run supersedes the scientifically invalid pre-fix
rule-RL attempt; the invalid artifacts remain audit-only.

## Frozen configuration

- 8 x A100, per-device batch 10, four generations, global 20 unique prompts per optimizer step.
- 150 optimizer steps, equivalent to three exposures of the fixed 1,000-prompt RL split.
- Language LoRA only (`r=16`, `alpha=32`); vision encoder and projector frozen.
- Learning rate `1e-6`, chosen by the completed PDB10 validation-only calibration.
- Maximum completion length 384; no judge LLM and no test split used for launch selection.

## First-step verification

The trainability audit passed and the first two optimizer steps were finite. Step 1 had gradient
norm `0.1291`, accuracy reward `0.4375`, format reward `0.9375`; step 2 had gradient norm `0.1533`,
accuracy reward `0.4375`, format reward `0.9125`, and KL `0.0003223`. Eight reward-event streams
are being written, and the corrected fail-closed prompt/problem alignment gates have not fired.

The observed first-step wall time was approximately 44 seconds. The conservative completion
estimate is 1.8 hours for training plus 5--15 minutes for final save and the exact post-run reward
alignment verifier. Completion and scientific validity are not claimed until that verifier passes.

Run directory:

`/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/formal_selected_rule_rl1000_corrected_20260811/formal_z2_mbs10_lr1e6_len384`

Machine-readable launch record:

`protocol/corrected_rule_rl_formal_launch_20260811_155938.json`
