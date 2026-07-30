# SFT4000 control and external evaluation authorized

Created: `2026-07-29T01:44:10+08:00`

The user authorized the following ordered execution:

1. download the exact selected SFT n3000 epoch-3/step-1125 and Outcome-GRPO
   epoch-2/step-1000 snapshots from their fixed private Hugging Face revisions;
2. run an exposure-matched SFT control by branching from the selected n3000 model weights and
   training on the exact Stage2 RL1000 records with their gold CoT targets for two ordinary SFT
   epochs;
3. evaluate the fixed SFT4000 epoch-2 checkpoint and selected Stage2 RL checkpoint on the already
   development-classified PathMMU test999 split, then update the joint bad-case report;
4. only after those steps, freeze common PathVQA and OmniMedVQA contracts, evaluate the available
   baselines and study any anomalously high results for disclosed dataset use, source-dataset
   reuse, exact overlap and near-duplicate risk.

The frozen training protocol is
`protocol/sft4000_control_n1000_seed42_manifest_20260729_014410.json`.

“SFT4000” denotes 3,000 previously selected SFT records plus the distinct 1,000 records allocated
to Stage2 RL; it does not mean restarting from the base model on a concatenated 4,000-record file.
The new branch uses a fresh SFT optimizer, matching the scientific branch point used by Stage2
rather than resuming the old ten-epoch SFT optimizer/scheduler state.

The two epochs and the epoch-2 evaluation checkpoint were fixed before this run. PathMMU test999
cannot select the checkpoint or alter the training configuration. Stage3 remains unstarted.

PathVQA and OmniMedVQA are no longer intended to stay permanently sealed: their result seal remains
in force only until the SFT4000/PathMMU work completes and their own model, prompt, decoding and
metric contracts are frozen. Their later results must distinguish standard use of an official
training split from known test overlap and from opaque training corpora for which contamination
cannot be excluded.
