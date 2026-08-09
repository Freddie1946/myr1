# Data-ratio ablation backup changed to results-only

Date: 2026-08-09 (Asia/Shanghai)

Small data-ratio ablation checkpoints are no longer uploaded automatically. Their scientific
value is carried by frozen inputs, training/evaluation metrics, logs, reward audits and the final
comparison table; those artifacts remain covered by GitHub and the private HF results dataset.
Full weights remain on the A100 workspace until the experiment series is evaluated.

The completed-model uploader, its active large-folder child and the future-model watcher were
stopped. The following private HF repositories were then removed after all three corresponding
local outputs were verified to contain their four safetensor shards and model index:

1. `Freddie1946/PathVLM-R1-Ratio-SFT0750-seed42`;
2. `Freddie1946/PathVLM-R1-Ratio-SFT0750-RuleRL0250-seed42`;
3. `Freddie1946/PathVLM-R1-Ratio-SFT0500-seed42` (the remote repository contained no committed
   model weights).

The first two deletions release approximately 33.2 GB of private HF storage. No formal model,
evaluation-results dataset, local checkpoint, training process or experiment result was deleted.

Going forward, the data-ratio queue is results-only for remote backup. After evaluation, one or
more checkpoints may be designated explicitly as final critical models and uploaded manually.
This prevents an automatic per-ratio policy from exhausting private storage again.
