# SFT4000 control completed and PathMMU diagnostics frozen

Completed: `2026-07-29T22:16:01+08:00`

The fixed n3000-parent plus exact-RL1000 two-epoch SFT control completed successfully:

`/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/stage2_control_sft4000/n1000_seed0042/sft4000_control_rl1000_seed0042_epoch02_20260729_212746`

All 13 terminal gates passed. The run consumed the exact ordered 1,000 records used by Stage2,
performed 250 optimizer steps with finite loss and nonzero finite gradients, saved model-only
snapshots at steps 125 and 250, retained both full resume checkpoints, and independently reloaded
the final 8,292,166,656-parameter BF16 model.

The representative language tensor changed in 10,225,041 of 12,845,056 elements. The
representative frozen visual tensor remained exactly equal in all 1,638,400 elements. PathMMU
test999 did not select the training configuration or checkpoint, and Stage3 did not start.

The fixed evaluation checkpoint is:

`/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/stage2_control_sft4000/n1000_seed0042/sft4000_control_rl1000_seed0042_epoch02_20260729_212746/epoch_snapshots/checkpoint-250`

Its snapshot-manifest SHA-256 is
`d966b80a86a74495f35e0a2f34e2c66e1347a674f6db8fa96ac922afa8dc5e1c`.

The next fixed work is a 16-record validation adapter smoke for this checkpoint and the exact
selected Stage2 Outcome-GRPO epoch-2/step-1000 checkpoint, followed by one full PathMMU test999
development-diagnostic run for each. Both use the already frozen common prompt, greedy BF16
decoding, no quantization, a 1,024-token cap, batch size 8 and parser v2.
