# Kimi segment03 checkpoint resume verified

Recorded at: `2026-08-04 23:20:00 CST`

## Launch

After all eight A100 GPUs became available, the final predeclared recovery segment was launched
under the formal supervisor from the structurally validated `checkpoint-200` snapshot:

- supervisor PID: `660299`
- launcher PID: `660305`
- torch distributed PID: `660325`
- master port: `29644`
- workers: 8, one per GPU
- supervisor log: `supervisor_resume_segment03_20260804_231710.log`
- training log: `train_segment03.log`

The resume snapshot contained four model shards, eight optimizer states, eight model states and
eight RNG states. The trusted-local-checkpoint compatibility setting
`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` was enabled only after this validation.

## Resume verification

All eight DeepSpeed optimizer partitions and all eight Trainer RNG states loaded successfully.
The PyTorch 2.6 `_pickle.UnpicklingError: Weights only load failed` failure from segment02 did not
recur. Training passed the previous failure point and emitted resumed optimizer metrics:

- step 201: loss `0.0003`, reward `2.6374998092651367`, KL `0.007232666015625`
- step 202: loss `0.0002`, reward `2.450000047683716`, KL `0.00433349609375`

During the first resumed forward pass, all eight GPUs were assigned to the expected workers and
sampled SM utilization was approximately 81%-98%. The Judge audit files and budget ledger
continued to update. An observation during the first resumed step showed 2,239 completed unique
requests, `$4.031979790000004` committed spend and eight in-flight reservations; these are a
runtime snapshot, not a final settlement.

## Status and guardrail

Segment03 is running and recovery is verified, but the 1,500-step formal run is not claimed as
complete. Segment03 is the final predeclared recovery index. If it terminates, its logs and audit
state must be preserved and the failure must be diagnosed before any further launch; no blind
restart or log overwrite is authorized by this record.

Code commit at launch: `05468f8a232e1423f1fdf421123c0574fb5ce45c`.
