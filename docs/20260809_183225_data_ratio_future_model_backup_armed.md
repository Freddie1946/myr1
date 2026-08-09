# Future data-ratio terminal-model backup armed

Timestamp: 2026-08-09 18:32:25 Asia/Shanghai

The private-HF backup correction now covers both already completed and future terminal models.

- Completed-model worker PID: `2884852`.
- Completed-model worker state:
  `/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/reports/data_ratio_model_hf_backup_20260809/state.json`.
- Future-model worker PID: `2886597`.
- Future-model worker state:
  `/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/reports/data_ratio_future_model_hf_backup_20260809/state.json`.
- Implementation commit: `96436a471971bc004a8855ffdbc69c7c72125946`.

The future worker first waits for all five completed model-only uploads to finish, preventing
concurrent large-folder uploads. It then checks formal state and the exact train-state audit before
uploading these terminal outputs:

1. `sft0250_rl0750_rule_rl`, expected step 1125;
2. `stage2_continue_rule_rl1000`, expected step 1500;
3. `base_rule_rl4000`, expected step 6000.

The 50-step 0+4000 gate is not a scientific terminal model and is not uploaded. Every repository is
private and receives model-only weights plus a content-hash manifest; optimizer/RNG state remains
in the two rolling local recovery checkpoints. This record arms the backup policy and does not
claim that future training or uploads have completed.
