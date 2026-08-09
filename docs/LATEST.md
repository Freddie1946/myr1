# Latest recovery pointer

Latest bbox-format probe inside the trained think/answer envelope:

0. `20260810_012000_bbox_output_format_probe.md`

Four XML, Qwen-native, plain-coordinate and embedded-JSON encodings were tested. None produced a
valid spatial bbox on the three-case probe; prompt wording is no longer the priority. A separate
fixed-answer turn with grammar-constrained decoding is the next justified method.

Latest direct-bbox self-report smoke:

0. `20260810_010500_direct_bbox_self_report_smoke.md`

Two Stage3 checkpoints can occasionally emit meaningful image-space boxes, but strict formatting,
answer preservation, abstention and spatial agreement are not yet reliable. Direct bbox output is
therefore an exploratory interface, not reviewer-facing localization evidence.

Latest explicitly selected critical-model HF backup policy:

0. `20260810_003000_critical_model_hf_backup_policy.md`
0. `../protocol/critical_model_hf_backup_policy_20260810_003000.json`

Exactly three model backups are allowed: private Grok Stage3, manually gated public Stage2
continued rule-RL, and manually gated public base+4,000 rule-RL. Gate state is verified before
weight upload. The results-only policy still excludes every small ratio checkpoint.

Latest data-ratio results-only backup policy:

0. `20260809_235000_data_ratio_weights_results_only_policy.md`
0. `../protocol/data_ratio_weights_results_only_policy_20260809_235000.json`

Automatic model uploads for the small data-ratio ablations are retired. Their weights remain
local, while metrics, logs, frozen data/protocol records and reward audits continue to be backed
up. Three ratio repositories were removed from private HF storage, releasing about 33.2 GB. A
model is uploaded only after it is explicitly selected as a final critical checkpoint.

Superseded future data-ratio terminal-model backup automation:

0. `20260809_183225_data_ratio_future_model_backup_armed.md`
0. `../protocol/data_ratio_future_model_backup_armed_20260809_183225.json`

One low-priority worker uploads the five already completed model-only outputs sequentially. A
second worker waits without GPU use and will upload RL750, Stage2 continued rule-RL and the final
0+4000 model only after their exact formal completion audits pass. Full optimizer checkpoints stay
local and the nonterminal 50-step gate is excluded.

Latest data-ratio remote backup correction and safe optimized-tail handoff:

0. `20260809_182842_data_ratio_remote_backup_and_safe_tail_handoff_started.md`
0. `../protocol/data_ratio_remote_backup_safe_tail_handoff_20260809_182842.json`

GitHub and the private HF results snapshot are now synchronized for the frozen protocol, data,
completed logs and rule-reward events. Five completed model-only backups are uploading
sequentially. The active 250+750 rule-RL child continues unchanged while a detached boundary
handoff waits to run SDPA-only throughput smokes; FlashAttention 2 is excluded. Batch, rewards,
formal steps and data exposure do not change, and an automatic baseline fallback is mandatory.

Latest sparse data-ratio rule-RL formal launch:

0. `20260809_034026_data_ratio_rule_rl_ablation_formal_sequence_started.md`
0. `../protocol/data_ratio_rule_rl_ablation_formal_launch_20260809_034026.json`

The nine-task formal sequence is running under detached supervisor PID 2715867 from frozen commit
`8f5eb1d`. Formal preflight passed, test was not accessed, and the first 750-example SFT task
entered its 940-step training loop with finite optimizer updates on all eight A100 GPUs. The queue
is not claimed complete; monitoring, final audit and evaluation belong to later goals.

Latest sparse data-ratio rule-RL preparation and sequential smoke:

0. `20260809_033236_data_ratio_rule_rl_ablation_prepared_and_smoke_completed.md`
0. `../protocol/data_ratio_rule_rl_ablation_v1_20260809.json`
0. `../protocol/data_ratio_rule_rl_ablation_smoke_completion_20260809_033236.json`

The new sparse cross-design uses one fixed 1,000-QA union for 750+250, 500+500 and 250+750
SFT/rule-RL arms. Deterministic image-complete bins, rule-only training contracts, a Stage2
continued-rule-RL control and a gated base+4,000-rule-RL stress test are implemented. Four
CUDA-free tests and all six real eight-A100 sequential smoke tasks passed, including recovery from
a post-training audit-code failure without repeating completed training. The formal detached queue
has now started; later completion monitoring and evaluation are separate goals.

Latest layerwise-attention unseen validation:

0. `20260806_020000_attention_unseen_validation_completed.md`
0. `../protocol/attention_layerwise_unseen_validation_result_20260806_020000.json`

The remaining 20 cases were frozen before new annotations or heatmaps. Claude
Sonnet 4.6 classified only five as spatially boxable; 12 were diffuse and three
were not localizable. On the five held-out spatial cases, the predefined
question/options middle/late attention trend replicated directionally, with
large per-case variance and only 2/5 model answers correct. No single layer or
causal explanation claim is selected.

Latest Grok budget resume and layerwise attention calibration pilot:

0. `20260806_013800_grok_budget_resume_and_layerwise_attention_pilot.md`
0. `../protocol/grok_budget_resume_layerwise_attention_pilot_20260806_013800.json`

The Grok arm restored the full step-200 optimizer state, resumed paid Judge calls,
and produced new optimizer steps under the amended USD 100 protective ceiling.
Separately, the four-case attention pilot now compares all 28 layers under three
query definitions, including a tokenizer-audited question/options-only method.
No primary layer is selected and the external-model ROIs remain pseudo labels.

Latest completed GPT-4o visual-fidelity arm and active partial full evaluation:

0. `20260805_222015_gpt4o_visual_completed_and_partial_full_eval_started.md`
0. `../protocol/gpt4o_visual_completed_partial_full_eval_started_20260805_222015.json`

The selected GPT-4o Stage3 checkpoint completed and independently verified all 24 frozen
perturbation-fidelity cases and 25 figures while sharing GPU 3 with the active Grok arm. Its narrow
fidelity metrics are recorded without an attention/localization claim. A new task-gated evaluation
run passed fresh PathMMU and PathVQA yes/no smokes and entered full 999/3362 inference on GPUs 0/1.
OmniMedVQA remains excluded after its separately preserved 93.75% cap-hit smoke failure.

Latest GPT-4o task-gated parallel evaluation update:

0. `20260805_221524_gpt4o_parallel_smoke_task_gate_and_partial_continuation.md`
0. `../protocol/gpt4o_parallel_smoke_task_gate_partial_continuation_20260805_221524.json`

The shared-GPU three-task smoke stopped before full inference because GPT-4o Stage3 reached the
64-token cap on 15/16 OmniMedVQA cases. PathMMU and PathVQA yes/no passed their independently frozen
behavior gates. The launcher now accepts an explicit task subset while retaining all three as its
default; a new run may continue only the two eligible tasks and must preserve the failed Omni smoke
separately. The independent GPT-4o visual-fidelity arm is active on GPU 3 beside Grok Stage3.

Latest Grok-parallel GPU admission preparation and Gemini 3.5 no-thinking route audit:

0. `20260805_220716_grok_parallel_gpt4o_admission_and_gemini35_route_audit.md`
0. `../protocol/grok_parallel_gpt4o_admission_gemini35_audit_20260805_220716.json`

The user authorized useful evaluations and experiments to run beside the active Grok Stage3 arm.
The prepared opt-in admission mode retains the default idle-GPU gate, requires the Grok owner to
have eight live workers and a completed optimizer step, and requires at least 45 GiB free per
secondary GPU. GPT-4o three-dataset evaluation rechecks the owner after its 16-case smokes before
full inference. The AIGCBest `gemini-3.5-flash-nothinking` route returned the base identity
`gemini-3.5-flash`; it is not eligible as a distinct exact-identity Judge candidate.

Latest approved Kimi balance continuation and verified HTTP 429 recovery:

0. `20260805_195300_kimi_balance_429_recovery_segment06_verified.md`
0. `../protocol/kimi_balance_429_recovery_segment06_verified_20260805_195300.json`

The runtime budget ceiling covers the settled prior spend plus the user-approved remaining USD 20
balance. This record diagnoses segment04's synchronized HTTP 429 outage and segment05's local
rate-limit contract mismatch, preserves both corrections, and records segment06 through completed
step 1105. The run is stopped at the approved 36-event total fallback limit; its latest complete
checkpoint remains checkpoint-1100 and the run is not claimed as complete.

Latest verified Kimi segment03 resume:

0. `20260804_232000_kimi_segment03_resume_verified.md`
0. `../protocol/kimi_segment03_resume_verified_20260804_232000.json`

Segment03 resumed from the structurally validated checkpoint-200 and passed the segment02
PyTorch 2.6 checkpoint-load failure point. Optimizer and RNG state restoration succeeded, all
eight workers are active, and resumed metrics were recorded at steps 201 and 202. The run remains
in progress and is not claimed as complete.

Latest Kimi segment02 pre-training checkpoint-load failure and correction:

0. `20260804_231344_kimi_segment02_checkpoint_load_failure_and_correction.md`
0. `../protocol/kimi_segment02_checkpoint_load_failure_correction_20260804_231344.json`

Segment02 stopped before any optimizer step or Judge call because the full Kimi launcher omitted
the already established PyTorch 2.6 trusted-local-checkpoint compatibility setting. The scoped
fix and a fail-closed atomic request-cap migration command pass 58 tests; segment03 is not yet
claimed as started by this record.

Latest approved Kimi recovery-cap configuration and GPU gate:

0. `20260804_230014_kimi_request_cap_approved_resume_ready_gpu_blocked.md`
0. `../protocol/kimi_request_cap_approved_resume_ready_20260804_230014.json`

The approved 12,976 physical-request recovery cap is frozen and verified. The checkpoint-200
segment02 resume is ready but not running because all eight GPUs are occupied by independent
AFI-VLA processes; none were modified.

Latest Kimi Stage3 HTTP 520 correction and resume gates:

0. `20260804_213149_kimi_http520_retry_fix_and_resume_gates.md`
0. `../protocol/kimi_http520_retry_fix_resume_gates_20260804_213149.json`

The Judge and supervisor correction is locally verified and a fresh paid synthetic smoke passed.
Training is not running: GPU 0 is occupied by an unrelated process, and increasing the exhausted
shared physical-request capacity from 12,361 to the calculated minimum 12,976 awaits explicit
approval.

Latest non-GPU preparation and immutable backup:

0. `20260804_005148_post_kimi_pipeline_and_local_baseline_backup.md`
0. `../protocol/post_kimi_closed_eval_visual_and_backup_20260804_005148.json`

This record provides the single-shot post-Kimi closed-question/visual-fidelity entry point and the
verified private-HF backup of completed Aug 3 local GPU baselines. It does not claim that Kimi,
Stage3 selected-model evaluations or visualization runs are complete.

Current result-table addendum:

0. `20260804_003226_current_evaluation_results_closed_question_only.md`

This addendum follows the current reporting decision to exclude all PathVQA free-form statistics
and shows the verified three-dataset coverage, partial runs and compatibility-gated omissions.

Latest timestamp: `20260802_214734`

Read in this order:

0. `20260802_214734_hosted_semantic_evaluation_completed_and_backed_up.md`
0. `../protocol/hosted_semantic_evaluation_completion_20260802_214734.json`

0. `20260802_160833_periodic_monitoring_disabled_by_user.md`
0. `../protocol/periodic_monitoring_disabled_20260802_160833.json`

0. `20260802_153819_periodic_backup_live_edit_attempt_failure.md`
0. `../protocol/periodic_backup_live_edit_failure_20260802_153819.json`

0. `20260802_153555_periodic_hf_wallclock_timeout_correction.md`
0. `../protocol/periodic_hf_wallclock_timeout_correction_20260802_153555.json`

0. `20260802_152807_aggregate_smoke_policy_frozen.md`
0. `../protocol/aggregate_local_baseline_smoke_policy_20260802_152807.json`

0. `20260802_151307_pathvqa_semantic_finalizer_and_completion_gate_prepared.md`
0. `../protocol/pathvqa_semantic_finalizer_prepared_20260802_151307.json`

0. `20260802_150310_corrected_periodic_hf_backup_verified.md`
0. `../protocol/corrected_periodic_hf_backup_verified_20260802_150310.json`

0. `20260802_150132_periodic_hf_transport_retry_correction.md`
0. `../protocol/periodic_hf_transport_retry_correction_20260802_150132.json`

0. `20260802_145957_periodic_backup_nested_path_filter_correction.md`
0. `../protocol/periodic_backup_nested_path_filter_correction_20260802_145957.json`

0. `20260802_145732_periodic_audit_and_private_hf_backup_enabled.md`
0. `../protocol/periodic_audit_private_hf_backup_enabled_20260802_145732.json`

0. `20260802_145416_periodic_audit_and_private_hf_backup_prepared.md`
0. `../protocol/periodic_audit_private_hf_backup_prepared_20260802_145416.json`

0. `20260802_145005_kimi_stage3_bounded_retry_hardening_prepared.md`
0. `../protocol/stage3_kimi26_bounded_retry_hardening_20260802_145005.json`

0. `20260802_143841_deepseek_vl2_external_vqa_adapter_prepared.md`
0. `../protocol/deepseek_vl2_external_vqa_adapter_20260802_143841.json`

0. `20260802_143422_modelscope_llama32_snapshots_and_local_adapter_prepared.md`
0. `../protocol/modelscope_llama32_local_adapter_20260802_143422.json`

0. `20260802_142834_stage3_post_training_validation_pipeline_prepared.md`
0. `../protocol/stage3_post_training_validation_pipeline_20260802_142834.json`

0. `20260802_141632_evaluation_results_private_hf_backup.md`
0. `../protocol/evaluation_results_private_hf_backup_20260802_141632.json`

0. `20260802_134947_stage3_and_pathvqa_bounded_retry_recovery.md`
0. `../protocol/stage3_pathvqa_bounded_retry_recovery_20260802_134947.json`

0. `20260801_194327_doubao_replacement_and_local_llama_feasibility.md`
0. `../protocol/doubao_replacement_llama_local_feasibility_20260801_194327.json`

0. `20260801_191524_hosted_pathmmu_full_evaluations_started.md`
0. `../protocol/hosted_pathmmu_full_launch_manifest_20260801_191524.json`

0. `20260801_191001_hosted_baseline_behavioral_smoke_results.md`
0. `../protocol/hosted_baseline_behavioral_smoke_manifest_20260801_191001.json`

0. `20260801_185228_gpt4o_formal_preflight_passed.md`
0. `20260801_184826_gpt4o_formal_launcher_and_execution_order_frozen.md`
0. `../protocol/gpt4o_formal_preflight_result_20260801_185228.json`
0. `../protocol/gpt4o_formal_launcher_frozen_20260801_184826.json`

1. `20260712_134550_stage_overview_and_plan.md`
2. `20260712_134550_stage1_sft_debug_log.md`
3. `20260712_134917_stage1_attempt04_plan.md`
4. `20260712_135503_small_model_e2e_plan.md`
5. `20260712_135810_stage1_3b_attempt01_result_and_retry.md`
6. `20260712_140346_stage1_3b_sft_completed.md`
7. `20260712_140619_stage2_outcome_grpo_plan.md`
8. `20260712_142223_stage2_attempt01_failure_and_environment_fix.md`
9. `20260712_142743_stage2_outcome_grpo_functional_completion.md`
10. `20260712_142914_validation_inference_offline_scoring_plan.md`
11. `20260712_144517_validation_inference_offline_scoring_completed.md`
12. `20260712_144517_stage3_process_reward_decisions_pending.md`
13. `20260712_144811_formal_machine_migration_package.md`
14. `20260712_150450_stage2_nonzero_gradient_retry_plan.md`
15. `20260712_151408_stage2_attempt03_reward_parser_audit.md`
16. `20260712_184322_stage2_attempt04_ready_waiting_for_gpus.md`
17. `20260712_191813_git_based_codex_handoff_plan.md`
18. `20260712_192846_formal_git_handoff_implemented.md`
19. `20260712_213340_pathmmu_download_and_processing.md`
20. `20260713_200011_manuscript_and_reviews_added.md`
21. `20260713_200535_formal_machine_codex_runbook.md`
22. `20260713_203631_formal_protocol_and_sft_gate_correction.md`
23. `20260713_204927_formal_gpu_allocation_and_machine_config.md`
24. `20260713_205956_llamafactory_pin_bootstrap_failure_and_correction.md`
25. `20260713_213248_pathmmu_xet_hang_and_http_retry.md`
26. `20260713_214352_model_download_concurrency_correction.md`
27. `20260713_222126_formal_bootstrap_and_preflight_completed.md`
28. `20260713_222306_grpo_dependency_gate_correction.md`
29. `20260714_001041_formal_data_adapter_idempotency_correction.md`
30. `20260714_002042_strengthened_preflight_rerun_completed.md`
31. `20260714_005450_sft_smoke_attempt01_resume_failure.md`
32. `20260714_010813_sft_smoke_attempt02_completed.md`
33. `20260714_012500_formal_sft_scale_execution_and_approval.md`
34. `20260715_230037_formal_sft_n0500_launched.md`
35. `20260715_235000_formal_sft_scale_supervisor.md`
36. `20260716_233301_sft_speed_smoke_and_ablation_correction.md`
37. `20260716_234731_sft_speed_smoke_attempt01_config_key_correction.md`
38. `20260716_235950_sft_speed_smoke_attempt02_completed_and_checkpoint_policy_pending.md`
39. `20260717_002323_two_tier_checkpoint_retention_implemented.md`
40. `20260717_010146_checkpoint_retention_smoke_attempt01_failure_and_retry.md`
41. `20260717_012107_checkpoint_retention_smoke_attempt02_completed.md`
42. `20260717_013408_sft_gc_speed_confirmation_completed.md`
43. `20260717_014045_pre_n3000_storage_cleanup_completed.md`
44. `20260717_014544_formal_sft_n3000_seed42_launched.md`
45. `20260717_015819_n3000_to_n2000_fail_closed_handoff.md`
46. `20260719_152742_formal_sft_completion_and_validation_curve_plan.md`
47. `20260719_162225_validation_curve_attempt01_chat_template_failure_and_retry.md`
48. `20260719_174130_pathmmu_exact_content_v2_and_reviewer_tracker.md`
49. `20260719_182337_validation_curve_v2_preflight_completed.md`
50. `20260719_210843_validation_curve_attempt02_completed.md`
51. `20260719_211417_n0500_n1000_comparability_and_storage_audit.md`
52. `20260720_001137_no_n0500_n1000_rerun_decision.md`
53. `20260720_014051_formal_outcome_grpo_one_step_gate_frozen.md`
54. `20260720_020021_outcome_grpo_parent_metric_evidence_correction.md`
55. `20260720_021919_formal_outcome_grpo_one_step_gate_completed.md`
56. `20260720_023057_outcome_grpo_prompt_v2_regate_frozen.md`
57. `20260720_025112_outcome_grpo_prompt_v2_regate_completed.md`
58. `20260720_033024_stage2_priority_pilot_and_n1000_protocol_frozen.md`
59. `20260720_050144_stage2_priority_pilot_completed_storage_gate_stopped.md`
60. `20260720_063310_pilot_resume_checkpoint_pruned.md`
61. `20260720_065200_stage2_formal_n1000_continuation_frozen.md`
62. `20260721_161141_validation_generation_cap_correction_pilot_plan.md`
63. `20260721_170223_validation_generation_cap_correction_pilot_completed.md`
64. `20260721_181115_validation_1024_all_candidates_frozen.md`
65. `20260721_182701_validation_1024_preflight_completed.md`
66. `20260721_225442_validation_1024_attempt01_degenerate_cap_gate_failure.md`
67. `20260722_000242_validation_1024_attempt02_frozen.md`
68. `20260722_002259_validation_1024_attempt02_preflight_completed.md`
69. `20260722_014643_validation_1024_attempt02_completed.md`
70. `../protocol/stage2_formal_n1000_seed42_completion_manifest_20260722_014643.json`
71. `../protocol/validation_1024_all_candidates_attempt02_manifest_20260722_000242.json`
72. `../protocol/validation_1024_all_candidates_attempt02_code_manifest_20260722.json`
73. `../protocol/validation_1024_all_candidates_manifest_20260721_181115.json`
74. `../protocol/validation_1024_all_candidates_code_manifest_20260721.json`
75. `../protocol/validation_generation_cap_pilot_manifest_20260721_161141.json`
76. `../FORMAL_MACHINE_CODEX_GUIDE.md`
77. `../manuscript/README.md` and its mandatory reading order
78. `../CODEX_START_HERE.md`
79. `../manuscript/revision/reviewer_response_tracker.md`
80. `../protocol/stage2_formal_n1000_seed42_continuation_manifest_20260720_064352.json`
81. `../protocol/stage2_formal_n1000_seed42_continuation_code_manifest_20260720_064352.json`
82. `../protocol/stage2_priority_n1000_seed42_manifest_20260720_033024.json`
83. `../protocol/stage2_priority_n1000_seed42_code_manifest_20260720_033024.json`
84. `../protocol/outcome_grpo_prompt_v2_gate_manifest_20260720_023057.json`
85. `../protocol/outcome_grpo_prompt_v2_gate_code_manifest_20260720_023057.json`
86. `../protocol/outcome_grpo_gate_manifest_20260720_013054.json`
87. `../protocol/outcome_grpo_gate_code_manifest_20260720_013054.json`
88. `../protocol/sft_validation_curve_manifest_20260719_172547.json`
89. `../protocol/sft_validation_curve_manifest_20260719_162000.json`
90. `../protocol/sft_validation_curve_manifest_20260719_152742.json`
91. `../protocol/n3000_to_n2000_handoff_manifest_20260717_015614.json`
92. `../protocol/code_hash_manifest_20260719_172547.json`
93. `../protocol/code_hash_manifest_20260717_010146.json`
94. `../protocol/code_hash_manifest_20260717_002323.json`
95. `../protocol/sft_speed_smoke_manifest_20260716_234731.json`
96. `../protocol/sft_speed_smoke_manifest_20260716_233301.json`
97. `../protocol/sft_scale_sequence_manifest_20260715_235000.json`
98. `../protocol/code_hash_manifest_20260714_012500.json`
99. `../protocol/code_hash_manifest_20260714_005450.json`
100. `../protocol/code_hash_manifest_20260714_001041.json`
101. `../protocol/code_hash_manifest_20260713_222306.json`
102. `../protocol/code_hash_manifest_20260713_214352.json`
103. `../protocol/code_hash_manifest_20260713_213248.json`
104. `../protocol/code_hash_manifest_20260713_205956.json`
105. `../protocol/code_hash_manifest_20260713_203631.json`
106. `../protocol/code_hash_manifest_20260712_213340.json`
107. `../protocol/code_hash_manifest_20260712_192846.json`
108. `../protocol/code_hash_manifest_20260712_144811.json`
109. `../protocol/debug_proxy_model_manifest.json`
110. `../protocol/training_plan.md`
111. `../protocol/old_sft_audit.md`
112. The latest manifest under `../debug_e2e/`
113. `20260725_000657_evaluation_assets_preparation_frozen.md`
114. `../protocol/evaluation_assets_preparation_manifest_20260725_000657.json`
115. `../protocol/pathology_baseline_source_manifest_20260725_000657.json`
116. `../protocol/pathology_clip_requirements_20260725.txt`
117. `20260725_004340_external_evaluation_assets_and_baseline_environment_prepared.md`
118. `../protocol/evaluation_assets_preparation_completion_manifest_20260725_004340.json`
119. `20260725_010533_baseline_access_and_environment_status.md`
120. `../protocol/baseline_access_status_manifest_20260725_010533.json`
121. `20260725_011528_all_manuscript_baselines_mandatory.md`
122. `../protocol/all_manuscript_baselines_manifest_20260725_011528.json`
123. `20260725_013154_manuscript_baseline_metadata_and_access_audit.md`
124. `../protocol/manuscript_baseline_metadata_manifest_20260725_013154.json`
125. `20260727_203723_two_machine_a100_stage3_eval_handoff.md`
126. `../protocol/a100_stage3_eval_handoff_manifest_20260727_203723.json`
127. `../A100_STAGE3_EVAL_CODEX_START.md`
128. `../formal_machine/a100_stage3_eval_workspace.env.example`
129. `../protocol/a100_codex_bootstrap_prompt_20260727.txt`
130. `../protocol/code_hash_manifest_20260727_203723.json`
131. `20260728_013529_a100_preparation_phases_1_3_progress.md`
132. `../protocol/a100_eval_environment_preparation_manifest_20260728_013529.json`
133. `20260728_014950_baseline_evaluation_contract_decisions_pending.md`
134. `20260728_015359_llama_vision_weight_access_gate.md`
135. `20260728_021824_a100_core_preflight_and_external_asset_audit_completed.md`
136. `../protocol/a100_core_preflight_completion_manifest_20260728_021824.json`
137. `20260728_213918_baseline_contract_partial_approval_and_weight_staging.md`
138. `../protocol/baseline_contract_partial_approval_manifest_20260728_213918.json`
139. `20260728_215133_pathmmu_test999_reclassified_as_stage3_development_diagnostic.md`
140. `../protocol/pathmmu_test999_development_reclassification_manifest_20260728_215133.json`
141. `20260728_215918_pathmmu_test999_qwen_family_diagnostic_launch.md`
142. `../protocol/pathmmu_test999_qwen_family_diagnostic_launch_manifest_20260728_215918.json`
143. `../protocol/pathmmu_test999_medgemma_diagnostic_launch_manifest_20260728_220617.json`
144. `../protocol/pathmmu_test999_internvl_diagnostic_launch_manifest_20260728_220821.json`
145. `../protocol/pathmmu_test999_huatuo_diagnostic_launch_manifest_20260728_221830.json`
146. `20260728_222041_pathmmu_test999_first_four_baseline_results_and_badcases.md`
147. `../protocol/pathmmu_deepseek_vl2_response_contract_gate_20260728_222041.json`
148. `../protocol/pathmmu_llava_med_response_contract_gate_20260728_222938.json`
149. `../protocol/pathmmu_test999_pathology_matching_launch_manifest_20260728_223308.json`
150. `20260728_225929_pathmmu_test999_available_baselines_completed.md`
151. `20260729_014410_sft4000_control_and_external_eval_authorized.md`
152. `../protocol/sft4000_control_n1000_seed42_manifest_20260729_014410.json`
153. `20260729_221640_sft4000_control_completed_and_pathmmu_diagnostics_frozen.md`
154. `../protocol/pathmmu_test999_sft4000_stage2_launch_manifest_20260729_221640.json`
155. `20260729_222000_pathmmu_qwen_runner_import_correction.md`
156. `../protocol/pathmmu_test999_sft4000_stage2_import_correction_20260729_222000.json`
157. `../protocol/pathmmu_test999_sft3000_parent_addendum_20260729_222733.json`
158. `20260729_223742_pathmmu_sft3000_sft4000_stage2_diagnostics_completed.md`
159. `../protocol/pathmmu_test999_sft3000_sft4000_stage2_completion_manifest_20260729_223742.json`
160. `../protocol/pathmmu_test999_available_baselines_completion_manifest_20260728_225929.json`
161. `20260729_224503_external_vqa_contract_frozen.md`
162. `../protocol/external_vqa_common_contract_manifest_20260729_224503.json`
163. `20260729_225023_external_vqa_native_adapters_frozen.md`
164. `../protocol/external_vqa_native_adapter_addendum_20260729_225023.json`
165. `20260729_225733_external_vqa_variable_option_serialization_correction.md`
166. `../protocol/external_vqa_variable_option_serialization_correction_20260729_225733.json`
167. `20260729_231550_openrouter_hosted_baseline_and_stage3_judge_audit.md`
168. `../protocol/openrouter_hosted_baseline_stage3_judge_audit_20260729_231550.json`
169. `20260729_232421_claude45_identity_corrected_to_haiku45.md`
170. `../protocol/claude45_haiku45_identity_addendum_20260729_232421.json`
171. `20260730_001946_stage3_50step_openrouter_pilot_proposed.md`
172. `../protocol/stage3_openrouter_50step_pilot_proposal_20260730_001946.json`
173. `20260730_002131_stage3_openrouter_key_gate_attempt01_missing_file.md`
174. `../protocol/stage3_openrouter_key_gate_attempt01_20260730_002131.json`
175. `20260730_002345_stage3_openrouter_key_gate_passed.md`
176. `../protocol/stage3_openrouter_key_gate_passed_20260730_002345.json`
177. `20260730_002723_stage3_50step_openrouter_pilot_approved.md`
178. `../protocol/stage3_openrouter_50step_pilot_approved_20260730_002723.json`
179. `20260730_003915_stage3_openrouter_smoke_blocked_by_distillation_terms.md`
180. `../protocol/stage3_openrouter_smoke_blocked_20260730_003915.json`
181. `20260730_004532_stage3_403_cause_attribution_corrected.md`
182. `../protocol/stage3_403_cause_attribution_correction_20260730_004532.json`
183. `20260730_005955_claude_sonnet46_nontraining_feasibility_check.md`
184. `../protocol/claude_sonnet46_feasibility_20260730_005955.json`
185. `20260730_010753_stage3_kimi26_pilot_prepared_waiting_gates.md`
186. `../protocol/stage3_kimi26_pilot_prepared_20260730_010753.json`
187. `20260730_102142_stage3_kimi26_50step_pilot_completed.md`
188. `../protocol/stage3_kimi26_50step_pilot_completion_20260730_102142.json`
189. `20260730_204758_stage3_kimi26_full_arm_prepared_waiting_budget.md`
190. `../protocol/stage3_kimi26_full_arm_prepared_20260730_204758.json`
191. `20260730_211820_stage3_kimi26_full_arm_started.md`
192. `../protocol/stage3_kimi26_full_arm_started_20260730_211820.json`
193. `20260730_214244_sft4000_checkpoint_uploaded_to_private_hf.md`
194. `../protocol/sft4000_hf_upload_completion_20260730_214244.json`
195. `20260731_215500_stage3_kimi26_attempt01_failure_and_attempt02_restart.md`
196. `20260731_215500_external_vqa_badcase_analysis.md`
197. `20260731_215500_aigcbest_closed_baseline_provider_gate.md`
198. `../protocol/stage3_restart_external_vqa_badcase_manifest_20260731_215500.json`
199. `20260731_221000_external_vqa_scoring_correction_stage3_pause_and_aigcbest.md`
200. `../protocol/external_vqa_scoring_v2_stage3_pause_aigcbest_addendum_20260731_221000.json`
201. `20260731_225832_vqa_examples_and_stage3_bounded_recovery.md`
202. `../protocol/vqa_examples_stage3_bounded_recovery_20260731_225832.json`
203. `20260731_231650_pathvqa_llm_judge_stage3_rule_fallback_and_aigcbest_smoke.md`
204. `../protocol/pathvqa_llm_judge_stage3_rule_fallback_aigcbest_smoke_20260731_231650.json`
205. `20260731_233857_gpt41mini_calibration_and_aigcbest_budget_estimate.md`
206. `../protocol/gpt41mini_calibration_aigcbest_budget_20260731_233857.json`
207. `../protocol/pathvqa_gpt41mini_manual_calibration_cases_20260731.json`
208. `20260801_000216_gpt5mini_pathvqa_judge_comparison.md`
209. `../protocol/gpt5mini_pathvqa_judge_comparison_20260801_000216.json`
210. `20260801_010917_stage3_future_training_plan_approved.md`
211. `../protocol/stage3_future_training_plan_approved_20260801_010917.json`
212. `20260801_012103_stage3_sensitivity_and_gpt4o_smoke.md`
213. `../protocol/stage3_sensitivity_gpt4o_smoke_20260801_012103.json`
214. `20260801_013252_gpt4o_coefficient_pilots_launch.md`
215. `../protocol/stage3_gpt4o_coefficient_pilots_launch_20260801_013252.json`
216. `20260801_014011_gpt4o_pilot_supervision_started.md`
217. `20260801_143832_gpt4o_pilot_progress_and_0p5_attempt_cap_stop.md`
218. `../protocol/gpt4o_pilot_progress_0p5_attempt_cap_stop_20260801_143832.json`
219. `20260801_144436_gpt4o_0p5_fresh_recovery_approved.md`
220. `../protocol/gpt4o_0p5_fresh_recovery_approved_20260801_144436.json`
221. `20260801_144907_gpt4o_recovery_and_validation_supervision.md`
222. `20260801_163520_gpt4o_penalty_pilots_complete.md`
223. `../protocol/gpt4o_penalty_pilots_complete_20260801_163520.json`
224. `20260801_173514_gpt4o_penalty_case_and_rollout_analysis.md`
225. `../protocol/gpt4o_penalty_case_and_rollout_analysis_20260801_173514.json`
226. `20260801_181414_stage3_fault_tolerance_hardening_and_coefficient_interpretation.md`
227. `../protocol/stage3_fault_tolerance_hardening_20260801_181414.json`
228. `20260801_182543_stage3_coefficient_frozen_aigcbest_hardening_and_launch_boundary.md`
229. `../protocol/stage3_coefficient_freeze_and_aigcbest_hardening_20260801_182543.json`
230. `20260801_184826_gpt4o_formal_launcher_and_execution_order_frozen.md`
231. `../protocol/gpt4o_formal_launcher_frozen_20260801_184826.json`
232. `20260803_011601_gpt4o_stage3_selected_uploaded_and_local_gpu_baselines_started.md`
233. `../protocol/gpt4o_stage3_selection_hf_and_local_baselines_20260803_011601.json`
234. `20260803_013500_local_gpu_to_kimi_fail_closed_sequence.md`
235. `20260803_145524_local_gpu_baselines_completed_kimi_stage3_launched.md`
236. `../protocol/local_gpu_baselines_completed_kimi_stage3_launched_20260803_145524.json`

The user authorized GPT-4o Stage3 first with compatible non-GPU hosted baselines in parallel,
GPU-local baselines next, and Kimi Stage3 last.  The user requested no USD budget ceiling and
declined a 4+4-GPU smoke.  The formal GPT-4o eight-GPU launcher/supervisor is prepared with 12,000
logical and at most 12,360 physical requests; USD 247.20 is only the finite ledger capacity implied
by the attempt cap, not a user budget or expected cost.  Every baseline must pass frozen behavioral
smokes before full evaluation.

The earlier shorthand that the 100-step pilots established coefficient insensitivity is corrected.
Their point estimates increase monotonically and the experiment has limited power; it detected no
statistically distinguishable paired difference but cannot exclude a beneficial larger-penalty
trend.  The user confirmed the original three-event scale-design rationale and formally froze 0.4
as an a-priori mechanistic/historical setting, not as the empirically best pilot arm.  The 300-step
extension remains cancelled.

The GPT-4o/AIGCBest client now follows the same fail-closed transport principle as the Kimi path:
only explicit transient HTTP statuses are retried; ambiguous connection/HTTP-200 body failures are
not resent; billed model/schema mismatch stops; and cache hits do not reset the outage counter.
Forty-four CUDA-free Stage3 tests pass.  Kimi retains USD 26.71782984 under its existing USD 30
aggregate gate.  A complete GPT-4o arm and hosted baseline runs still require separate paid caps.
Local PathVQA/OmniMedVQA baseline reruns are authorized but queued behind eight-GPU Stage3.

The user cancelled the 300-step coefficient extension.  The 100-step experiment is now interpreted
as evidence of local robustness over 0.3--0.5, not evidence that 0.4 is statistically optimal.  A
revised proposal would retain 0.4 as the prior historical/mechanistic setting: among the three
tested candidates it preserves four ordered per-subscale severity levels and reaches zero exactly
when all three events fail.  This justification is pending user acceptance, so the formal
coefficient remains unfrozen.

Stage3 fault tolerance is now fail-closed.  Automatic restart is whitelisted only for recognized
transient Judge transport or consecutive Judge-outage failures.  Interrupt, OOM/disk/numeric,
budget, identity/schema, source/contract, total-fallback and unknown failures stop after
conservative reservation settlement.  Cache hits no longer masquerade as remote recovery, and
the automatic recovery cap is fixed at three.  Forty-one CUDA-free Stage3 tests and historical Kimi-log
replay pass; no paid call, training or test access occurred.

The three matched GPT-4o 100-step coefficient pilots and their common PathMMU validation385 runs
are complete; entry 222 is the current result and interpretation. Accuracy was 60.78%, 61.56%
and 62.34% for penalties 0.3, 0.4 and 0.5 respectively. The point estimates are monotonic, but
paired differences are not statistically significant. The complete paid-pilot accounting,
including the conservatively settled stopped 0.5 arm, is $16.6228825 under the approved $18 gate.
This sensitivity result does not itself freeze a coefficient for the formal long run.

The complete offline 34-case and retained-rollout analysis is entry 224. It finds that 0.5's
higher validation point estimate mixes plausible corrections, clear regressions and answer-only
scoring artifacts. Training rollouts show fewer explicit contradictions at 0.5 but more
histological-definition errors. If a coefficient must be selected before more evidence, the
analysis recommends 0.4 as the conservative primary default while retaining 0.3 and 0.5 as
reported sensitivity arms; this recommendation is not a frozen formal-run decision.

The exact checkpoint downloads, SFT4000 control and common-protocol PathMMU triad are complete.
SFT3000 scored 582/999 (58.26%), SFT4000 scored 595/999 (59.56%), and selected Stage2
Outcome-GRPO scored 609/999 (60.96%). The twelve-model joint bad-case package contains 23 all-wrong
cases and 988 prediction-disagreement cases. These are pre-Stage3 development diagnostics, not
untouched final results. The next external-evaluation action is to evaluate every scientifically
compatible available local baseline and audit anomalously high
results for disclosed source-data reuse, exact test overlap, near-duplicate risk, or opaque
training data. The common PathVQA/OmniMedVQA prompt, generation, scoring and model-compatibility
contract is now frozen before inference, and its deterministic tests pass. The formal long Stage3
run remains unstarted; only bounded development pilots have run. DeepSeek-VL2 and LLaVA-Med
require separate native external-task adapter gates; UNI is
representation-only. Hosted models/APIs,
unavailable Meta Llama weights, further Stage3 Judge calls, further paid APIs and cleanup remain
separately gated.
The earlier OpenRouter audit found no live endpoints for Qwen-VL-Plus, Claude-3.5-Haiku,
Grok-4-Fast or Llama 3.2 Vision 11B/90B and no current Doubao 1.5 Vision catalog entry. Claude
Sonnet 4.5 is runnable if the manuscript's ambiguous Claude-4.5 identity is confirmed.
`openai/gpt-4o-2024-08-06` is the recommended historically aligned Stage3 judge candidate; it is
live with image and strict structured-output support, but its use still requires a frozen Stage3
protocol, external credentials and an explicit paid-call budget.
The user subsequently resolved the manuscript's ambiguous `Claude-4.5` row as Claude Haiku 4.5.
The first formal Kimi 2.6 Stage3 run stopped at step 274/1500 because an upstream chunked HTTP
response ended without its terminal chunk; no resumable checkpoint existed. Its unresolved calls
were conservatively settled, the transport was corrected without weakening fail-closed parsing,
and attempt 02 started from the same Stage2 parent with 100-step resumable saves. The two run caps
together remain exactly within the approved USD 30 ceiling. The external-VQA bad-case audit shows
both severe answer-format/scorer mismatch and genuine modality-domain errors. AIGCBest advertises
the desired vision baselines but explicitly excludes mainland-China users, so no paid request or
medical image was sent there.
Its frozen OpenRouter candidate is `anthropic/claude-haiku-4.5`; Sonnet 4.5 is not the manuscript
baseline.
The user requested an initial Stage3 run. A 50-step, seed-42, penalty-0.4 GPT-4o 2024-08-06
OpenRouter pilot with a USD 15 hard cap and validation-only post-check is now explicitly approved
and frozen. Inspection corrected the maximum Judge-call count from 800 to 400: 50 steps times 8
global completions. With accuracy, format and process, the maximum reward-audit count is 1,200.
The first credential gate found the mode-0700 secret directory but no `openrouter.env` file.
Authentication, paid calls and Stage3 therefore remain unstarted until the hidden-input/file-write
sequence is completed.
The corrected secret handoff passed: mode 0600, paid account, successful authentication and about
USD 19.64 available credit. No model call had occurred at that gate. The user has since approved
the client-side USD 15 ceiling and six-event penalty-0.4 contract. Offline tests, one paid API
smoke, eight-idle-GPU preflight, training and validation remain.
The Judge implementation now passes fifteen CUDA-free tests. Two OpenRouter smoke attempts were
rejected before provider selection: first because Azure requires `max_completion_tokens`, then
because GPT-4o training/distillation use was prohibited by the provider Terms-of-Service filter.
Both were proven unbilled (key usage remained USD 0 and account total usage remained exactly
USD 0.358321845), and their conservative reservations were audit-released. Training remains
unstarted. The client now requires OpenRouter's `enforce_distillable_text=true`; proceeding
requires a frozen distillable vision-Judge replacement or explicit compatible provider
permission. A read-only live query found 23 image+ZDR+distillable structured-output candidates.
The exact 403 cause attribution is subsequently corrected: the generic response proves a
pre-provider TOS block but does not name the firing rule. Distillation remains a material
compliance constraint for the intended online reward use, while medical content or another
provider/account permission rule remains possible. Controlled, truthfully described differential
requests are required to distinguish these causes; no restriction will be concealed or bypassed.
The user then selected the distillable multimodal `moonshotai/kimi-k2.6` Judge. The engineering
pilot completed 50 optimizer steps and saved a full checkpoint plus gathered model. The final
trajectory has exactly 1,200 reward events; mean accuracy/format/process rewards are
`0.685/1.000/0.646`. Conservative API accounting is USD `0.88919683`, actual key usage is
USD `0.59393869`, unresolved reservations are zero and the USD 15 gate was not breached. The
frozen validation385 post-check scored 243/385 (63.12%) versus the Stage2 parent's historical
241/385 (62.60%), a two-item point-estimate increase that is not a material-improvement claim.
This remains `formal_result: false`: the first and second halves used different Judge
response/evidence caps after a structured-decoding recovery, although the six events and penalty
0.4 were unchanged. A formal run must restart from the frozen parent with one Judge contract.
The user then authorized the complete Kimi-specific Stage3 validation arm with the separately
prepared USD 30 client-side hard ceiling. Its exact 1,024-token/512-character Judge contract smoke
passed, and the new 1,500-step run restarted from the frozen Stage2 parent on eight A100 GPUs. The
first optimizer step completed with finite nonzero gradients and the trainability/freeze gates
passed. This arm is currently running and remains `formal_result: false` until it terminates and is
validated; it is a contemporary Kimi comparison rather than recovery of the historical Judge.
The fixed SFT4000 control epoch-2/step-250 model-only snapshot is now uploaded to the private
Hugging Face repository `Freddie1946/PathVLM-R1-SFT-n4000-control-seed42-epoch2` at revision
`31ecd18b9dc9de5c4118efde586bb625e4a6da06`. All manifest files are present remotely, and the LFS
SHA-256 of all four safetensor shards matches the local frozen snapshot manifest.

GPT-4o Stage3 completed all 1,500 steps and the frozen validation385 pipeline selected epoch 2
(247/385). The selected model-only snapshot is independently verified at private-HF revision
`3ade3cffd46b64abc864ed9f271b47632810ec9c`. Llama 11B and DeepSeek-VL2 aggregate smokes have run:
eligible full tasks are active, while Llama 11B PathVQA is withheld after a predeclared cap-hit
gate failure. Llama 90B remains queued for all-eight-GPU execution, followed by Kimi Stage3.
