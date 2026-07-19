# Latest recovery pointer

Latest timestamp: `20260719_174130`

Read in this order:

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
49. `../FORMAL_MACHINE_CODEX_GUIDE.md`
50. `../manuscript/README.md` and its mandatory reading order
51. `../CODEX_START_HERE.md`
52. `../manuscript/revision/reviewer_response_tracker.md`
53. `../protocol/sft_validation_curve_manifest_20260719_172547.json`
54. `../protocol/sft_validation_curve_manifest_20260719_162000.json`
55. `../protocol/sft_validation_curve_manifest_20260719_152742.json`
56. `../protocol/n3000_to_n2000_handoff_manifest_20260717_015614.json`
57. `../protocol/code_hash_manifest_20260719_172547.json`
58. `../protocol/code_hash_manifest_20260717_010146.json`
59. `../protocol/code_hash_manifest_20260717_002323.json`
60. `../protocol/sft_speed_smoke_manifest_20260716_234731.json`
61. `../protocol/sft_speed_smoke_manifest_20260716_233301.json`
62. `../protocol/sft_scale_sequence_manifest_20260715_235000.json`
63. `../protocol/code_hash_manifest_20260714_012500.json`
64. `../protocol/code_hash_manifest_20260714_005450.json`
65. `../protocol/code_hash_manifest_20260714_001041.json`
66. `../protocol/code_hash_manifest_20260713_222306.json`
67. `../protocol/code_hash_manifest_20260713_214352.json`
68. `../protocol/code_hash_manifest_20260713_213248.json`
69. `../protocol/code_hash_manifest_20260713_205956.json`
70. `../protocol/code_hash_manifest_20260713_203631.json`
71. `../protocol/code_hash_manifest_20260712_213340.json`
72. `../protocol/code_hash_manifest_20260712_192846.json`
73. `../protocol/code_hash_manifest_20260712_144811.json`
74. `../protocol/debug_proxy_model_manifest.json`
75. `../protocol/training_plan.md`
76. `../protocol/old_sft_audit.md`
77. The latest manifest under `../debug_e2e/`

Current next action: freeze and verify the v2/code/validation manifests, commit the exact-content
correction, then run only the new validation-curve preflight. The formal v2 adapter and strengthened
machine preflight already pass. A later Attempt02 must be a fresh 21-job run without reusing
Attempt01 outputs and must preserve all 8,085 raw generations and offline parser audits. Test remains
untouched. Do not start RL, resume the old scale supervisor, or launch another formal training run.
