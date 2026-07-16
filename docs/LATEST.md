# Latest recovery pointer

Latest timestamp: `20260717_012107`

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
42. `../FORMAL_MACHINE_CODEX_GUIDE.md`
43. `../manuscript/README.md` and its mandatory reading order
44. `../CODEX_START_HERE.md`
45. `../protocol/code_hash_manifest_20260717_010146.json`
46. `../protocol/code_hash_manifest_20260717_002323.json`
47. `../protocol/sft_speed_smoke_manifest_20260716_234731.json`
48. `../protocol/sft_speed_smoke_manifest_20260716_233301.json`
49. `../protocol/sft_scale_sequence_manifest_20260715_235000.json`
50. `../protocol/code_hash_manifest_20260714_012500.json`
51. `../protocol/code_hash_manifest_20260714_005450.json`
52. `../protocol/code_hash_manifest_20260714_001041.json`
53. `../protocol/code_hash_manifest_20260713_222306.json`
54. `../protocol/code_hash_manifest_20260713_214352.json`
55. `../protocol/code_hash_manifest_20260713_213248.json`
56. `../protocol/code_hash_manifest_20260713_205956.json`
57. `../protocol/code_hash_manifest_20260713_203631.json`
58. `../protocol/code_hash_manifest_20260712_213340.json`
59. `../protocol/code_hash_manifest_20260712_192846.json`
60. `../protocol/code_hash_manifest_20260712_144811.json`
61. `../protocol/debug_proxy_model_manifest.json`
62. `../protocol/training_plan.md`
63. `../protocol/old_sft_audit.md`
64. The latest manifest under `../debug_e2e/`

Current next action: commit the successful checkpoint-retention Attempt02 record and run the
correctly labeled `z2_gpu_gc_fused` 20-step confirmation. Attempt02 passed all save, hash, independent
reload, twice-resume, rotation, trainability, freeze, backend, and test-isolation gates. Only if the
GC confirmation and projected 550-GiB reserve also pass may the fresh n=3000, seed-42 formal SFT
duration sweep launch. Do not resume the old scale supervisor. Stage 2 long runs and Stage 3 remain
outside the authorized scope.
