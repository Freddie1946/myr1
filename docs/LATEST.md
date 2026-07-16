# Latest recovery pointer

Latest timestamp: `20260716_233301`

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
37. `../FORMAL_MACHINE_CODEX_GUIDE.md`
38. `../manuscript/README.md` and its mandatory reading order
39. `../CODEX_START_HERE.md`
40. `../protocol/sft_speed_smoke_manifest_20260716_233301.json`
41. `../protocol/sft_scale_sequence_manifest_20260715_235000.json`
42. `../protocol/code_hash_manifest_20260714_012500.json`
43. `../protocol/code_hash_manifest_20260714_005450.json`
44. `../protocol/code_hash_manifest_20260714_001041.json`
45. `../protocol/code_hash_manifest_20260713_222306.json`
46. `../protocol/code_hash_manifest_20260713_214352.json`
47. `../protocol/code_hash_manifest_20260713_213248.json`
48. `../protocol/code_hash_manifest_20260713_205956.json`
49. `../protocol/code_hash_manifest_20260713_203631.json`
50. `../protocol/code_hash_manifest_20260712_213340.json`
51. `../protocol/code_hash_manifest_20260712_192846.json`
52. `../protocol/code_hash_manifest_20260712_144811.json`
53. `../protocol/debug_proxy_model_manifest.json`
54. `../protocol/training_plan.md`
55. `../protocol/old_sft_audit.md`
56. The latest manifest under `../debug_e2e/`

Current next action: commit and run the training-only `z2_gpu_nogc_fused` eight-GPU SFT throughput
smoke. The prior CPU-offload n=2000 attempt and its scale supervisor stopped fail-closed. Do not
resume the old sequence or launch another formal scale run until the speed/memory candidate and the
validation-only duration-selection plan have been reviewed. Stage 2 long runs and Stage 3 remain
outside the authorized scope.
