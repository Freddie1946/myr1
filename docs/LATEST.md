# Latest recovery pointer

Latest timestamp: `20260713_204927`

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
24. `../FORMAL_MACHINE_CODEX_GUIDE.md`
25. `../manuscript/README.md` and its mandatory reading order
26. `../CODEX_START_HERE.md`
27. `../protocol/code_hash_manifest_20260713_203631.json`
28. `../protocol/code_hash_manifest_20260712_213340.json`
29. `../protocol/code_hash_manifest_20260712_192846.json`
30. `../protocol/code_hash_manifest_20260712_144811.json`
31. `../protocol/debug_proxy_model_manifest.json`
32. `../protocol/training_plan.md`
33. `../protocol/old_sft_audit.md`
34. The latest manifest under `../debug_e2e/`

Current next actions: authenticate the approved PathMMU Hugging Face account, confirm the existing
GPU-0 process has exited, and run bootstrap/strengthened preflight with the prepared machine-local
configuration. Then run the audited formal SFT save/load/resume smoke and report before any long
training. Attempt04 remains pending on the old debug host; Stage 3 remains pending user agreement.
