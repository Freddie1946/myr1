# Configuration policy

Store immutable copies of the exact YAML files used by runs in this directory and copy the selected YAML into each run directory.

Suggested names:

```text
sft/sft_n3000_seed0042.yaml
sft/sft_n1000_seed0042.yaml
outcome_grpo/outcome_n1000_seed0042.yaml
process_grpo/process_n1000_seed0042.yaml
evaluation/validation_greedy.yaml
evaluation/test_locked_greedy.yaml
```

Never reuse one mutable file such as `qwen2_5_vl_full_sft.yaml` for multiple formal experiments.
