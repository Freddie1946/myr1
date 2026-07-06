# PathVLM-R1 Revision Core

This package contains the cleaned core training/evaluation code for the revision experiments.

Key decisions:
- SFT remains in LLaMA-Factory.
- PathMMU answers keep the existing format: `<answer>B) full option text</answer>`.
- Format reward checks only `<think>...</think><answer>...</answer>`.
- Accuracy reward uses exact match of extracted `<answer>` text.
- Optional GPT process reward inherits the `integrity_score + knowledge_score` interface from `evaluate_with_gpt4o_v2.py`/`ewg.py`.
- GPT reward is disabled by default. Enable with `PATHVLM_ENABLE_GPT_REWARD=true` and set `OPENAI_API_KEY`.

Main files:
- `src/pathvlm_core/grpo_pathmmu.py`: GRPO training entrypoint for PathMMU.
- `src/pathvlm_core/evaluate_qwen_pathmmu.py`: deterministic Qwen2.5-VL style evaluation.
- `src/pathvlm_core/answer_parser.py`: shared answer extraction and exact matching.
- `src/pathvlm_core/gpt_process_reward.py`: optional GPT-4o process reward interface.
- `scripts/run_grpo_pathmmu.sh`: reusable training launcher.
- `scripts/run_eval_qwen_pathmmu.sh`: reusable evaluation launcher.

Before uploading/running:
1. Confirm image-level disjoint split.
2. Set `PROJECT_ROOT`, `OPEN_R1_ROOT`, `MODEL_PATH`, data paths, output paths.
3. Run a small smoke test before full 8-GPU training.
