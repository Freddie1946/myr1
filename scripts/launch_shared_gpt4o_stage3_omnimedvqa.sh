#!/usr/bin/env bash
set -euo pipefail
umask 077

WORK_ROOT=/home/dataset-assist-0/czy/wjy
REPO=$WORK_ROOT/myr1
INSTALL=$WORK_ROOT/pathvlm_r1_v1_a100
EVAL=$WORK_ROOT/pathvlm_revision_eval_a100
PYTHON=$INSTALL/envs/sft/bin/python
MODEL=$INSTALL/runs/stage3_process_grpo/gpt4o_full3epoch_seed42_20260801/epoch_model_snapshots/checkpoint-1000
DATA=$EVAL/datasets/external_vqa_contract_v1_20260729/omnimedvqa_four_sources_8518.json
OUTPUT=$EVAL/runs/stage3_selected_gpt4o_full_20260805_parallel_grok43/stage3_gpt4o/omnimedvqa_full8518

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false PYTHONPATH=$REPO/scripts
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

RESUME=()
[[ ! -e "$OUTPUT" ]] || RESUME=(--resume)
exec "$PYTHON" "$REPO/scripts/run_external_vqa_qwen.py" \
  --task omnimedvqa \
  --model "$MODEL" \
  --backend qwen2_5_vl \
  --data "$DATA" \
  --output-dir "$OUTPUT" \
  --split-role external_test \
  --batch-size 8 \
  "${RESUME[@]}"
