#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/formal_selected_rule_rl1000_20260811/parents/sft_step080_merged \
RUN_DIR=/home/dataset-assist-0/czy/wjy/pathvlm_r1_v1_a100/runs/formal_selected_rule_rl1000_corrected_20260811/formal_z2_mbs10_lr1e6_len384 \
MODE=formal \
RUN_ROLE=formal_corrected_reward_alignment \
PER_DEVICE_BATCH=10 \
GRADIENT_CHECKPOINTING=false \
MAX_COMPLETION_LENGTH=384 \
LEARNING_RATE=1.0e-6 \
MASTER_PORT=33741 \
bash /home/dataset-assist-0/czy/wjy/myr1/scripts/launch_formal_selected_rule_rl1000.sh
