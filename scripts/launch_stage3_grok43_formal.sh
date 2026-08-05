#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/home/dataset-assist-0/czy/wjy"
REPORT_ROOT="$WORKSPACE/pathvlm_revision_eval_a100/reports/aigcbest_stage3_candidate_smoke_20260805"
export PATHVLM_AIGCBEST_MODEL_ID="grok-4.3"
export PATHVLM_AIGCBEST_MODEL_RATIO="0.625"
export PATHVLM_AIGCBEST_COMPLETION_RATIO="2"
export PATHVLM_AIGCBEST_INPUT_USD_PER_MILLION="1.25"
export PATHVLM_AIGCBEST_OUTPUT_USD_PER_MILLION="2.5"
export PATHVLM_AIGCBEST_LIMIT_USD="50"
export PATHVLM_AIGCBEST_RESERVE_USD="0.01"
export PATHVLM_AIGCBEST_SMOKE_RESULT="$REPORT_ROOT/grok-4.3.json"
export PATHVLM_AIGCBEST_SMOKE_RESULT_SHA256="c7dfe402b26fc318ebe43942a0eebc6639a5e2d510ab7778964e2b1e4136483e"
export PATHVLM_AIGCBEST_STABILITY_RESULT="$REPORT_ROOT/grok-4.3-stability-gate.json"
export PATHVLM_AIGCBEST_STABILITY_RESULT_SHA256="c9881552154ffbe927ab8aced7265448629ec8fc9d6851d49b7a712b98282752"
export PATHVLM_STAGE3_RUN_CLASS="formal_stage3_grok43_via_aigcbest_seed42"
export PATHVLM_TRAIN_STATE_AUDIT_NAME="grok43_full_train_state_audit.json"
exec bash "$WORKSPACE/myr1/scripts/launch_stage3_gpt4o_formal.sh"
