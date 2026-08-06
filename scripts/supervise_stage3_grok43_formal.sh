#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/home/dataset-assist-0/czy/wjy"
export PATHVLM_STAGE3_LAUNCHER="$WORKSPACE/myr1/scripts/launch_stage3_grok43_formal.sh"
export PATHVLM_AIGCBEST_LIMIT_USD="${PATHVLM_AIGCBEST_LIMIT_USD:-50}"
export PATHVLM_AIGCBEST_RESERVE_USD="${PATHVLM_AIGCBEST_RESERVE_USD:-0.01}"
exec bash "$WORKSPACE/myr1/scripts/supervise_stage3_gpt4o_formal.sh"
