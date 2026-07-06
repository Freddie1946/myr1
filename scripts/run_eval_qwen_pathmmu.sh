#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_ROOT:?set PROJECT_ROOT to pathvlm_revision_core}"
: "${MODEL_PATH:?set MODEL_PATH}"
: "${EVAL_JSON:?set EVAL_JSON}"
: "${OUTPUT_JSON:?set OUTPUT_JSON}"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
cmd=(python "$PROJECT_ROOT/src/pathvlm_core/evaluate_qwen_pathmmu.py"
  --model-path "$MODEL_PATH"
  --data "$EVAL_JSON"
  --image-root "${IMAGE_ROOT:-}"
  --output "$OUTPUT_JSON")
if [[ -n "${LIMIT:-}" ]]; then
  cmd+=(--limit "$LIMIT")
fi
"${cmd[@]}"
