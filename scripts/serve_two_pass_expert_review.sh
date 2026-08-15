#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8765}"
ROOT="/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/human_review/two_pass_expert_review_20260815"
OUTPUT="/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/human_review/expert_submissions_20260815"
test -f "$ROOT/index.html"
echo "Expert-review portal: http://127.0.0.1:${PORT}/"
echo "Reward ratings are saved under: $OUTPUT"
echo "If this is a remote IDE, forward/preview TCP port ${PORT}."
exec python3 scripts/serve_expert_review_app.py \
  --root "$ROOT" \
  --output "$OUTPUT" \
  --host 0.0.0.0 \
  --port "$PORT"
