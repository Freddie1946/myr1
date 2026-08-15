#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8765}"
ROOT="/home/dataset-assist-0/czy/wjy/pathvlm_revision_eval_a100/human_review/two_pass_expert_review_20260815"
test -f "$ROOT/index.html"
echo "Expert-review portal: http://127.0.0.1:${PORT}/"
echo "If this is a remote IDE, forward/preview TCP port ${PORT}."
exec python3 -m http.server "$PORT" --bind 0.0.0.0 --directory "$ROOT"
