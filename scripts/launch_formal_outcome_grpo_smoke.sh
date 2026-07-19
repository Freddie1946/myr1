#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--approval-check" ]]; then
  printf '%s\n' \
    'APPROVAL SCOPE: prompt-contract-v2 one-step Outcome GRPO engineering re-gate only' \
    'PARENT: validation-selected formal n3000 epoch-3/step-1125 SFT snapshot' \
    'DATA: frozen v2 RL smoke n0008; no validation/test training access' \
    'HARDWARE: GPUs 0-7, only if all eight are idle at launch time' \
    'PROMPT: strict think/answer only; contradictory JSON-format sentence removed' \
    'GUARDS: exact hashes, parser regression, positive format reward, per-rank raw rewards,' \
    '        positive total-reward variance,' \
    '        finite nonzero gradient, saved-model load, language delta, visual equality' \
    'FAILURE: preserve artifacts and stop; no automatic retry or protocol change' \
    'LONG GRPO: not authorized by this launcher'
  exit 0
fi

REPO_ROOT="${PATHVLM_REPO:-/home/wjy/myr1}"
INSTALL_ROOT="${PATHVLM_INSTALL_ROOT:-/home/wjy/pathvlm_r1_v1_formal}"
GRPO_PYTHON="${PATHVLM_GRPO_PYTHON:-${INSTALL_ROOT}/envs/grpo/bin/python}"

exec "$GRPO_PYTHON" \
  "$REPO_ROOT/formal_machine/run_formal_outcome_grpo_smoke.py" \
  --repo-root "$REPO_ROOT" \
  --install-root "$INSTALL_ROOT" \
  "$@"
