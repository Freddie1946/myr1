#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--approval-check" ]]; then
  printf '%s\n' \
    'APPROVAL SCOPE: Outcome GRPO engineering smoke only' \
    'PARENT: newly trained and validation-qualified formal SFT checkpoint only' \
    'DATA: frozen RL smoke n0008; no validation/test training access' \
    'GUARDS: parser regression, reward audit/variance, gradient and tensor-delta gates' \
    'LONG GRPO: not authorized by this launcher'
  exit 0
fi

printf '%s\n' \
  'Outcome GRPO smoke backend is intentionally locked until formal SFT completes and the Stage-2 gate is audited.' >&2
exit 3
