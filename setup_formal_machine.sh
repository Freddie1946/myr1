#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-$REPO_ROOT/formal_machine.env}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config: $CONFIG_PATH" >&2
  echo "Copy formal_machine/formal_machine.env.example to formal_machine.env and edit it first." >&2
  exit 2
fi

exec bash "$REPO_ROOT/formal_machine/bootstrap_formal_machine.sh" "$CONFIG_PATH"

